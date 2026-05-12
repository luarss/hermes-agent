"""AutonomyProfileEnv -- Cross-Domain Agent Autonomy Profiling

Evaluates the agent across an O*NET-derived taxonomy of work domains and
complexity levels and reports its *autonomy frontier*: the maximum
complexity level :math:`k` at which the success rate stays at or above a
threshold ``H`` (default 0.8).

Inspired by AI4Work (https://arxiv.org/abs/2603.01203). The taxonomy ships
as a curated subset of O*NET Generalized Work Activities (public-domain
USDOL/ETA data). Tasks are hand-crafted JSONL records validated against
deterministic file/terminal checkers -- no LLM-as-judge in this MVP.

Run via::

    python environments/benchmarks/autonomy_profile/autonomy_profile_env.py evaluate \\
        --config environments/benchmarks/autonomy_profile/default.yaml
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pydantic import Field, field_validator

from atroposlib.envs.base import EvalHandlingEnum
from atroposlib.envs.server_handling.server_manager import APIServerConfig

from environments.agent_loop import HermesAgentLoop
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.tool_context import ToolContext

from environments.benchmarks.autonomy_profile.loader import (
    load_task_bank,
    load_taxonomy,
    slug,
)
from environments.benchmarks.autonomy_profile.scoring import (
    TaskResult,
    autonomy_frontier,
    per_cell_success_rate,
    per_domain_frontiers,
    run_check,
    run_setup_action,
)

logger = logging.getLogger(__name__)


AUTONOMY_PROFILE_SYSTEM_PROMPT = """\
You are completing a structured work task in a sandboxed environment.

## Rules

- Read the task instruction carefully and follow it exactly.
- You have a terminal and a writable filesystem. Use them.
- When you are done, simply stop calling tools. The task will be graded
  automatically by inspecting the files you produced and the commands
  you ran.
- Do NOT ask the user clarifying questions. Make a reasonable choice when
  an instruction is ambiguous and continue.
- Do NOT explain your work in a final message -- the grader does not read it.
- All required output files are mentioned in the instruction. Create them
  at the exact paths requested.
"""


# =============================================================================
# Configuration
# =============================================================================


class AutonomyProfileConfig(HermesAgentEnvConfig):
    """Configuration for the Autonomy Profile benchmark."""

    task_files: List[str] = Field(
        default_factory=lambda: [
            "tasks/computer.jsonl",
            "tasks/office_admin.jsonl",
            "tasks/business_financial.jsonl",
            "tasks/management.jsonl",
            "tasks/data_analysis.jsonl",
        ],
        description=(
            "Task bank JSONL files, resolved relative to the benchmark directory "
            "unless absolute. Each line is one task record."
        ),
    )
    taxonomy_dir: str = Field(
        default="taxonomy",
        description="Directory (relative to the benchmark) holding onet_domains.json and onet_skills.json.",
    )
    domain_filter: Optional[List[str]] = Field(
        default=None,
        description="If set, only tasks whose ``domain`` is in this list run.",
    )
    complexity_range: List[int] = Field(
        default_factory=lambda: [1, 7],
        description="Inclusive [min, max] complexity range to include.",
    )
    success_threshold: float = Field(
        default=0.8,
        description="``H`` in ``max{k | SR(k) >= H}``. Inclusive lower bound.",
    )
    task_repetitions: int = Field(
        default=1,
        description="How many times to run each task (>1 tightens SR confidence intervals).",
    )
    task_timeout: int = Field(
        default=900,
        description="Per-task wall-clock timeout in seconds (agent loop + checks).",
    )
    max_concurrent_tasks: int = Field(
        default=8,
        description="Maximum concurrent task evaluations.",
    )

    @field_validator("complexity_range")
    @classmethod
    def _validate_complexity_range(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("complexity_range must be a [min, max] pair")
        lo, hi = v
        if lo < 1 or hi < lo:
            raise ValueError("complexity_range must satisfy 1 <= min <= max")
        return v

    @field_validator("success_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("success_threshold must lie in [0.0, 1.0]")
        return v


# =============================================================================
# Main Environment
# =============================================================================


class AutonomyProfileEvalEnv(HermesAgentBaseEnv):
    """Cross-domain autonomy-profiling evaluation environment.

    The training-pipeline abstract methods (``get_next_item``, ``compute_reward``,
    ``collect_trajectories``, ``score``) are present as stubs because the base
    class declares them abstract; the eval subcommand bypasses them entirely
    and drives ``rollout_and_score_eval`` from :meth:`evaluate`.
    """

    name = "autonomy-profile"
    env_config_cls = AutonomyProfileConfig

    # ---------------------------------------------------------------------
    # config_init -- programmatic defaults (CLI users prefer default.yaml)
    # ---------------------------------------------------------------------

    @classmethod
    def config_init(cls) -> Tuple[AutonomyProfileConfig, List[APIServerConfig]]:
        env_config = AutonomyProfileConfig(
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=None,
            distribution=None,
            max_agent_turns=40,
            max_token_length=32000,
            agent_temperature=0.2,
            system_prompt=AUTONOMY_PROFILE_SYSTEM_PROMPT,
            terminal_backend="modal",
            terminal_timeout=180,
            success_threshold=0.8,
            task_repetitions=1,
            task_timeout=900,
            max_concurrent_tasks=8,
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            group_size=1,
            steps_per_eval=1,
            total_steps=1,
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            use_wandb=True,
            wandb_name="autonomy-profile",
            ensure_scores_are_not_same=False,
        )
        server_configs = [
            APIServerConfig(
                base_url="https://openrouter.ai/api/v1",
                model_name="anthropic/claude-opus-4.6",
                server_type="openai",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                health_check=False,
            )
        ]
        return env_config, server_configs

    # ---------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------

    async def setup(self) -> None:
        # Auto-extend terminal_lifetime so Modal sandboxes survive long tasks.
        lifetime = self.config.task_timeout + 120
        self.config.terminal_lifetime = lifetime
        os.environ["TERMINAL_LIFETIME_SECONDS"] = str(lifetime)

        self.domains_by_name, self.skills_by_name = load_taxonomy(self.config.taxonomy_dir)
        all_tasks = load_task_bank(
            self.config.task_files, self.domains_by_name, self.skills_by_name
        )

        lo, hi = self.config.complexity_range
        domain_filter = set(self.config.domain_filter) if self.config.domain_filter else None
        filtered = [
            t for t in all_tasks
            if lo <= t["complexity"] <= hi
            and (domain_filter is None or t["domain"] in domain_filter)
        ]
        if not filtered:
            raise RuntimeError(
                f"No tasks matched filters: domain={domain_filter}, "
                f"complexity_range=[{lo},{hi}]"
            )

        reps = max(1, int(self.config.task_repetitions))
        self.all_eval_items: List[Dict[str, Any]] = []
        for task in filtered:
            for rep in range(reps):
                expanded = dict(task)
                expanded["_rep"] = rep
                self.all_eval_items.append(expanded)

        self.iter = 0
        self.eval_metrics: List[Tuple[str, float]] = []

        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(exist_ok=True)
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._streaming_path = str(log_dir / f"samples_{run_ts}.jsonl")
        self._streaming_file = open(self._streaming_path, "w", encoding="utf-8")
        self._streaming_lock = threading.Lock()

        # Group counts for printout
        by_domain: Dict[str, int] = defaultdict(int)
        by_level: Dict[int, int] = defaultdict(int)
        for t in self.all_eval_items:
            by_domain[t["domain"]] += 1
            by_level[t["complexity"]] += 1

        print(
            f"\nAutonomy Profile eval matrix: {len(self.all_eval_items)} rollouts "
            f"({len(filtered)} unique tasks x {reps} reps)"
        )
        for domain, count in sorted(by_domain.items()):
            print(f"  domain {domain!r}: {count} rollouts")
        for level, count in sorted(by_level.items()):
            print(f"  level L{level}: {count} rollouts")
        print(f"  success threshold H = {self.config.success_threshold:.2f}")
        print(f"  streaming results to {self._streaming_path}\n")

    def _save_result(self, record: Dict[str, Any]) -> None:
        if not hasattr(self, "_streaming_file") or self._streaming_file.closed:
            return
        with self._streaming_lock:
            self._streaming_file.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )
            self._streaming_file.flush()

    # ---------------------------------------------------------------------
    # Abstract-method stubs (training pipeline -- unused in eval-only)
    # ---------------------------------------------------------------------

    async def get_next_item(self) -> Dict[str, Any]:
        item = self.all_eval_items[self.iter % len(self.all_eval_items)]
        self.iter += 1
        return item

    def format_prompt(self, item: Dict[str, Any]) -> str:
        return item["instruction"]

    async def compute_reward(self, item, result, ctx) -> float:  # noqa: D401, ARG002
        return 0.0

    async def collect_trajectories(self, item):  # noqa: ARG002
        return None, []

    async def score(self, rollout_group_data):  # noqa: ARG002
        return None

    # ---------------------------------------------------------------------
    # Per-task evaluation
    # ---------------------------------------------------------------------

    async def rollout_and_score_eval(self, eval_item: Dict[str, Any]) -> Dict[str, Any]:
        """Run setup actions, the agent loop, and graded checks for one task."""
        task_id = str(uuid.uuid4())
        rep = eval_item.get("_rep", 0)
        record_id = f"{eval_item['id']}#rep{rep}"

        from tqdm import tqdm
        tqdm.write(f"  [START] {record_id} (task_id={task_id[:8]})")
        task_start = time.time()

        checks_results: List[Dict[str, Any]] = []
        setup_results: List[Dict[str, Any]] = []
        agent_turns = 0
        finished_naturally = False
        messages: List[Dict[str, Any]] = []
        error: Optional[str] = None

        ctx: Optional[ToolContext] = None
        try:
            ctx = ToolContext(task_id)

            # ---- Setup actions ----
            for action in eval_item.get("setup", []):
                outcome = run_setup_action(ctx, action)
                setup_results.append(outcome.to_dict())
                if not outcome.passed:
                    raise RuntimeError(f"setup failed: {outcome.detail}")

            # ---- Agent loop ----
            tools, valid_names = self._resolve_tools_for_group()
            messages = [
                {"role": "system", "content": AUTONOMY_PROFILE_SYSTEM_PROMPT},
                {"role": "user", "content": self.format_prompt(eval_item)},
            ]
            agent = HermesAgentLoop(
                server=self.server,
                tool_schemas=tools,
                valid_tool_names=valid_names,
                max_turns=self.config.max_agent_turns,
                task_id=task_id,
                temperature=self.config.agent_temperature,
                max_tokens=self.config.max_token_length,
                extra_body=self.config.extra_body,
                budget_config=self.config.build_budget_config(),
            )
            agent_result = await agent.run(messages)
            messages = agent_result.messages
            agent_turns = agent_result.turns_used
            finished_naturally = agent_result.finished_naturally

            # ---- Checks ----
            only_system_and_user = all(
                msg.get("role") in {"system", "user"} for msg in messages
            )
            if agent_turns == 0 or only_system_and_user:
                logger.warning("%s: agent produced no output, scoring as fail", record_id)
                for spec in eval_item["evaluation"]["checks"]:
                    checks_results.append({
                        "name": spec.get("check", "?"),
                        "passed": False,
                        "detail": "agent produced no output",
                    })
            else:
                loop = asyncio.get_running_loop()
                checks_results = await loop.run_in_executor(
                    None,
                    self._run_checks_sync,
                    ctx,
                    eval_item["evaluation"]["checks"],
                )

        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            logger.error("%s: rollout failed: %s", record_id, exc, exc_info=True)
        finally:
            if ctx is not None:
                try:
                    ctx.cleanup()
                except Exception as cleanup_exc:  # noqa: BLE001
                    logger.debug("ctx.cleanup() error for %s: %s", record_id, cleanup_exc)

        elapsed = time.time() - task_start
        passed_checks = sum(1 for c in checks_results if c.get("passed"))
        total_checks = len(checks_results)
        passed = bool(checks_results) and passed_checks == total_checks
        reward = 1.0 if passed else 0.0

        status = "PASS" if passed else ("ERROR" if error else "FAIL")
        tqdm.write(
            f"  [{status}] {record_id} checks={passed_checks}/{total_checks} "
            f"turns={agent_turns} ({elapsed:.0f}s)"
        )

        out = {
            "id": record_id,
            "task_id": eval_item["id"],
            "rep": rep,
            "domain": eval_item["domain"],
            "skills": eval_item.get("skills", []),
            "complexity": eval_item["complexity"],
            "passed": passed,
            "reward": reward,
            "checks_passed": passed_checks,
            "checks_total": total_checks,
            "checks": checks_results,
            "setup": setup_results,
            "turns_used": agent_turns,
            "finished_naturally": finished_naturally,
            "elapsed_seconds": elapsed,
            "messages": messages,
        }
        if error:
            out["error"] = error
        self._save_result(out)
        return out

    def _run_checks_sync(
        self,
        ctx: ToolContext,
        check_specs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Sync wrapper so check execution (which blocks on terminal calls) doesn't freeze the loop."""
        results: List[Dict[str, Any]] = []
        for spec in check_specs:
            outcome = run_check(ctx, spec)
            results.append(outcome.to_dict())
        return results

    async def _eval_with_timeout(self, item: Dict[str, Any]) -> Dict[str, Any]:
        record_id = f"{item['id']}#rep{item.get('_rep', 0)}"
        try:
            return await asyncio.wait_for(
                self.rollout_and_score_eval(item),
                timeout=self.config.task_timeout,
            )
        except asyncio.TimeoutError:
            from tqdm import tqdm
            tqdm.write(
                f"  [TIMEOUT] {record_id} (exceeded {self.config.task_timeout}s)"
            )
            out = {
                "id": record_id,
                "task_id": item["id"],
                "rep": item.get("_rep", 0),
                "domain": item["domain"],
                "skills": item.get("skills", []),
                "complexity": item["complexity"],
                "passed": False,
                "reward": 0.0,
                "checks_passed": 0,
                "checks_total": len(item["evaluation"]["checks"]),
                "checks": [],
                "setup": [],
                "turns_used": 0,
                "finished_naturally": False,
                "elapsed_seconds": float(self.config.task_timeout),
                "error": f"timeout ({self.config.task_timeout}s)",
            }
            self._save_result(out)
            return out

    # ---------------------------------------------------------------------
    # Evaluate -- iterate all tasks, then aggregate
    # ---------------------------------------------------------------------

    async def evaluate(self, *args, **kwargs) -> None:  # noqa: ARG002
        start_time = time.time()
        from tqdm import tqdm

        class _TqdmHandler(logging.Handler):
            def emit(self, record):  # type: ignore[override]
                try:
                    tqdm.write(self.format(record))
                except Exception:
                    self.handleError(record)

        root = logging.getLogger()
        handler = _TqdmHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.handlers = [handler]
        for noisy in ("httpx", "openai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        print(f"\n{'=' * 60}")
        print("Starting Autonomy Profile Evaluation")
        print(f"{'=' * 60}")
        print(f"  Total rollouts: {len(self.all_eval_items)}")
        print(f"  Max agent turns: {self.config.max_agent_turns}")
        print(f"  Task timeout: {self.config.task_timeout}s")
        print(f"  Concurrency: {self.config.max_concurrent_tasks}")
        print(f"  Threshold H: {self.config.success_threshold}")
        print(f"{'=' * 60}\n")

        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        async def _bounded(item: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self._eval_with_timeout(item)

        eval_tasks = [
            asyncio.ensure_future(_bounded(item)) for item in self.all_eval_items
        ]

        results: List[Dict[str, Any]] = []
        passed_count = 0
        pbar = tqdm(total=len(eval_tasks), desc="autonomy-profile", dynamic_ncols=True)
        try:
            for coro in asyncio.as_completed(eval_tasks):
                result = await coro
                results.append(result)
                if result.get("passed"):
                    passed_count += 1
                pbar.set_postfix_str(f"pass={passed_count}/{len(results)}")
                pbar.update(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            tqdm.write("\n[INTERRUPTED] cancelling pending tasks ...")
            for task in eval_tasks:
                task.cancel()
            await asyncio.gather(*eval_tasks, return_exceptions=True)
            try:
                from tools.terminal_tool import cleanup_all_environments
                cleanup_all_environments()
            except Exception:
                pass
            if hasattr(self, "_streaming_file") and not self._streaming_file.closed:
                self._streaming_file.close()
            pbar.close()
            return
        finally:
            pbar.close()

        end_time = time.time()
        if not results:
            print("Warning: no results recorded.")
            return

        # ----- Aggregate -----
        task_results = [
            TaskResult(
                task_id=r["id"],
                domain=r["domain"],
                complexity=r["complexity"],
                passed=bool(r.get("passed")),
            )
            for r in results
        ]
        threshold = self.config.success_threshold
        overall_frontier = autonomy_frontier(task_results, threshold)
        domain_frontiers = per_domain_frontiers(task_results, threshold)
        cell_sr = per_cell_success_rate(task_results)

        total = len(results)
        passed = sum(1 for r in results if r.get("passed"))
        eval_metrics: Dict[str, float] = {
            "eval/total_tasks": total,
            "eval/passed_tasks": passed,
            "eval/pass_rate": passed / total if total else 0.0,
            "eval/autonomy_frontier_overall": float(overall_frontier),
            "eval/evaluation_time_seconds": end_time - start_time,
        }
        for domain, frontier in domain_frontiers.items():
            key = slug(domain)
            eval_metrics[f"eval/autonomy_frontier_{key}"] = float(frontier)
        for (domain, level), sr in cell_sr.items():
            key = slug(domain)
            eval_metrics[f"eval/success_rate_{key}_L{level}"] = sr

        self.eval_metrics = list(eval_metrics.items())

        # ----- Print summary -----
        print(f"\n{'=' * 60}")
        print("Autonomy Profile Results")
        print(f"{'=' * 60}")
        print(f"Total rollouts: {total}  passed: {passed} ({passed / total:.1%})")
        print(f"Overall autonomy frontier (H={threshold:.2f}): L{overall_frontier}")
        print("\nPer-domain frontier:")
        for domain in sorted(domain_frontiers):
            print(f"  {domain}: L{domain_frontiers[domain]}")
        print("\nPer-cell success rate:")
        for (domain, level) in sorted(cell_sr):
            print(f"  {domain} L{level}: {cell_sr[(domain, level)]:.2f}")
        print(f"{'=' * 60}\n")

        # ----- Log to evaluate_log / wandb -----
        samples = [
            {k: v for k, v in r.items() if k not in {"messages"}} for r in results
        ]
        try:
            await self.evaluate_log(
                metrics=eval_metrics,
                samples=samples,
                start_time=start_time,
                end_time=end_time,
                generation_parameters={
                    "temperature": self.config.agent_temperature,
                    "max_tokens": self.config.max_token_length,
                    "max_agent_turns": self.config.max_agent_turns,
                    "success_threshold": threshold,
                },
            )
        except Exception as exc:  # noqa: BLE001
            print(f"evaluate_log failed: {exc}")

        # ----- Cleanup -----
        if hasattr(self, "_streaming_file") and not self._streaming_file.closed:
            self._streaming_file.close()
            print(f"Results saved to: {self._streaming_path}")
        try:
            from tools.terminal_tool import cleanup_all_environments
            cleanup_all_environments()
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Wandb metric forwarding
    # ---------------------------------------------------------------------

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, Any]] = None) -> None:
        if wandb_metrics is None:
            wandb_metrics = {}
        for key, value in self.eval_metrics:
            wandb_metrics[key] = value
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    AutonomyProfileEvalEnv.cli()
