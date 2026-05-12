# Autonomy Profile

Cross-domain agent autonomy profiling, inspired by [AI4Work](https://arxiv.org/abs/2603.01203).

The benchmark runs the agent against a curated task bank tagged by
`(domain, skills, complexity)` and reports the agent's **autonomy
frontier**: the maximum complexity level `k` at which the success rate
stays at or above a configurable threshold `H` (default 0.8):

```
Autonomy = max{k | SR(k) >= H}
```

Unlike domain-specific benchmarks (TerminalBench2, YC-Bench), this one
measures *breadth* -- how the agent's capability frontier moves across
work domains.

## What ships in the MVP

- **5 domains** (curated O\*NET subset, public-domain US Dept of Labor data):
  - Computer & Mathematical
  - Office & Administrative
  - Business & Financial
  - Management
  - Data Analysis
- **150 hand-crafted tasks** (30 per domain, complexity levels 1-7).
- **Deterministic scoring**: file/terminal/json checkers only -- no LLM
  judge in this phase. See `scoring/checkers.py` for the registry.
- **Per-domain + overall frontier** emission to console, JSONL, and wandb.

Soft domains (Legal, Communication, Education) and LLM-as-judge scoring
are intentionally deferred to a Phase 2 PR (see "Future work" below).

## Quickstart

```bash
# Full profile
bash environments/benchmarks/autonomy_profile/run_eval.sh

# Smoke run: one domain at low complexity (cheap)
bash environments/benchmarks/autonomy_profile/run_eval.sh \
    --env.task_files '["tasks/computer.jsonl"]' \
    --env.complexity_range '[1,3]' \
    --env.max_concurrent_tasks 2

# Override the model
bash environments/benchmarks/autonomy_profile/run_eval.sh \
    --openai.model_name anthropic/claude-sonnet-4.6
```

Results stream to `environments/benchmarks/autonomy_profile/logs/samples_<ts>.jsonl`.

## Layout

```
autonomy_profile/
├── autonomy_profile_env.py     Env class (extends HermesAgentBaseEnv)
├── default.yaml                Tunable knobs
├── run_eval.sh                 Convenience wrapper
├── taxonomy/                   O*NET subset (public domain)
├── tasks/                      Hand-crafted JSONL task bank
└── scoring/
    ├── checkers.py             Reusable deterministic checks
    └── frontier.py             max{k | SR(k) >= H}
```

Tests live alongside the rest of the project at
`tests/test_autonomy_profile_*.py` -- run them with
`pytest tests/test_autonomy_profile_*.py`.

## Task schema

Each task is one JSON record on its own line:

```json
{
  "id": "computer-003-L1",
  "domain": "Computer & Mathematical",
  "skills": ["Programming"],
  "complexity": 1,
  "instruction": "Write a Python script `/workspace/say_ok.py` that prints `OK`.",
  "setup": [
    {"action": "terminal_run", "command": "mkdir -p /workspace", "expected_exit": 0}
  ],
  "evaluation": {
    "type": "checks",
    "checks": [
      {"check": "file_exists", "args": {"path": "/workspace/say_ok.py"}},
      {"check": "terminal_stdout_equals",
       "args": {"command": "python3 /workspace/say_ok.py", "expected": "OK\n"}}
    ]
  }
}
```

A task counts as `passed` only if *every* check returns `passed: true`.
Per-check breakdowns are still recorded in the JSONL output for
debugging.

### Available setup actions

| Action          | Required args                                |
|-----------------|----------------------------------------------|
| `write_file`    | `path`, `content`                            |
| `terminal_run`  | `command` (optional: `timeout`, `expected_exit`) |

### Available checkers

| Checker                   | Required args                                |
|---------------------------|----------------------------------------------|
| `file_exists`             | `path`                                       |
| `file_contains`           | `path`, `pattern` (regex; `ignore_case`)     |
| `file_has_sections`       | `path`, `sections` (list), `level` (1-6)     |
| `terminal_exit_code`      | `command`, `expected` (int)                  |
| `terminal_stdout_equals`  | `command`, `expected` (str; optional `strip`)|
| `terminal_stdout_matches` | `command`, `pattern` (regex)                 |
| `json_valid`              | `path`                                       |
| `json_has_keys`           | `path`, `keys` (dot-path supported)          |
| `numeric_close`           | `path`, `expected`, `tolerance`, `key`       |

## Adding a task

1. Pick the target domain JSONL under `tasks/` (or add a new file and
   list it in `default.yaml`'s `env.task_files`).
2. Pick a complexity level 1-7 and a stable `id` (`<domain-slug>-NNN-L<k>`).
3. Use only `domain` / `skills` values that appear in
   `taxonomy/onet_domains.json` / `taxonomy/onet_skills.json`.
4. Express the grader as a list of checkers from the registry above.
5. Run `pytest tests/test_autonomy_profile_task_bank.py` -- it lint-checks
   schema, taxonomy references, and registry coverage.

## Adding a checker

Add a function in `scoring/checkers.py`, register it in `CHECKER_REGISTRY`,
write a happy-path + failure-path unit test in
`tests/test_autonomy_profile_checkers.py`, and add a row to the
checker table above.

## Output

Each completed rollout streams to JSONL:

```json
{
  "id": "computer-003-L1#rep0",
  "task_id": "computer-003-L1",
  "rep": 0,
  "domain": "Computer & Mathematical",
  "skills": ["Programming"],
  "complexity": 1,
  "passed": true,
  "reward": 1.0,
  "checks_passed": 2,
  "checks_total": 2,
  "checks": [{"name": "file_exists", "passed": true, "detail": "..."}, ...],
  "setup": [...],
  "turns_used": 4,
  "finished_naturally": true,
  "elapsed_seconds": 31.7,
  "messages": [...]
}
```

`evaluate()` aggregates the stream into wandb metrics:

```
eval/autonomy_frontier_overall
eval/autonomy_frontier_<domain-slug>
eval/success_rate_<domain-slug>_L<k>
eval/total_tasks
eval/passed_tasks
eval/pass_rate
```

## Future work (out of MVP scope)

- LLM-as-judge for soft domains (Legal, Communication, Education).
- Full 23-domain O\*NET coverage + 41-skill heatmap.
- Temporal tracking across releases, model-comparison mode.
- Workflow induction (post-hoc complexity grading from agent trajectories).
- Publish profile results back to AI4Work's submission database.

## Attribution

Taxonomy data under `taxonomy/` is a curated subset of [O\*NET
Online](https://www.onetonline.org/) -- public-domain Generalized Work
Activities and SOC occupational groups published by the U.S. Department
of Labor / Employment and Training Administration (USDOL/ETA). O\*NET
data is in the public domain and free to use.
