#!/bin/bash

# Autonomy Profile Evaluation
#
# Runs the agent against a JSONL-defined task bank tagged by O*NET
# domain x skill x complexity, then reports the autonomy frontier
# per domain (max{k | SR(k) >= H}).
#
# Run from repo root:
#   bash environments/benchmarks/autonomy_profile/run_eval.sh
#
# Override model:
#   bash environments/benchmarks/autonomy_profile/run_eval.sh \
#       --openai.model_name anthropic/claude-sonnet-4.6
#
# Smoke run (one domain, low complexity):
#   bash environments/benchmarks/autonomy_profile/run_eval.sh \
#       --env.task_files '["tasks/computer.jsonl"]' \
#       --env.complexity_range '[1,3]' \
#       --env.max_concurrent_tasks 2

set -euo pipefail

mkdir -p environments/benchmarks/autonomy_profile/logs
LOG_FILE="environments/benchmarks/autonomy_profile/logs/run_$(date +%Y%m%d_%H%M%S).log"

echo "Autonomy Profile Evaluation"
echo "Log: $LOG_FILE"
echo ""

PYTHONUNBUFFERED=1 LOGLEVEL="${LOGLEVEL:-INFO}" \
  python environments/benchmarks/autonomy_profile/autonomy_profile_env.py evaluate \
  --config environments/benchmarks/autonomy_profile/default.yaml \
  "$@" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
