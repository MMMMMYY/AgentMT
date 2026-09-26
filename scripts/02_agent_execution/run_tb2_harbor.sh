#!/usr/bin/env bash
# RQ3 Terminal-Bench runs via Harbor (local Docker). Run on the Mac.
#   bash scripts/02_agent_execution/run_tb2_harbor.sh smoke        # 1 task x source x 1 attempt, both agents
#   bash scripts/02_agent_execution/run_tb2_harbor.sh full         # 15 tasks x 9 forms x 3 attempts, both agents
#   bash scripts/02_agent_execution/run_tb2_harbor.sh claude-smoke # Claude Code only: 1 smoke trial
#   bash scripts/02_agent_execution/run_tb2_harbor.sh claude-full  # Claude Code only: 405 full trials
#   bash scripts/02_agent_execution/run_tb2_harbor.sh codex-smoke  # Codex only: 1 smoke trial
#   bash scripts/02_agent_execution/run_tb2_harbor.sh codex-full   # Codex only: 405 full trials
# Keys: ANTHROPIC_API_KEY for claude-code; OPENAI_API_KEY for codex (read from secrets/ if unset).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat secrets/openai_key.txt)}"
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f secrets/anthropic_key.txt ]; then
  export ANTHROPIC_API_KEY="$(cat secrets/anthropic_key.txt)"
fi
TASKS=dataset/terminal_bench/variant_tasks
OUT=runs/terminal_bench
JOB_TAG="${HARBOR_JOB_TAG:-}"
HARBOR_BIN="${HARBOR_BIN:-$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))')/harbor}"
if [ ! -x "$HARBOR_BIN" ]; then
  HARBOR_BIN="$(command -v harbor)"
fi
MODE="${1:-smoke}"
if [[ "$MODE" == *smoke ]]; then
  SEL=(-i "overfull-hbox__source__source"); K=1; N=1; SUFFIX=smoke
else
  SEL=(); K=3; N=4; SUFFIX=full
fi

case "$MODE" in
  smoke|full)
    "$HARBOR_BIN" run -p "$TASKS" "${SEL[@]}" -a claude-code -m anthropic/claude-haiku-4-5 \
      -k "$K" -n "$N" -o "$OUT" --job-name "claude_code_haiku45_$SUFFIX$JOB_TAG"
    "$HARBOR_BIN" run -p "$TASKS" "${SEL[@]}" -a codex -m openai/gpt-6-luna \
      -k "$K" -n "$N" -o "$OUT" --job-name "codex_gpt6luna_$SUFFIX$JOB_TAG"
    ;;
  claude-smoke|claude-full)
    if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
      echo "Missing Claude authentication: set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN." >&2
      exit 2
    fi
    "$HARBOR_BIN" run -p "$TASKS" "${SEL[@]}" -a claude-code -m anthropic/claude-haiku-4-5 \
      -k "$K" -n "$N" -o "$OUT" --job-name "claude_code_haiku45_$SUFFIX$JOB_TAG"
    ;;
  codex-smoke|codex-full)
    "$HARBOR_BIN" run -p "$TASKS" "${SEL[@]}" -a codex -m openai/gpt-6-luna \
      -k "$K" -n "$N" -o "$OUT" --job-name "codex_gpt6luna_$SUFFIX$JOB_TAG"
    ;;
  *)
    echo "Usage: $0 {smoke|full|claude-smoke|claude-full|codex-smoke|codex-full}" >&2
    exit 2
    ;;
esac
