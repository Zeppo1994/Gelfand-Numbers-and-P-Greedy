#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runs_dir="$repo_dir/runs"
mkdir -p "$runs_dir"

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [existing-run-directory]" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    run_dir="$(realpath -m -- "$1")"
else
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    run_dir="$runs_dir/manuscript_full_$timestamp"
fi
mkdir -p "$run_dir"

run_label="$(basename -- "$run_dir")"
session_name="manuscript_${run_label//[^[:alnum:]_]/_}"
session_file="$run_dir/tmux_session"
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "run is already active in tmux session $session_name: $run_dir" >&2
    exit 1
fi
printf '%s\n' "$session_name" > "$session_file"

conda_executable="$(command -v conda || true)"
if [[ -z "$conda_executable" ]]; then
    echo "conda was not found on PATH" >&2
    exit 1
fi

printf '%s\n' "$run_dir" > "$runs_dir/latest_full_run.txt.tmp"
mv -f "$runs_dir/latest_full_run.txt.tmp" "$runs_dir/latest_full_run.txt"

{
    printf '[%s] launcher PID: %s\n' "$(date -u +%FT%TZ)" "$$"
    printf '[%s] repository: %s\n' "$(date -u +%FT%TZ)" "$repo_dir"
    printf '[%s] run directory: %s\n' "$(date -u +%FT%TZ)" "$run_dir"
    printf '[%s] conda executable: %s\n' "$(date -u +%FT%TZ)" "$conda_executable"
    printf '[%s] tmux session: %s\n' "$(date -u +%FT%TZ)" "$session_name"
} >> "$run_dir/run.log"

printf -v conda_command '%q ' \
    "$conda_executable" run --no-capture-output -n PyTorch \
    python -u "$repo_dir/run_manuscript_figures.py" \
    --output-dir "$run_dir" --resume
printf -v worker_command \
    'set -o pipefail; %s 2>&1 | tee -a %q' \
    "$conda_command" "$run_dir/run.log"
tmux new-session -d -s "$session_name" "$worker_command"

echo "started manuscript run"
echo "  directory: $run_dir"
echo "  tmux:      $session_name"
echo "  inspect:   $repo_dir/manuscript_run_status.sh \"$run_dir\""
echo "  attach:    tmux attach-session -t \"$session_name\""
