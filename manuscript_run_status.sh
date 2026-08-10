#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
latest_file="$repo_dir/runs/latest_full_run.txt"

if [[ $# -gt 2 ]]; then
    echo "usage: $0 [run-directory] [log-lines]" >&2
    exit 2
fi

if [[ $# -ge 1 ]]; then
    run_dir="$(realpath -m -- "$1")"
elif [[ -s "$latest_file" ]]; then
    run_dir="$(<"$latest_file")"
else
    echo "no run directory supplied and no latest run recorded" >&2
    exit 1
fi
log_lines="${2:-60}"

echo "run directory: $run_dir"
if [[ -s "$run_dir/tmux_session" ]]; then
    session_name="$(<"$run_dir/tmux_session")"
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "tmux: running ($session_name)"
        echo "attach: tmux attach-session -t \"$session_name\""
    else
        echo "tmux: not running ($session_name)"
    fi
else
    echo "tmux: no session file"
fi
if [[ -f "$run_dir/status.json" ]]; then
    worker_pid="$(sed -n 's/^[[:space:]]*"pid": \([0-9][0-9]*\),*$/\1/p' "$run_dir/status.json")"
    if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then
        echo "worker: running (PID $worker_pid)"
    elif [[ -n "$worker_pid" ]]; then
        echo "worker: not running (recorded PID $worker_pid)"
    fi
fi

echo
echo "status:"
if [[ -f "$run_dir/status.json" ]]; then
    cat "$run_dir/status.json"
else
    echo "status.json has not been written yet"
fi

echo
echo "outputs:"
find "$run_dir" -maxdepth 1 -type f -printf '%f\t%TY-%Tm-%Td %TH:%TM:%TS\t%s bytes\n' | sort

echo
echo "last $log_lines log lines:"
if [[ -f "$run_dir/run.log" ]]; then
    tail -n "$log_lines" "$run_dir/run.log"
else
    echo "run.log has not been written yet"
fi
