"""Run the publication-scale manuscript figures sequentially and resumably."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import traceback

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent

import legendre
import matern
import paley_wiener
import periodic_mixed

LEGENDRE_OPTIONS = {
    "smoothness_values": legendre.SMOOTHNESS_VALUES,
    "sel_grid": legendre.CANDIDATE_GRID_SIZE,
}

LEGENDRE_POINT_OPTIONS = {
    "smoothness_values": legendre.SMOOTHNESS_VALUES,
    "m": legendre.POINT_DESIGN_SIZE,
    "grid": legendre.CANDIDATE_GRID_SIZE,
}

STAGES = (
    ("legendre", legendre.comparison_figure, LEGENDRE_OPTIONS),
    ("legendre_points", legendre.points_figure, LEGENDRE_POINT_OPTIONS),
    ("matern", matern.comparison_figure, {}),
    ("matern_points", matern.points_figure, {}),
    ("periodic_mixed", periodic_mixed.comparison_figure, {}),
    ("periodic_mixed_points", periodic_mixed.points_figure, {}),
    ("paley_wiener", paley_wiener.comparison_figure, {}),
    ("paley_wiener_points", paley_wiener.points_figure, {}),
)

PUBLICATION_SETTINGS = {
    "stages": [name for name, _, _ in STAGES],
    "legendre": LEGENDRE_OPTIONS,
    "legendre_points": LEGENDRE_POINT_OPTIONS,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_status(path: Path, status: dict) -> None:
    status["updated_at"] = utc_now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def initial_status(output_dir: Path) -> dict:
    return {
        "schema_version": 1,
        "state": "pending",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "current_stage": None,
        "completed_stages": [],
        "failed_stage": None,
        "error": None,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "repository": str(REPO_ROOT),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "output_dir": str(output_dir),
        "settings": PUBLICATION_SETTINGS,
        "outputs": {},
    }


def load_status(status_path: Path, output_dir: Path, resume: bool) -> dict:
    if status_path.exists():
        if not resume:
            raise RuntimeError(
                f"{status_path} already exists; pass --resume to continue it"
            )
        status = json.loads(status_path.read_text())
    else:
        status = initial_status(output_dir)
    status.update(
        state="running",
        current_stage=None,
        failed_stage=None,
        error=None,
        pid=os.getpid(),
        finished_at=None,
    )
    status["settings"] = PUBLICATION_SETTINGS
    return status


def save_array_result(output_dir: Path, stage_name: str, result) -> str | None:
    if not isinstance(result, dict):
        return None
    arrays = {
        key: np.asarray(value)
        for key, value in result.items()
        if isinstance(value, (np.ndarray, int, float, bool, np.number))
    }
    if not arrays:
        return None
    path = output_dir / f"{stage_name}_data.npz"
    np.savez_compressed(path, **arrays)
    return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages recorded complete when their output files still exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stage plan without starting publication-scale computations.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    plan = {
        "output_dir": str(output_dir),
        "environment": os.environ.get("CONDA_DEFAULT_ENV"),
        **PUBLICATION_SETTINGS,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    status = load_status(status_path, output_dir, args.resume)
    write_status(status_path, status)
    print(f"[{utc_now()}] output directory: {output_dir}", flush=True)

    for name, run_stage, options in STAGES:
        relative_output = Path("figures") / f"{name}.png"
        expected_output = output_dir / relative_output
        if name in status["completed_stages"] and expected_output.exists():
            print(f"[{utc_now()}] skipping completed stage: {name}", flush=True)
            continue

        status["state"] = "running"
        status["current_stage"] = name
        write_status(status_path, status)
        print(f"[{utc_now()}] starting stage: {name}", flush=True)
        try:
            result = run_stage(out=expected_output, **options)
            if not expected_output.exists():
                raise RuntimeError(
                    f"stage {name} did not create {expected_output.name}"
                )
            data_output = save_array_result(output_dir, name, result)
            status["outputs"][name] = {
                "figure": str(relative_output),
                "data": data_output,
            }
            if name not in status["completed_stages"]:
                status["completed_stages"].append(name)
            status["current_stage"] = None
            write_status(status_path, status)
            print(f"[{utc_now()}] completed stage: {name}", flush=True)
        except Exception as exc:
            status["state"] = "failed"
            status["failed_stage"] = name
            status["error"] = f"{type(exc).__name__}: {exc}"
            status["current_stage"] = None
            status["finished_at"] = utc_now()
            write_status(status_path, status)
            traceback.print_exc()
            return 1

    status["state"] = "completed"
    status["current_stage"] = None
    status["finished_at"] = utc_now()
    write_status(status_path, status)
    print(f"[{utc_now()}] all stages completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
