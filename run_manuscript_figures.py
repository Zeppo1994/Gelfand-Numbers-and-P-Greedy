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
import paley_wiener
import periodic_mixed

COMPRESS_IRLS = False
LEGENDRE_OPTIONS = {
    "smoothness_values": legendre.SMOOTHNESS_VALUES,
    "n_trunc": legendre.MERCER_TRUNCATION,
    "sel_grid": legendre.CANDIDATE_GRID_SIZE,
    "edge_ladder": legendre.ENDPOINT_LADDER_SIZE,
}
POINT_DESIGN_OPTIONS = {
    "smoothness_values": legendre.SMOOTHNESS_VALUES,
    "m": legendre.POINT_DESIGN_SIZE,
    "grid": legendre.CANDIDATE_GRID_SIZE,
    "n_trunc": legendre.MERCER_TRUNCATION,
}


STAGES = (
    {
        "name": "legendre_points",
        "output": "figures/legendre_points.png",
        "run": lambda: legendre.points_figure(**POINT_DESIGN_OPTIONS),
    },
    {
        "name": "legendre",
        "output": "figures/legendre.png",
        "run": lambda: legendre.comparison_figure(
            compress_irls=COMPRESS_IRLS,
            **LEGENDRE_OPTIONS,
        ),
    },
    {
        "name": "periodic_mixed",
        "output": "figures/periodic_mixed.png",
        "run": lambda: periodic_mixed.rates_figure(compress_irls=COMPRESS_IRLS),
    },
    {
        "name": "paley_wiener",
        "output": "figures/paley_wiener.png",
        "run": lambda: paley_wiener.rates_figure(compress_irls=COMPRESS_IRLS),
    },
)

PUBLICATION_SETTINGS = {
    "compress_irls": COMPRESS_IRLS,
    "stages": [stage["name"] for stage in STAGES],
    "legendre": LEGENDRE_OPTIONS,
    "point_design": POINT_DESIGN_OPTIONS,
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
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "repository": str(REPO_ROOT),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "output_dir": str(output_dir),
        "settings": PUBLICATION_SETTINGS,
        "outputs": {},
    }


def save_array_result(output_dir: Path, stage_name: str, result) -> str | None:
    if not isinstance(result, dict):
        return None
    arrays = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, (int, float, bool, np.number)):
            arrays[key] = np.asarray(value)
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
    if args.resume and status_path.exists():
        status = json.loads(status_path.read_text())
        status.update(
            state="running",
            current_stage=None,
            failed_stage=None,
            error=None,
            pid=os.getpid(),
            finished_at=None,
        )
        status["settings"] = PUBLICATION_SETTINGS
    elif status_path.exists():
        raise RuntimeError(f"{status_path} already exists; pass --resume to continue it")
    else:
        status = initial_status(output_dir)
        status["state"] = "running"

    os.chdir(output_dir)
    write_status(status_path, status)
    print(f"[{utc_now()}] output directory: {output_dir}", flush=True)

    for stage in STAGES:
        name = stage["name"]
        expected_output = output_dir / stage["output"]
        if name in status["completed_stages"] and expected_output.exists():
            print(f"[{utc_now()}] skipping completed stage: {name}", flush=True)
            continue

        status["state"] = "running"
        status["current_stage"] = name
        write_status(status_path, status)
        print(f"[{utc_now()}] starting stage: {name}", flush=True)
        try:
            result = stage["run"]()
            if not expected_output.exists():
                raise RuntimeError(f"stage {name} did not create {expected_output.name}")
            data_output = save_array_result(output_dir, name, result)
            status["outputs"][name] = {
                "figure": expected_output.name,
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
