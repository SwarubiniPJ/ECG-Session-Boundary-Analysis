#!/usr/bin/env python3
"""Two-process launcher for the Nature-oriented ECG transition and timing V4 pipeline.

The first clean Python process constructs and calibrates all boundary-level
results. After it exits, a second clean process performs participant bootstrap,
matched inference, paper tables, and figures. Keeping these stages in separate
processes prevents large scientific-library state from affecting report
post-processing on long runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _output_root(arguments: list[str], base_dir: Path) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Nature_Timing_Validated_Results_V4"),
    )
    namespace, _ = parser.parse_known_args(arguments)
    path = namespace.output_root.expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def main() -> None:
    package_root = Path(__file__).resolve().parent
    invocation_cwd = Path.cwd().resolve()
    arguments = sys.argv[1:]
    stage1 = [sys.executable, "-m", "ecg_transition_v4.run_analysis", *arguments]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(package_root)
        if not existing_pythonpath
        else str(package_root) + os.pathsep + existing_pythonpath
    )
    subprocess.run(stage1, check=True, cwd=invocation_cwd, env=environment)

    # Help mode exits after displaying the stage-1 parser.
    if "--help" in arguments or "-h" in arguments:
        return
    print("[launcher] construction process exited", flush=True)

    output_root = _output_root(arguments, invocation_cwd)
    request_path = output_root / "00_audit" / "postprocess_request.json"
    if not request_path.exists():
        raise FileNotFoundError(
            f"Construction completed without a post-processing request: {request_path}"
        )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    print(f"[launcher] loaded postprocess request from {request_path}", flush=True)
    # Give the operating system a brief interval to reclaim the construction
    # process' large scientific arrays before reloading the canonical files.
    time.sleep(2.0)
    print("[launcher] starting clean postprocess process", flush=True)
    stage2 = [
        sys.executable,
        "-m",
        "ecg_transition_v4.postprocess",
        "--output-root",
        str(request["output_root"]),
        "--bootstrap",
        str(request["bootstrap"]),
        "--permutations",
        str(request["permutations"]),
        "--seed",
        str(request["seed"]),
        "--figure-dpi",
        str(request["figure_dpi"]),
    ]
    # Replace the lightweight launcher with the post-processing process rather
    # than spawning another child. This keeps the two scientific stages fully
    # isolated and lets the invoking shell receive the final exit status.
    os.chdir(package_root)
    os.execv(sys.executable, stage2)


if __name__ == "__main__":
    main()

