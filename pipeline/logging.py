from __future__ import annotations

import time

from pipeline.config import N_PIPELINE_STAGES


def log_progress(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def log_stage(stage: int, message: str, *, enabled: bool = True) -> float:
    log_progress(f"Stage {stage}/{N_PIPELINE_STAGES}: {message}", enabled=enabled)
    return time.perf_counter()


def log_stage_done(started_at: float, message: str, *, enabled: bool = True) -> None:
    elapsed = time.perf_counter() - started_at
    log_progress(f"Done in {elapsed:.1f}s: {message}", enabled=enabled)
