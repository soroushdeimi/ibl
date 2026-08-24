"""Behavioural / video covariates for Stage 3 confound control.

Pulled at extraction time (not later) because these live in separate datasets
from the trials table, and a session whose DLC failed should be discovered now
rather than after the decoding analysis is already built on it.

Each covariate is reduced to a per-trial scalar over the same pre-stimulus window
used for the spike counts, so it can be dropped straight into a nuisance-regressor
matrix aligned with the neural data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config

COVARIATE_COLUMNS: tuple[str, ...] = (
    "wheel_speed_mean", "wheel_velocity_mean", "whisker_me_mean", "pupil_diameter_mean",
)

# Wheel encoder samples are event-driven (a sample only when the wheel moves), so
# velocity has to be taken on a uniformly resampled trace; 1 kHz is IBL's own
# default in brainbox.
WHEEL_RESAMPLE_HZ: float = 1000.0


def _window_mean(
    timestamps: np.ndarray,
    values: np.ndarray,
    events: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    """Mean of `values` inside `window` around each event; NaN where no samples."""
    out = np.full(events.size, np.nan, dtype=float)
    if timestamps.size == 0 or values.size == 0:
        return out

    order = np.argsort(timestamps, kind="stable")
    ts, vals = timestamps[order], values[order]
    finite_events = np.isfinite(events)
    starts = np.searchsorted(ts, events[finite_events] + window[0], side="left")
    ends = np.searchsorted(ts, events[finite_events] + window[1], side="right")

    # Cumulative sums turn each window into two lookups instead of a slice copy.
    valid_vals = np.nan_to_num(vals, nan=0.0)
    counts = np.cumsum(np.isfinite(vals).astype(float))
    sums = np.cumsum(valid_vals)
    n = counts[ends - 1] - np.where(starts > 0, counts[starts - 1], 0.0)
    s = sums[ends - 1] - np.where(starts > 0, sums[starts - 1], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(n > 0, s / n, np.nan)
    out[finite_events] = np.where(ends > starts, means, np.nan)
    return out


def _wheel_velocity(one: Any, eid: str) -> tuple[np.ndarray, np.ndarray]:
    from brainbox.behavior.wheel import interpolate_position, velocity_filtered

    wheel = one.load_object(eid, "wheel")
    position = np.asarray(wheel["position"], dtype=float)
    timestamps = np.asarray(wheel["timestamps"], dtype=float)
    if position.size < 2:
        raise ValueError("wheel trace too short")
    # interpolate_position returns (position, time) in that order -- taking it
    # the other way round silently differentiates the time axis and yields a
    # constant 1.0 "velocity", which is why this order is asserted in
    # sanity_check.check_wheel_velocity_against_one.
    pos_uniform, ts_uniform = interpolate_position(
        timestamps, position, freq=WHEEL_RESAMPLE_HZ)
    velocity, _ = velocity_filtered(pos_uniform, WHEEL_RESAMPLE_HZ)
    return np.asarray(ts_uniform, dtype=float), np.asarray(velocity, dtype=float)


def extract_covariates(
    one: Any,
    eid: str,
    event_times: np.ndarray,
    window: tuple[float, float] = config.PRESTIM_WINDOW_S,
    camera: str = "leftCamera",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Per-trial covariates plus a per-covariate status map.

    Never raises for a missing covariate: a session with no DLC is still usable
    for the neural analysis, so the failure is recorded in `status` (and reported
    in the pipeline doc) instead of discarding the session.
    """
    events = np.asarray(event_times, dtype=float)
    frame = pd.DataFrame({c: np.full(events.size, np.nan) for c in COVARIATE_COLUMNS})
    status: dict[str, str] = {}

    try:
        ts, velocity = _wheel_velocity(one, eid)
        frame["wheel_velocity_mean"] = _window_mean(ts, velocity, events, window)
        frame["wheel_speed_mean"] = _window_mean(ts, np.abs(velocity), events, window)
        status["wheel"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["wheel"] = f"missing ({type(exc).__name__}: {exc})"

    try:
        cam = one.load_object(eid, camera, attribute=["times", "features", "ROIMotionEnergy"])
        cam_times = np.asarray(cam["times"], dtype=float)

        if "ROIMotionEnergy" in cam:
            me = np.asarray(cam["ROIMotionEnergy"], dtype=float)
            # The left-camera ROI motion energy is IBL's whisker-pad ROI.
            n = min(me.size, cam_times.size)
            frame["whisker_me_mean"] = _window_mean(cam_times[:n], me[:n], events, window)
            status["whisker_me"] = "ok"
        else:
            status["whisker_me"] = "missing (no ROIMotionEnergy dataset)"

        features = cam.get("features")
        if features is not None and "pupilDiameter_smooth" in getattr(features, "columns", []):
            # Smoothed diameter: the raw trace carries DLC frame-dropout spikes.
            pupil = np.asarray(features["pupilDiameter_smooth"], dtype=float)
            n = min(pupil.size, cam_times.size)
            frame["pupil_diameter_mean"] = _window_mean(cam_times[:n], pupil[:n], events, window)
            n_valid = int(np.isfinite(frame["pupil_diameter_mean"]).sum())
            status["pupil"] = "ok" if n_valid else "present but all-NaN (DLC pupil fit failed)"
        else:
            status["pupil"] = "missing (no camera features dataset)"
    except Exception as exc:  # noqa: BLE001
        status.setdefault("whisker_me", f"missing ({type(exc).__name__}: {exc})")
        status.setdefault("pupil", f"missing ({type(exc).__name__}: {exc})")

    return frame, status
