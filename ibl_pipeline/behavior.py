"""Trial-level extraction and derived behavioural regressors.

The derived-column conventions here are lifted unchanged from the exploratory
notebook (`IBL_prior_choice_exploration.ipynb`, sections 5 and 7) so that the
multi-region tables share one schema with the existing KS023 extraction.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config

# Columns the trials table must expose for a session to be usable at all.
REQUIRED_TRIAL_COLUMNS: tuple[str, ...] = (
    "choice", "feedbackType", "probabilityLeft", "stimOn_times",
    "contrastLeft", "contrastRight",
)


class TrialExtractionError(RuntimeError):
    """Raised when a session's trials table is missing or unusable."""


def classify_stage(task_protocol: str | None) -> str:
    """Map an Alyx task_protocol string onto the notebook's stage labels.

    Kept identical to the notebook so `stage` means the same thing in both the
    single-subject and multi-region tables.
    """
    protocol = (task_protocol or "").lower()
    if "training" in protocol:
        return "training"
    if "biased" in protocol:
        return "biased"
    if "ephys" in protocol:
        return "ephys"
    return "other"


def load_trials(one: Any, eid: str) -> pd.DataFrame:
    """Load one session's trials table as a DataFrame.

    Guard clauses first: an absent or truncated trials object is a hard failure
    for that session, and must be reported rather than silently skipped.
    """
    try:
        trials = one.load_object(eid, "trials")
    except Exception as exc:  # noqa: BLE001
        raise TrialExtractionError(f"{eid}: trials object could not be loaded ({exc})") from exc

    if trials is None or len(getattr(trials, "keys", lambda: [])()) == 0:
        raise TrialExtractionError(f"{eid}: trials object is empty")

    frame = trials.to_df()
    missing = [c for c in REQUIRED_TRIAL_COLUMNS if c not in frame.columns]
    if missing:
        raise TrialExtractionError(f"{eid}: trials table missing columns {missing}")
    if frame.empty:
        raise TrialExtractionError(f"{eid}: trials table has zero rows")
    return frame


def add_derived_columns(frame: pd.DataFrame, task_protocol: str | None) -> pd.DataFrame:
    """Add the notebook's derived behavioural variables.

    `choice_right` uses the sign convention verified empirically in the notebook
    (§4): choice == -1 is a rightward turn, choice == +1 leftward, 0 an omission.
    """
    out = frame.copy()
    out["stage"] = classify_stage(task_protocol)
    out["signed_contrast"] = np.where(
        out["contrastRight"].notna(), out["contrastRight"],
        np.where(out["contrastLeft"].notna(), -out["contrastLeft"], np.nan),
    )
    out["choice_right"] = np.where(out["choice"] == -1, 1.0,
                                   np.where(out["choice"] == 1, 0.0, np.nan))
    out["rewarded"] = (out["feedbackType"] == 1).astype(float)
    out.loc[out["choice"] == 0, "rewarded"] = np.nan
    # Positive when the block favours a right-side stimulus, so its sign matches
    # choice_right and the regression coefficient reads as "prior pulls right".
    out["prior_bias_right"] = 0.5 - out["probabilityLeft"]
    if "response_times" in out.columns:
        out["reaction_time"] = out["response_times"] - out["stimOn_times"]
    return out


def add_history_features(
    frame: pd.DataFrame,
    lags: tuple[int, ...] = config.HISTORY_LAGS,
    group_key: str = "eid",
) -> pd.DataFrame:
    """Add lagged choice/reward regressors, shifted within session only.

    Omission trials are dropped *before* shifting, matching the notebook: a
    non-response carries no choice to carry forward, so leaving it in would
    inject NaNs into an otherwise valid history term. Grouping by session keeps
    history from leaking across a session boundary.
    """
    out = frame[frame["choice"] != 0].copy()
    if out.empty:
        return out

    sort_key = "intervals_0" if "intervals_0" in out.columns else "stimOn_times"
    out = out.sort_values(sort_key)
    grouped = out.groupby(group_key, sort=False) if group_key in out.columns else None

    for lag in lags:
        prefix = "prev" if lag == 1 else f"prev{lag}"
        choice_col, reward_col = f"{prefix}_choice_right", f"{prefix}_rewarded"
        if grouped is not None:
            out[choice_col] = grouped["choice_right"].shift(lag)
            out[reward_col] = grouped["rewarded"].shift(lag)
        else:
            out[choice_col] = out["choice_right"].shift(lag)
            out[reward_col] = out["rewarded"].shift(lag)

        # Reward-gated vs unrewarded history, as separate regressors: the
        # notebook found these two dissociate once contrast/block are controlled.
        signed = 2 * out[choice_col] - 1
        out[f"{prefix}_choice_signed"] = signed
        out[f"r_choice_prev{lag}"] = signed * out[reward_col]
        out[f"u_choice_prev{lag}"] = signed * (1 - out[reward_col])

    return out


def extract_session_behavior(one: Any, eid: str, task_protocol: str | None) -> pd.DataFrame:
    """Full trial-level extraction for one session, in the established schema."""
    frame = load_trials(one, eid)
    frame["eid"] = str(eid)
    frame["task_protocol"] = task_protocol
    frame = add_derived_columns(frame, task_protocol)
    return add_history_features(frame)
