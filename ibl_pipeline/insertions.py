"""Discovery of Neuropixels insertions in the target regions.

Everything here is metadata-only (Alyx REST queries), so the full brain-wide
survey runs without downloading any electrophysiology.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import config
from .behavior import classify_stage

INSERTION_COLUMNS: tuple[str, ...] = (
    "target_region", "pid", "eid", "probe", "subject", "lab",
    "task_protocol", "stage", "session_start", "included", "exclusion_reason",
)


def protocol_verdict(task_protocol: str | None) -> tuple[bool, str]:
    """Decide whether a session's protocol passes the established filter.

    Returns (included, reason). The reason is always populated so that excluded
    sessions can be counted and reported instead of vanishing.
    """
    if not task_protocol:
        return False, "missing task_protocol"

    lowered = task_protocol.lower()
    for pattern in config.EXCLUDED_PROTOCOL_PATTERNS:
        if pattern in lowered:
            # These match "ephys" textually but are a different task; see config.
            return False, f"non-standard task variant ({pattern.strip('_')})"

    stage = classify_stage(task_protocol)
    if stage not in config.INCLUDED_PROTOCOL_STAGES:
        return False, f"stage={stage} (not biased/ephys)"
    return True, ""


def search_region_insertions(one: Any, region: str) -> pd.DataFrame:
    """List every public insertion touching `region`, annotated with its verdict."""
    pids, details = one.search_insertions(
        atlas_acronym=region, details=True, query_type="remote"
    )

    rows = []
    for pid, detail in zip(pids, details):
        info = detail.get("session_info") or {}
        protocol = info.get("task_protocol")
        included, reason = protocol_verdict(protocol)
        rows.append({
            "target_region": region,
            "pid": str(pid),
            "eid": str(detail.get("session")),
            "probe": detail.get("name"),
            "subject": info.get("subject"),
            "lab": info.get("lab"),
            "task_protocol": protocol,
            "stage": classify_stage(protocol),
            "session_start": info.get("start_time"),
            "included": included,
            "exclusion_reason": reason,
        })

    if not rows:
        return pd.DataFrame(columns=list(INSERTION_COLUMNS))
    return pd.DataFrame(rows)[list(INSERTION_COLUMNS)]


def build_insertion_table(one: Any, regions: tuple[str, ...] = config.TARGET_REGIONS) -> pd.DataFrame:
    """Insertion table across all target regions.

    One row per (region, insertion): an insertion crossing two target regions
    appears twice, which is what makes the coverage-overlap analysis possible.
    """
    frames = [search_region_insertions(one, region) for region in regions]
    return pd.concat(frames, ignore_index=True)
