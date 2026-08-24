"""Configuration for the IBL prior/choice-history pipeline.

Single source of truth for target regions, session filters, analysis windows and
on-disk layout. Everything downstream imports from here so that a convention
(e.g. the pre-stimulus window) is stated in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# --- Target regions -------------------------------------------------------
# Allen CCF acronyms spanning the sensory -> motor hierarchy. Chosen to follow
# Findling et al. 2025 (Nature), which reports prior-related coding at every
# processing level, so a decoding comparison needs coverage at every level
# rather than only the motor/striatal areas we already had for KS023.
REGION_LEVELS: Final[dict[str, tuple[str, str]]] = {
    "LGd":   ("lateral geniculate nucleus (dorsal)", "early sensory (thalamus)"),
    "VISp":  ("primary visual cortex",               "early sensory (cortex)"),
    "MOs":   ("secondary motor cortex",              "motor preparation"),
    "MOp":   ("primary motor cortex",                "motor preparation"),
    "ACAd":  ("dorsal anterior cingulate",           "high-level associative"),
    "ORBvl": ("ventrolateral orbitofrontal",         "high-level associative"),
    "CP":    ("caudoputamen (dorsal striatum)",      "decision / action"),
    # CA1 does not sit on the sensory->motor axis and is deliberately not
    # labelled as a rung of it. It is included because the IBL block prior is an
    # uncued, slowly-switching hidden state, and latent-state/context inference
    # is the hippocampal function most likely to carry it. It is also the
    # best-covered structure in the release (the "repeated site" trajectory runs
    # VISp -> CA1 -> DG -> LP), so it co-occurs with VISp in the same session.
    "CA1":   ("hippocampus CA1",                     "latent state / context inference"),
}
TARGET_REGIONS: Final[tuple[str, ...]] = tuple(REGION_LEVELS)

# --- Session (task-protocol) filter ---------------------------------------
# The established filter keeps only block-structured sessions ("biased"/"ephys")
# and drops early `trainingChoiceWorld`, where `probabilityLeft` is not a real
# experimenter-set block prior (see notebook practical issue #4).
INCLUDED_PROTOCOL_STAGES: Final[tuple[str, ...]] = ("biased", "ephys")

# Protocol substrings that superficially match the "ephys" filter but are NOT the
# standard block-prior task, and would silently contaminate a prior analysis:
#   _bandit_*  : Witten-lab two-armed bandit; reward schedule replaces the
#                probabilityLeft block structure entirely.
#   _opto_     : optogenetic perturbation interleaved with the choice task; the
#                laser is a within-session manipulation of the very circuits we
#                want to decode the prior from.
#   passive_   : passive replay, no choices at all.
# Order matters: `passive_opto_*` matches both patterns, and "passive" is the
# more informative reason to report, so it is tested first.
EXCLUDED_PROTOCOL_PATTERNS: Final[tuple[str, ...]] = ("_bandit_", "passive_", "_opto_")

# --- Analysis windows -----------------------------------------------------
# Findling, Hubert et al. 2025 (Nature): -600 to -100 ms before stimOn.
# Stops 100 ms short of onset so the counts are not sitting on preparatory
# activity at the stimulus edge. Extract schema 2 is this window; schema 1
# was the earlier -400 to 0 ms default.
PRESTIM_WINDOW_S: Final[tuple[float, float]] = (-0.600, -0.100)

# Lag depth for choice/reward history features. The exploratory notebook used
# t-1 and t-2; the Stage 1 history-GLM in the proposal (Q8, Model 1) specifies
# k=1..5, so this is now the deeper of the two and the notebook's model is the
# k<=2 restriction of the same feature set, not a different one.
HISTORY_LAGS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)

# --- Cluster QC -----------------------------------------------------------
# IBL ships a per-cluster `label` in clusters.metrics that is the fraction of the
# three default QC criteria passed (0, 1/3, 2/3, 1). label == 1 is IBL's own
# "good unit" definition used in the brain-wide map; anything below that mixes in
# MUA/noise, which would inflate cluster counts and add non-neural variance to a
# decoder.
QC_LABEL_PASS: Final[float] = 1.0

# --- Paths ----------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_ROOT: Final[Path] = Path(os.environ.get("IBL_PIPELINE_DATA", PROJECT_ROOT / "data"))
SURVEY_DIR: Final[Path] = DATA_ROOT / "survey"
SESSIONS_DIR: Final[Path] = DATA_ROOT / "sessions"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"

# ONE download cache. Kept overridable because the spike-sorting downloads are
# large and the project drive may not be the right place for them.
ONE_CACHE_DIR: Final[str | None] = os.environ.get("IBL_ONE_CACHE")

# --- Power flags ----------------------------------------------------------
# A region below either threshold cannot support a credible cross-subject
# decoding claim; flagged loudly rather than discovered after Stage 2.
MIN_SUBJECTS_FOR_POWER: Final[int] = 3
MIN_SESSIONS_FOR_POWER: Final[int] = 5

PUBLIC_ONE_BASE_URL: Final[str] = "https://openalyx.internationalbrainlab.org"
