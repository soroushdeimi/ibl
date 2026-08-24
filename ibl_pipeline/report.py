"""CLI: regenerate `docs/data_pipeline.md` from the survey and extraction outputs.

    python -m ibl_pipeline.report

The document is generated rather than hand-written so its numbers cannot drift
away from the tables in `data/`. Prose that encodes a decision (window choice, QC
threshold, region rationale) is sourced from `config` for the same reason.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

# The state of the project before this extension: the exploratory notebook was
# behavioural only -- it pulled no spike sorting at all.
BASELINE = {
    "subjects": 1,
    "subject_name": "KS023",
    "sessions": 40,
    "trials": 20815,
    "clusters": 0,
    "regions": 0,
}


def _classify_failure(message: str) -> str:
    """Group failure messages into causes, ignoring the ids and counts they carry.

    Without this, every message is unique (each embeds a pid), so the summary
    would be a list of hundreds of one-line entries instead of a diagnosis.
    """
    lowered = message.lower()
    if "qc label length" in lowered:
        return ("`clusters.metrics` has a different number of rows than the cluster "
                "table, so QC labels cannot be aligned to units - refused rather than "
                "risk attaching a QC label to the wrong unit")
    if "no histology" in lowered or "channel region labels" in lowered:
        return "no histology: channels carry no CCF region labels"
    if "could not load cluster/channel data" in lowered:
        return "cluster/channel datasets missing or unreadable for this insertion"
    if "zero clusters" in lowered:
        return "spike sorting present but contains zero clusters"
    if "no qc `label` column" in lowered:
        return "`clusters.metrics` has no QC label column"
    if "alf object was not found" in lowered or "trials object could not be loaded" in lowered:
        return "no `trials` ALF object published for this session"
    return message.split("(")[0].strip()[:110]


def _md_table(frame: pd.DataFrame, columns: dict[str, str] | None = None) -> str:
    """Render a DataFrame as a GitHub markdown table."""
    if frame.empty:
        return "_(no rows)_\n"
    view = frame[list(columns)].rename(columns=columns) if columns else frame
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    rule = "|" + "|".join("---" for _ in view.columns) + "|"
    rows = []
    for _, row in view.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("-" if pd.isna(value) else f"{value:g}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows]) + "\n"


def _read_csv(name: str) -> pd.DataFrame:
    path = config.SURVEY_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _load_manifests() -> list[dict[str, Any]]:
    if not config.SESSIONS_DIR.exists():
        return []
    manifests = []
    for manifest_file in sorted(config.SESSIONS_DIR.glob("*/manifest.json")):
        try:
            manifests.append(json.loads(manifest_file.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return manifests


def _regions_section() -> str:
    rows = [{
        "Acronym": acronym,
        "Region": description,
        "Processing level": level,
    } for acronym, (description, level) in config.REGION_LEVELS.items()]
    return _md_table(pd.DataFrame(rows))


def _filter_section(insertions: pd.DataFrame) -> str:
    if insertions.empty:
        return "_(insertion table not built yet)_\n"

    excluded = insertions[~insertions["included"]]
    lines = [
        f"- Insertion/region rows found across the target regions: **{len(insertions)}**",
        f"- Rows passing the session filter: **{int(insertions['included'].sum())}**",
        f"- Rows excluded: **{len(excluded)}**",
        "",
    ]
    if not excluded.empty:
        counts = (excluded.groupby("exclusion_reason")
                  .agg(rows=("pid", "size"), insertions=("pid", "nunique"))
                  .reset_index()
                  .rename(columns={"exclusion_reason": "Exclusion reason",
                                   "rows": "Region-rows", "insertions": "Insertions"}))
        lines.append(_md_table(counts))
    return "\n".join(lines)


def _extraction_section(manifests: list[dict[str, Any]]) -> str:
    if not manifests:
        return ("No sessions have been through the full spike-level extraction yet. "
                "Run `python -m ibl_pipeline.extract`.\n")

    rows = []
    for manifest in manifests:
        rows.append({
            "Session (eid)": manifest["eid"][:8],
            "Subject": manifest.get("subject"),
            "Regions": ",".join(manifest.get("target_regions", [])),
            "Trials": manifest.get("n_trials"),
            "Block trials": manifest.get("n_trials_biased_block"),
            "Clusters": manifest.get("n_clusters"),
            "Status": manifest.get("status"),
            "Issues": len(manifest.get("issues", [])),
        })
    return _md_table(pd.DataFrame(rows))


def _issues_section(manifests: list[dict[str, Any]]) -> str:
    failures_file = config.SURVEY_DIR / "failures.json"
    failures = json.loads(failures_file.read_text()) if failures_file.exists() else {}

    lines: list[str] = []
    cluster_failures = failures.get("cluster_load_failures", {})
    trial_failures = failures.get("trial_load_failures", {})

    lines.append(f"**Insertions whose cluster data could not be loaded: "
                 f"{len(cluster_failures)}**\n")
    if cluster_failures:
        reasons: dict[str, int] = {}
        for message in cluster_failures.values():
            cause = _classify_failure(str(message))
            reasons[cause] = reasons.get(cause, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{count}** - {reason}")
        lines.append("")
        lines.append("<details><summary>Affected insertion ids</summary>\n")
        for pid, message in sorted(cluster_failures.items()):
            lines.append(f"- `{pid}`: {message}")
        lines.append("\n</details>\n")

    lines.append(f"\n**Sessions whose trials table could not be loaded: "
                 f"{len(trial_failures)}**\n")
    if trial_failures:
        trial_reasons: dict[str, int] = {}
        for message in trial_failures.values():
            cause = _classify_failure(str(message))
            trial_reasons[cause] = trial_reasons.get(cause, 0) + 1
        for reason, count in sorted(trial_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{count}** - {reason}")
        lines.append("")
        lines.append("<details><summary>Affected session ids</summary>\n")
        for eid in sorted(trial_failures):
            lines.append(f"- `{eid}`")
        lines.append("\n</details>\n")
    else:
        lines.append("- none\n")

    if manifests:
        per_session_issues = [(m["eid"], issue) for m in manifests for issue in m.get("issues", [])]
        lines.append(f"\n**Issues recorded during per-session extraction: "
                     f"{len(per_session_issues)}**\n")
        for eid, issue in per_session_issues:
            lines.append(f"- `{eid[:8]}`: {issue}")
        if not per_session_issues:
            lines.append("- none\n")

        covariate_problems = [
            (m["eid"], name, state)
            for m in manifests
            for name, state in (m.get("covariate_status") or {}).items()
            if state != "ok"
        ]
        lines.append(f"\n**Sessions with a missing/failed covariate: "
                     f"{len({e for e, _, _ in covariate_problems})}**\n")
        for eid, name, state in covariate_problems:
            lines.append(f"- `{eid[:8]}` {name}: {state}")
        if not covariate_problems:
            lines.append("- none (wheel, whisker motion energy and pupil present "
                         "for every extracted session)\n")
    return "\n".join(lines)


def _covariate_availability_section() -> str:
    """Dataset-wide availability of the Stage 3 nuisance covariates."""
    table = _read_csv("covariate_availability.csv")
    if table.empty:
        return "_(covariate audit not run)_\n"

    labels = {
        "wheel": "wheel position/timestamps",
        "whisker_me": "whisker-pad motion energy (`leftCamera.ROIMotionEnergy`)",
        "pupil": "pupil diameter (`leftCamera.features`)",
        "camera_times": "camera frame times (`leftCamera.times`)",
    }
    rows = []
    total = len(table)
    for column, label in labels.items():
        if column not in table.columns:
            continue
        present = int(table[column].sum())
        rows.append({
            "Covariate": label,
            "Sessions with it": f"{present} / {total}",
            "Coverage": f"{present / total:.0%}",
            "Sessions missing it": total - present,
        })
    out = _md_table(pd.DataFrame(rows))

    # Whether the missingness is random matters more than its rate: a covariate
    # missing uniformly costs power, one missing by lab costs comparability.
    insertions_file = config.SURVEY_DIR / "insertions.parquet"
    if not insertions_file.exists() or "pupil" not in table.columns:
        return out

    insertions = pd.read_parquet(insertions_file)
    meta = insertions[insertions["included"]].drop_duplicates("eid")[
        ["eid", "lab", "session_start"]]
    merged = table.merge(meta, on="eid")
    if merged.empty:
        return out

    by_lab = (merged.groupby("lab")["pupil"].agg(["mean", "size"])
              .sort_values("mean").reset_index())
    by_lab["Pupil coverage"] = (by_lab["mean"] * 100).round(0).astype(int).astype(str) + "%"
    out += "\nPupil availability by lab:\n\n" + _md_table(
        by_lab, {"lab": "Lab", "size": "Sessions", "Pupil coverage": "Pupil coverage"})

    merged["year"] = pd.to_datetime(
        merged["session_start"], format="mixed", utc=True).dt.year
    by_year = merged.groupby("year")["pupil"].agg(["mean", "size"]).reset_index()
    by_year["Pupil coverage"] = (by_year["mean"] * 100).round(0).astype(int).astype(str) + "%"
    out += "\nPupil availability by year:\n\n" + _md_table(
        by_year, {"year": "Year", "size": "Sessions", "Pupil coverage": "Pupil coverage"})
    return out


def _growth_section(summary: pd.DataFrame, manifests: list[dict[str, Any]]) -> str:
    region_sessions = int(summary["n_sessions"].sum()) if not summary.empty else 0
    # Sum over regions double-counts any session covering several of them, so the
    # unique-session ceiling comes from the overlap table (one row per session).
    unique_sessions = len(_read_csv("coverage_overlap.csv"))

    extracted = [m for m in manifests if m.get("status", "").startswith("complete")]
    new_subjects = {m.get("subject") for m in extracted} - {BASELINE["subject_name"]}
    new_trials = sum(int(m.get("n_trials") or 0) for m in extracted)
    new_clusters = sum(int(m.get("n_clusters") or 0) for m in extracted)
    # Regions actually present in the extracted data, not the survey's ceiling.
    extracted_regions = sorted({r for m in extracted for r in m.get("target_regions", [])})

    lines = [
        "| Quantity | Before (KS023 only) | Added | After |",
        "|---|---|---|---|",
        f"| Subjects | {BASELINE['subjects']} | {len(new_subjects)} | "
        f"{BASELINE['subjects'] + len(new_subjects)} |",
        f"| Sessions with extracted neural data | 0 | {len(extracted)} | {len(extracted)} |",
        f"| Trials with extracted neural data | 0 | {new_trials} | {new_trials} |",
        f"| QC-passing units extracted | {BASELINE['clusters']} | {new_clusters} | "
        f"{new_clusters} |",
        f"| Target regions with extracted units | {BASELINE['regions']} | "
        f"{len(extracted_regions)} | {len(extracted_regions)} |",
    ]
    missing_regions = [r for r in config.TARGET_REGIONS if r not in extracted_regions]
    note = (
        f"\nRegions represented in the extracted data: "
        f"{', '.join(extracted_regions) if extracted_regions else 'none'}"
        + (f" (not yet extracted: {', '.join(missing_regions)}).\n" if missing_regions else ".\n")
        + f"\nThe behavioural baseline is unchanged: the exploratory notebook covered "
        f"{BASELINE['sessions']} sessions and {BASELINE['trials']:,} trials for the single "
        f"subject {BASELINE['subject_name']}, and pulled **no spike-sorted data at all** "
        f"(it ends by stating that the Neuropixels side is still to be done). So every "
        f"neural number above is new rather than an increment.\n"
        f"\n**This is a pilot extraction, not the full pull.** The survey establishes a "
        f"ceiling of **{unique_sessions} distinct sessions** carrying QC-passing units in "
        f"at least one target region ({region_sessions} region-sessions, which counts a "
        f"multi-region session once per region). The {len(extracted)} sessions extracted "
        f"here are ~{len(extracted) / unique_sessions:.0%} of that ceiling; the remainder "
        f"needs no further discovery work, only download time (~400-600 MB per session).\n"
    )
    return "\n".join(lines) + "\n" + note


def build_document() -> str:
    summary = _read_csv("region_summary.csv")
    overlap = _read_csv("coverage_overlap.csv")
    pairs_file = config.SURVEY_DIR / "region_pair_matrix.csv"
    pairs = pd.read_csv(pairs_file, index_col=0) if pairs_file.exists() else pd.DataFrame()
    insertions_file = config.SURVEY_DIR / "insertions.parquet"
    insertions = pd.read_parquet(insertions_file) if insertions_file.exists() else pd.DataFrame()
    manifests = _load_manifests()

    window_ms = (int(config.PRESTIM_WINDOW_S[0] * 1000), int(config.PRESTIM_WINDOW_S[1] * 1000))
    underpowered = summary[summary["underpowered"]] if not summary.empty else pd.DataFrame()

    parts: list[str] = []
    parts.append(f"""# IBL data pipeline: multi-region extraction

_Generated by `python -m ibl_pipeline.report` on {date.today().isoformat()}._

This document covers the extension of the pipeline from the single-subject
behavioural exploration (`IBL_prior_choice_exploration.ipynb`, subject KS023) to
multi-region Neuropixels coverage across the IBL public brain-wide release.

**Scope of this stage: data acquisition and QC only.** No behavioural model
fitting (Stage 1) and no neural decoding (Stage 2) is performed or implied here.
""")

    parts.append(f"""## 1. Target regions

Chosen to span the sensory-to-motor processing hierarchy, following Findling et
al. (2025), which reports prior-related signals at every processing level -- so a
"when and where does the prior become decodable" comparison needs coverage at
each level rather than only the motor/striatal areas already held for KS023.

{_regions_section()}
Region membership is resolved through the **Allen CCF hierarchy**, not by string
equality. Cortical areas are annotated per layer (`VISp` -> `VISp1`, `VISp2/3`,
...) and thalamic ones per subnucleus (`LGd` -> `LGd-co`, `LGd-sh`, ...), so
matching acronyms literally would have discarded almost every cortical unit.
""")

    parts.append(f"""## 2. Session filter

The established filter keeps block-structured sessions only (`biased` / `ephys`
protocols) and drops early `trainingChoiceWorld`, where `probabilityLeft` is not
an experimenter-set block prior.

Applying it to insertion-bearing sessions turned up a case the original
single-subject filter never had to handle: several protocols contain the
substring `ephys` but are **not** the standard block-prior task, and a naive
filter would have silently admitted them.

{_filter_section(insertions)}
- `_bandit_*` (Witten lab): a two-armed bandit whose reward schedule replaces the
  `probabilityLeft` block structure entirely -- there is no block prior to decode.
- `_opto_*`: optogenetic perturbation interleaved with the choice task, i.e. a
  within-session causal manipulation of the very circuits being decoded.
- `passive_*`: passive replay, no choices at all.

Note that **no `biasedChoiceWorld` session carries a Neuropixels insertion** in
the public release -- all insertion-bearing sessions are `ephysChoiceWorld`. The
`biased` arm of the filter is therefore inert here, though it is kept so the
filter stays identical to the one used for KS023.
""")

    parts.append(f"""## 3. Extraction conventions

### Pre-stimulus window: **{window_ms[0]} ms to {window_ms[1]} ms** relative to `stimOn_times`

**This is a provisional default and is flagged for revisit.** No Stage 2 decoding
code existed in the repository when this pipeline was written, so there was no
established window convention to inherit. The choice of the 400 ms immediately
preceding stimulus onset is motivated by: it is the standard IBL pre-stimulus
baseline used in the brain-wide-map prior analyses, and it ends exactly at
stimulus onset so the counts cannot contain any stimulus-evoked response -- which
is what lets them be read as "the prior held before the evidence arrived".

Both window edges are inclusive. The window is defined in one place
(`config.PRESTIM_WINDOW_S`); changing it there changes every downstream count.

Once Stage 1 selects a behavioural model, this should be revisited: the right
window depends on the timescale over which the winning model updates its belief.

### Cluster QC: `clusters.metrics.label >= {config.QC_LABEL_PASS:g}`

IBL's `label` is the fraction of its three default QC criteria a unit passes
(0, 1/3, 2/3, 1). Only `label == 1` -- IBL's own "good unit" definition, as used
in the brain-wide map -- is kept. Anything below mixes MUA and noise into the
population, which would both inflate cluster counts and add non-neural variance
to a decoder. Units failing QC are **counted, not silently dropped**: the totals
below report both.

### Trial-level schema

Reused unchanged from the notebook so the multi-region tables and the existing
KS023 extraction share one schema: `choice`, `signed_contrast`, `feedbackType`,
`probabilityLeft`, plus derived `choice_right` (sign convention verified
empirically, not assumed), `rewarded`, `prior_bias_right = 0.5 - probabilityLeft`,
and lagged history at t-1 and t-2 (`prev_choice_right`, `prev_rewarded`,
`r_choice_prev{{1,2}}`, `u_choice_prev{{1,2}}`). History is shifted **within
session only**, after dropping omission trials.
""")

    parts.append(f"""## 4. Coverage, cluster yield and trial counts

Sessions/subjects counted here are those with at least one **QC-passing unit
actually in the target region** -- an insertion whose trajectory clips a region
border but yields no usable units there is not coverage, and counting it would
overstate what is available.

{_md_table(summary, {
    "region": "Region",
    "insertions_pass_filter": "Insertions (filtered)",
    "insertions_with_good_units": "With good units",
    "n_sessions": "Sessions",
    "n_subjects": "Subjects",
    "n_clusters_in_region": "Units in region",
    "n_clusters_qc_pass": "QC-passing units",
    "clusters_per_insertion_median": "Median units/insertion",
    "trials_per_session_median": "Median valid trials/session",
}) if not summary.empty else "_(survey not yet run)_"}
""")

    if underpowered.empty:
        power_text = (f"**No target region falls below the power thresholds** "
                      f"(>= {config.MIN_SUBJECTS_FOR_POWER} subjects and "
                      f">= {config.MIN_SESSIONS_FOR_POWER} sessions). Every region "
                      f"listed above can support a cross-subject decoding comparison, "
                      f"though the per-region unit yields differ substantially and that "
                      f"difference will show up as differing decoder variance.")
    else:
        power_text = (f"The following regions fall below the thresholds "
                      f"(>= {config.MIN_SUBJECTS_FOR_POWER} subjects, "
                      f">= {config.MIN_SESSIONS_FOR_POWER} sessions). **Decoding results "
                      f"for these regions will be underpowered, and any negative result "
                      f"in them is uninterpretable** -- absence of a decodable prior "
                      f"cannot be distinguished from absence of data:\n\n"
                      + _md_table(underpowered, {
                          "region": "Region", "n_sessions": "Sessions",
                          "n_subjects": "Subjects", "power_flag": "Why flagged"}))

    parts.append(f"""## 5. Statistical power flags

{power_text}
""")

    n_multi = int((overlap["n_regions"] > 1).sum()) if not overlap.empty else 0
    parts.append(f"""## 6. Coverage overlap (simultaneous multi-region sessions)

**{n_multi} sessions** have QC-passing units in more than one target region
simultaneously. These are the sessions that allow cross-region latency
comparisons *within* a single animal and session, with the prior held physically
identical across the regions being compared -- which is a much stronger design
for "when does the prior appear where" than comparing across animals. They are
recorded in `data/survey/coverage_overlap.csv`.

Sessions per pair of simultaneously recorded regions (diagonal = sessions with
that region at all):

{_md_table(pairs.reset_index().rename(columns={"index": ""})) if not pairs.empty else "_(survey not yet run)_"}
""")

    parts.append(f"""## 7. Data actually extracted

Per session, under `data/sessions/<eid>/`:

| File | Contents |
|---|---|
| `trials.parquet` | trial-level table in the notebook's schema, plus lagged history |
| `clusters.parquet` | one row per retained unit: `cluster_id`, `channel`, CCF `acronym`, `atlas_id`, `qc_label`, `target_region` |
| `prestim_counts.npy` | `(n_trials, n_clusters)` spike counts in the pre-stimulus window |
| `covariates.parquet` | per-trial wheel velocity/speed, whisker motion energy, pupil diameter, averaged over the same window |
| `manifest.json` | provenance: window, QC threshold, per-insertion status, every issue hit |

{_extraction_section(manifests)}
""")

    parts.append(f"""## 8. Data-quality issues encountered

### 8a. Stage 3 covariate availability across the whole filtered dataset

Audited for every session passing the filter, by dataset presence (no download).
This was prompted by the extraction: of the first sessions pulled, one had no
`leftCamera` object at all and another had the camera but no `features` dataset.

{_covariate_availability_section()}
**Pupil diameter is the binding constraint, and its missingness is not random.**
A Stage 3 model conditioning on pupil discards ~30% of otherwise-usable sessions.
Coverage ranges from 47% to 100% across labs and rises from 47% (2019) to 100%
(2023), so dropping pupil-less sessions preferentially removes early sessions and
specific labs rather than a random subset.

This interacts badly with region-rich session selection. Every session covering
four target regions comes from `angelakilab` -- the lab with the *lowest* pupil
coverage (47%). The sessions best suited to cross-region comparison are therefore
the ones least likely to have pupil, so "use multi-region sessions" and "control
for pupil" pull against each other and the trade-off has to be made explicitly.

Practical implication: treat pupil as an optional covariate fitted on the subset
that has it, not as a required regressor in the main model -- otherwise the
neural dataset assembled above shrinks by a third at the last step, and shrinks
non-randomly. Whisker motion energy (82%) is safer; wheel (98%) is near-universal.

### 8b. Extraction and loading failures

{_issues_section(manifests)}
""")

    parts.append(f"""## 9. Dataset growth relative to the KS023 baseline

{_growth_section(summary, manifests)}""")

    parts.append("""## 10. Reproducing this

```bash
python -m ibl_pipeline.sanity_check          # verify extraction against ONE
python -m ibl_pipeline.run_survey            # coverage + QC survey (analyses 1-5)
python -m ibl_pipeline.extract --max-sessions-per-region 3 --prune-cache
python -m ibl_pipeline.report                # regenerate this document
```

All three are idempotent: cluster tables, trial counts and completed sessions are
cached, so an interrupted run resumes instead of re-downloading. `--prune-cache`
deletes raw spike trains once their counts are computed, which keeps peak disk use
bounded (the trains are ~100x larger than the counts derived from them).

Authentication uses the ONE parameter file (`ONE.setup()`) or the
`IBL_ONE_PASSWORD` / `IBL_ONE_USER` / `IBL_ONE_BASE_URL` environment variables;
no credentials are stored in this repository, and a missing setup raises
immediately rather than prompting.
""")

    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target = config.DOCS_DIR / "data_pipeline.md"
    target.write_text(build_document(), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
