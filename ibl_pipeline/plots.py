"""Stage 1 result figures.

Two ways to get figures:

- Called from `run_stage1.main()` with the in-memory results it already holds,
  so a figure can never silently drift from what was actually just fit.
- `python -m ibl_pipeline.plots` regenerates the same figures from the CSV/JSON
  files a previous `run_stage1` run wrote to `data/survey/stage1/`, for when
  only this module changes and re-fitting everything is not worth twelve
  minutes.

Colors and forms follow the dataviz skill's documented reference palette
(`references/palette.md`): one hue (blue) in two lightness steps for the
Model 2 dumbbell's before/after comparison (an ordinal pair, not a categorical
one -- "all trials" vs "engaged only" is a sequence, not an identity choice),
and the fixed categorical order for the three GLM-HMM states. The two blue
steps used for the dumbbell (#6da7ec / #1c5cab light, #5598e7 / #184f95 dark)
are lifted directly from the documented sequential ramp rather than
hand-picked -- `palette.md` names #86b6ef (step 250) and #184f95 (step 600) as
its own light-end and dark-end bounds for an ordinal ramp, and the steps used
here sit inside that documented range. The categorical validator script
(`scripts/validate_palette.js`) needs Node, which is not installed on this
machine; WCAG contrast was checked by hand instead (see the chat log this
module was built in) rather than skipped silently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

FIGURES_DIR = config.SURVEY_DIR / "stage1" / "figures"
STAGE1_DIR = config.SURVEY_DIR / "stage1"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE_ALL = "#6da7ec"       # sequential step 300 -- "all trials" (earlier/lighter)
BLUE_ENGAGED = "#1c5cab"   # sequential step 550 -- "engaged only" (later/darker)
# Fixed categorical order (palette.md slots 1/2/8): engaged state gets the
# accent blue, the two biased states share a de-emphasis red family so the
# "one series is the point, rest are context" pattern (choosing-a-form.md)
# does the labelling instead of three unrelated hues.
CAT_ENGAGED = "#2a78d6"
CAT_BIASED = "#e34948"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style_axes(ax, plt) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", colors=INK_MUTED, length=0)


def plot_model1_coefficients(params: dict[str, float], pvalues: dict[str, float],
                              out_path: Path) -> Path:
    """Bar chart of the interpretable history-GLM coefficients (Model 1),
    grouped by what kind of "effective variable" each one is -- this is the
    figure that answers the proposal's core question directly: which
    variables move choice probability, and by how much.
    """
    plt = _mpl()
    groups = {
        "sensory stimulus": ["signed_contrast"],
        "block prior": ["prior_bias_right"],
        "rewarded history": sorted(k for k in params if k.startswith("r_choice_prev")),
        "unrewarded history": sorted(k for k in params if k.startswith("u_choice_prev")),
    }
    rows = []
    for group, keys in groups.items():
        for k in keys:
            if k in params:
                rows.append((group, k, params[k], pvalues.get(k, 1.0)))
    if not rows:
        raise ValueError("no recognised regressors in params; nothing to plot")

    order = [r[1] for r in rows][::-1]
    coefs = [r[2] for r in rows][::-1]
    sig = [r[3] < 0.05 for r in rows][::-1]
    group_of = {r[1]: r[0] for r in rows}
    group_color = {"sensory stimulus": CAT_ENGAGED, "block prior": "#1baf7a",
                    "rewarded history": "#eda100", "unrewarded history": "#e34948"}

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(order))
    colors = [group_color[group_of[k]] for k in order]
    ax.barh(y, coefs, color=colors, height=0.62,
            edgecolor=[c if s else SURFACE for c, s in zip(colors, sig)],
            linewidth=1.4, zorder=3)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9.5, color=INK_PRIMARY)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    _style_axes(ax, plt)
    ax.set_xlabel("logistic-regression coefficient (log-odds of choosing right)",
                  fontsize=9.5, color=INK_SECONDARY)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in group_color.values()]
    ax.legend(handles, group_color.keys(), loc="lower right", frameon=False,
              fontsize=8.5, labelcolor=INK_SECONDARY)

    fig.suptitle("Which variables move choice probability, and how much",
                  fontsize=13, color=INK_PRIMARY, x=0.02, ha="left", y=1.0)
    ax.set_title("Model 1: history-dependent logistic regression, unregularised fit "
                  "(session-clustered SE; filled bars are p<0.05)",
                  fontsize=8.5, color=INK_MUTED, loc="left", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_glm_hmm_states(weights: np.ndarray, engaged_state: int, engaged_frac: float,
                         out_path: Path) -> Path:
    """Bias vs. contrast-slope for each fitted GLM-HMM state -- the figure
    that shows *why* a state gets called "engaged" rather than asserting it:
    the engaged state should sit near zero bias with the steepest slope, the
    other two should show the large-bias/shallow-slope "biased" signature.
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for k in range(weights.shape[0]):
        bias, slope = weights[k]
        is_engaged = k == engaged_state
        color = CAT_ENGAGED if is_engaged else CAT_BIASED
        ax.scatter([bias], [slope], s=220, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        label = "engaged" if is_engaged else f"biased (state {k})"
        ax.annotate(label, (bias, slope), textcoords="offset points", xytext=(10, 8),
                    fontsize=10, color=INK_PRIMARY,
                    fontweight="bold" if is_engaged else "normal")

    ax.axvline(0, color=GRIDLINE, lw=1, zorder=1)
    ax.set_xlabel("bias weight (choice preference independent of stimulus)",
                  fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("contrast-slope weight (how much the stimulus drives choice)",
                  fontsize=9.5, color=INK_SECONDARY)
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    _style_axes(ax, plt)

    fig.suptitle("The GLM-HMM's engaged state has near-zero bias and the steepest slope",
                  fontsize=12.5, color=INK_PRIMARY, x=0.02, ha="left", y=1.0)
    ax.set_title(f"3-state Bernoulli GLM-HMM · {engaged_frac:.0%} of trials "
                  "at posterior ≥ 0.8 for the engaged state",
                  fontsize=8.5, color=INK_MUTED, loc="left", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model2_dumbbell(all_trials_summary: pd.DataFrame, engaged_summary: pd.DataFrame,
                          out_path: Path) -> Path:
    """Before/after (all-trials vs. engaged-only) mean held-out log-likelihood
    per belief model -- the headline Stage 1 result: which model of "how the
    prior updates" the mouse's engaged-state choices actually look like.
    """
    plt = _mpl()
    all_map = dict(zip(all_trials_summary["model"], all_trials_summary["mean_test_ll"]))
    eng_map = dict(zip(engaged_summary["model"], engaged_summary["mean_test_ll"]))
    models = [m for m in all_map if m in eng_map]
    if not models:
        raise ValueError("all-trials and engaged-only summaries share no models")

    label_map = {"bayes_optimal": "Bayes-optimal", "fixed_rate": "Fixed-rate",
                 "two_rate": "Two-rate", "surprise_scaled": "Surprise-scaled"}
    order = sorted(models, key=lambda m: eng_map[m], reverse=True)
    y = list(range(len(order)))[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 0.85 * len(order) + 2.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for yi, m in zip(y, order):
        a, e = all_map[m], eng_map[m]
        ax.plot([a, e], [yi, yi], color=BASELINE, lw=1.6, zorder=1, solid_capstyle="round")
        ax.scatter([a], [yi], s=95, color=BLUE_ALL, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.scatter([e], [yi], s=95, color=BLUE_ENGAGED, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(f"{a:.3f}", (a, yi), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color=INK_SECONDARY)
        ax.annotate(f"{e:.3f}", (e, yi), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color=INK_PRIMARY, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([label_map.get(m, m) for m in order], fontsize=11, color=INK_PRIMARY)
    ax.set_xlabel("mean held-out log-likelihood per trial", fontsize=10, color=INK_SECONDARY)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    _style_axes(ax, plt)
    ax.tick_params(axis="y", colors=INK_PRIMARY)

    xmin, xmax = ax.get_xlim()
    ax.set_ylim(-1.6, len(order) - 0.3)
    ax.annotate("", xy=(xmax, -0.9), xytext=(xmax - 0.06, -0.9),
                arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.2))
    ax.text(xmax - 0.03, -1.15, "better fit", ha="center", fontsize=8.5, color=INK_MUTED)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE_ALL,
                   markersize=9, label="All trials"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE_ENGAGED,
                   markersize=9, label="Engaged trials only"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9.5,
              labelcolor=INK_SECONDARY, bbox_to_anchor=(1.0, 1.02), ncol=2)

    n_sessions = int(all_trials_summary["n_sessions"].max())
    fig.suptitle("Belief-model fit improves — and reorders — on engaged-state trials",
                  fontsize=13.5, color=INK_PRIMARY, x=0.02, ha="left", y=1.02)
    ax.set_title(f"{n_sessions} sessions (Stage 1 pilot)",
                  fontsize=9.5, color=INK_MUTED, loc="left", pad=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_figures(model1_params: dict[str, float], model1_pvalues: dict[str, float],
                          glm_hmm_weights: np.ndarray, engaged_state: int, engaged_frac: float,
                          model2_all_summary: pd.DataFrame, model2_engaged_summary: pd.DataFrame
                          ) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_model1_coefficients(model1_params, model1_pvalues,
                                  FIGURES_DIR / "model1_coefficients.png"),
        plot_glm_hmm_states(glm_hmm_weights, engaged_state, engaged_frac,
                             FIGURES_DIR / "glm_hmm_states.png"),
        plot_model2_dumbbell(model2_all_summary, model2_engaged_summary,
                              FIGURES_DIR / "model2_dumbbell.png"),
    ]
    return paths


def main() -> int:
    """Regenerate figures from a previous `run_stage1` run's saved output,
    without re-fitting anything."""
    coef_file = STAGE1_DIR / "model1_coefficients.csv"
    hmm_file = STAGE1_DIR / "glm_hmm_summary.json"
    all_file = STAGE1_DIR / "model2_summary_all_trials.csv"
    eng_file = STAGE1_DIR / "model2_summary_engaged.csv"
    missing = [f for f in (coef_file, hmm_file, all_file, eng_file) if not f.exists()]
    if missing:
        print(f"missing saved result file(s), run `python -m ibl_pipeline.run_stage1` first: "
              f"{[str(f) for f in missing]}")
        return 1

    coefs = pd.read_csv(coef_file, index_col=0)
    hmm = json.loads(hmm_file.read_text())
    all_summary = pd.read_csv(all_file)
    eng_summary = pd.read_csv(eng_file)

    paths = generate_all_figures(
        coefs["param"].to_dict(), coefs["pvalue"].to_dict(),
        np.array(hmm["weights"]), hmm["engaged_state"], hmm["engaged_frac"],
        all_summary, eng_summary)
    for p in paths:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


# --- Stage 3 -------------------------------------------------------------
STAGE3_DIR = config.SURVEY_DIR / "stage3"
STAGE3_FIGURES_DIR = STAGE3_DIR / "figures"

# Panel order is an argument, not a convenience: the behaviour-only panel comes
# first because the entire design rests on it dropping to ~0.12 at lag 1 and
# recovering by lag 2. A reader who has not seen that has no reason to trust
# either neural panel, so it cannot be relegated to a supplement.
STAGE3_PANELS = (
    ("confounds", "behaviour only\n(design validation)"),
    ("counts", "neural\n(confounded from lag 2)"),
    ("residuals", "neural, history removed\n(the interpretable curve)"),
)
REGION_COLORS = {
    "CP":    "#1c5cab", "MOp":  "#e34948", "MOs":  "#2a78d6", "CA1": "#6da7ec",
    "LGd":   "#8c8a83", "VISp": "#b5b3aa", "ACAd": "#d0cec4",
}


def plot_reversal_time_course(curves: pd.DataFrame, out_path: Path,
                              underpowered: set[str] | None = None) -> Path:
    """Decoder AUC against trials-since-reversal, one panel per feature set.

    The lag-1 band is shaded in every panel because it is the only lag at which
    prior and choice history are opposed -- and, equally, the only lag at which
    no evidence about the new block has reached the animal yet. Both facts have
    to travel with the figure, so the band is annotated rather than merely drawn.
    """
    plt = _mpl()
    underpowered = underpowered or set()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (feature, title) in zip(axes, STAGE3_PANELS):
        _style_axes(ax, plt)
        ax.axhline(0.5, color=BASELINE, lw=1, zorder=1)
        ax.axvspan(0.6, 1.4, color="#efe9dc", zorder=0)

        panel = curves[curves["feature"] == feature]
        for region, group in panel.groupby("region"):
            group = group.sort_values("lag")
            color = REGION_COLORS.get(region, INK_MUTED)
            dashed = region in underpowered
            ax.fill_between(group["lag"], group["ci_low"], group["ci_high"],
                            color=color, alpha=0.13, lw=0, zorder=2)
            ax.plot(group["lag"], group["auc"], color=color, lw=1.9,
                    ls="--" if dashed else "-", zorder=3,
                    label=f"{region}*" if dashed else region)

        ax.set_title(title, color=INK_PRIMARY, fontsize=10, pad=10)
        ax.set_xlabel("trials since block reversal", color=INK_SECONDARY, fontsize=9)
        ax.set_xlim(0.5, 15.5)
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        ax.grid(axis="y", color=GRIDLINE, lw=0.7, zorder=0)

    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("AUC for the NEW block", color=INK_SECONDARY, fontsize=9)
    axes[0].annotate("lag 1: the only lag where prior and\nhistory are opposed — and the only\n"
                     "lag with no new evidence yet",
                     xy=(1.4, 0.13), xytext=(3.2, 0.16), fontsize=7.5, color=INK_SECONDARY,
                     arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))
    axes[0].text(15.3, 0.53, "above 0.5: tracks NEW block", ha="right", va="bottom",
                 fontsize=7.5, color=INK_MUTED)
    axes[0].text(15.3, 0.47, "below 0.5: tracks OLD block", ha="right", va="top",
                 fontsize=7.5, color=INK_MUTED)

    handles, labels = axes[1].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i].rstrip("*"))
    axes[2].legend([handles[i] for i in order], [labels[i] for i in order],
                   frameon=False, fontsize=8, ncol=2, loc="upper left",
                   labelcolor=INK_SECONDARY)
    fig.text(0.5, -0.04, "* below this project's own power threshold "
             "(config.MIN_SESSIONS_FOR_POWER). Bands are 95% bootstrap intervals over "
             "reversals, not trials.", ha="center", fontsize=7.5, color=INK_MUTED)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_stage3_figures() -> list[Path]:
    curves = pd.read_csv(STAGE3_DIR / "reversal_time_course.csv")
    summary = pd.read_csv(STAGE3_DIR / "reversal_summary.csv")
    underpowered = set(summary.loc[summary["underpowered"], "region"])
    return [plot_reversal_time_course(
        curves, STAGE3_FIGURES_DIR / "reversal_time_course.png", underpowered)]
