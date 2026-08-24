"""Publication figures for docs/latex/report.tex. Regenerates from saved CSVs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ibl_pipeline import plots  # noqa: E402
from ibl_pipeline.plots import (  # noqa: E402
    BASELINE,
    CAT_BIASED,
    CAT_ENGAGED,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    _style_axes,
)

OUT = Path(__file__).resolve().parent / "figures"
SURVEY = ROOT / "data" / "survey"

REGION_ORDER = ["CP", "MOp", "ORBvl", "MOs", "LGd", "ACAd", "VISp", "CA1"]
COUNTS = "#1c5cab"
RESID = "#e34948"
OLD = "#8c8a83"
CONF = "#1baf7a"


def _plt():
    return plots._mpl()


def _save(fig, plt, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_model1() -> None:
    coefs = pd.read_csv(SURVEY / "stage1" / "model1_coefficients.csv", index_col=0)
    params = coefs["param"].to_dict()
    pvalues = coefs["pvalue"].to_dict()
    spec = [
        ("signed_contrast", "signed contrast", CAT_ENGAGED),
        ("prior_bias_right", "block prior (right)", "#1baf7a"),
        ("r_choice_prev1", "rewarded, lag 1", "#eda100"),
        ("r_choice_prev2", "rewarded, lag 2", "#eda100"),
        ("r_choice_prev3", "rewarded, lag 3", "#eda100"),
        ("r_choice_prev4", "rewarded, lag 4", "#eda100"),
        ("r_choice_prev5", "rewarded, lag 5", "#eda100"),
        ("u_choice_prev1", "unrewarded, lag 1", CAT_BIASED),
        ("u_choice_prev2", "unrewarded, lag 2", CAT_BIASED),
        ("u_choice_prev3", "unrewarded, lag 3", CAT_BIASED),
        ("u_choice_prev4", "unrewarded, lag 4", CAT_BIASED),
        ("u_choice_prev5", "unrewarded, lag 5", CAT_BIASED),
    ]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    keys = [k for k, _, _ in spec if k in params][::-1]
    labels = {k: lab for k, lab, _ in spec}
    colors = {k: c for k, _, c in spec}
    y = np.arange(len(keys))
    vals = [params[k] for k in keys]
    cols = [colors[k] for k in keys]
    ax.barh(y, vals, color=cols, height=0.68, zorder=3)
    ax.axvline(0, color=BASELINE, lw=1.1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in keys], fontsize=9, color=INK_PRIMARY)
    ax.set_xlabel("coefficient (log-odds of choosing right)", fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="x", color=GRIDLINE, lw=0.8, zorder=0)
    _style_axes(ax, plt)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CAT_ENGAGED),
        plt.Rectangle((0, 0), 1, 1, color="#1baf7a"),
        plt.Rectangle((0, 0), 1, 1, color="#eda100"),
        plt.Rectangle((0, 0), 1, 1, color=CAT_BIASED),
    ]
    ax.legend(handles, ["stimulus", "block prior", "rewarded history", "unrewarded history"],
              loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, plt, "fig1_model1_coefficients")
    del pvalues


def fig_glm_hmm() -> None:
    hmm = json.loads((SURVEY / "stage1" / "glm_hmm_summary.json").read_text())
    plt = _plt()
    weights = np.array(hmm["weights"])
    engaged = hmm["engaged_state"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for k in range(weights.shape[0]):
        bias, slope = weights[k]
        is_eng = k == engaged
        ax.scatter([bias], [slope], s=200, color=CAT_ENGAGED if is_eng else CAT_BIASED,
                   zorder=3, edgecolor=SURFACE, linewidth=1.4)
        ax.annotate("engaged" if is_eng else f"biased {k}",
                    (bias, slope), textcoords="offset points", xytext=(8, 7),
                    fontsize=9, color=INK_PRIMARY, fontweight="bold" if is_eng else "normal")
    ax.axvline(0, color=GRIDLINE, lw=1, zorder=1)
    ax.set_xlabel("bias weight", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("contrast-slope weight", fontsize=9, color=INK_SECONDARY)
    ax.grid(color=GRIDLINE, lw=0.8, zorder=0)
    _style_axes(ax, plt)
    fig.tight_layout()
    _save(fig, plt, "fig2_glm_hmm_states")


def fig_belief() -> None:
    all_s = pd.read_csv(SURVEY / "stage1" / "model2_summary_all_trials.csv")
    eng_s = pd.read_csv(SURVEY / "stage1" / "model2_summary_engaged.csv")
    plt = _plt()
    all_map = dict(zip(all_s["model"], all_s["mean_test_ll"]))
    eng_map = dict(zip(eng_s["model"], eng_s["mean_test_ll"]))
    label_map = {"bayes_optimal": "Bayes-optimal", "fixed_rate": "Fixed-rate",
                 "two_rate": "Two-rate", "surprise_scaled": "Surprise-scaled"}
    order = sorted(all_map, key=lambda m: eng_map[m], reverse=True)
    y = list(range(len(order)))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for yi, m in zip(y, order):
        a, e = all_map[m], eng_map[m]
        ax.plot([a, e], [yi, yi], color=BASELINE, lw=1.6, zorder=1, solid_capstyle="round")
        ax.scatter([a], [yi], s=90, color="#6da7ec", zorder=3, edgecolor=SURFACE, lw=1.1)
        ax.scatter([e], [yi], s=90, color="#1c5cab", zorder=3, edgecolor=SURFACE, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([label_map[m] for m in order], fontsize=10, color=INK_PRIMARY)
    ax.set_xlabel("mean held-out log-likelihood per trial (higher is better)",
                  fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="x", color=GRIDLINE, lw=0.8, zorder=0)
    _style_axes(ax, plt)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#6da7ec",
                       markersize=8, label="all trials"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1c5cab",
                       markersize=8, label="engaged trials"),
        ],
        loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, plt, "fig3_belief_models")


def fig_decode_excess() -> None:
    s = pd.read_csv(SURVEY / "stage2" / "prior_decode_summary_by_region.csv")
    s = s.set_index("region").loc[REGION_ORDER]
    sess = pd.read_csv(SURVEY / "stage2" / "prior_decode_results.csv")
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.8), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    panels = [
        ("r2_excess", "r2_excess_median", "bh_q", COUNTS, "counts", (-0.08, 0.20)),
        ("r2_excess_residuals", "r2_excess_median_residuals", "bh_q_residuals",
         RESID, "residuals", (-0.08, 0.20)),
        ("r2_excess_confounds", "r2_excess_median_confounds", "bh_q_confounds",
         CONF, "confounds", (0, 0.75)),
    ]
    y = np.arange(len(REGION_ORDER))
    for ax, (sess_col, med_col, q_col, color, title, xlim) in zip(axes, panels):
        _style_axes(ax, plt)
        ax.axvline(0, color=BASELINE, lw=1.1, zorder=1)
        for i, region in enumerate(REGION_ORDER):
            vals = sess.loc[sess["region"] == region, sess_col].to_numpy()
            vals = vals[np.isfinite(vals)]
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.22
            ax.scatter(vals, np.full(len(vals), i) + jitter, s=14, color=color,
                       alpha=0.35, zorder=2, linewidths=0, clip_on=False)
            med = float(s.loc[region, med_col])
            ax.plot([med], [i], marker="D", color=color, markersize=7, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
            q = float(s.loc[region, q_col])
            if q < 0.05:
                ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]), i, "*",
                        ha="right", va="center", fontsize=12, color=INK_PRIMARY)
        ax.set_yticks(y)
        ax.set_yticklabels(REGION_ORDER, fontsize=9, color=INK_PRIMARY)
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=11, color=INK_PRIMARY, pad=8)
        ax.grid(axis="x", color=GRIDLINE, lw=0.7, zorder=0)
        ax.set_xlabel("excess $R^2$", fontsize=9, color=INK_SECONDARY)
    axes[0].set_ylabel("region", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, plt, "fig4_decode_excess")


def fig_window() -> None:
    old = pd.read_csv(SURVEY / "stage2" / "prior_decode_summary_by_region_window400ms.csv")
    new = pd.read_csv(SURVEY / "stage2" / "prior_decode_summary_by_region.csv")
    old = old.set_index("region").loc[REGION_ORDER]
    new = new.set_index("region").loc[REGION_ORDER]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(REGION_ORDER))
    w = 0.38
    ax.bar(x - w / 2, old["r2_excess_median"], w, color=OLD, label="$-400$ to $0$ ms", zorder=3)
    ax.bar(x + w / 2, new["r2_excess_median"], w, color=COUNTS, label="$-600$ to $-100$ ms", zorder=3)
    ax.axhline(0, color=BASELINE, lw=1.1, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER, fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel("median excess $R^2$ (counts)", fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRIDLINE, lw=0.8, zorder=0)
    _style_axes(ax, plt)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, plt, "fig5_window_compare")


def fig_reversal() -> None:
    curves = pd.read_csv(SURVEY / "stage3" / "reversal_time_course.csv")
    summary = pd.read_csv(SURVEY / "stage3" / "reversal_summary.csv")
    underpowered = set(summary.loc[summary["underpowered"], "region"])
    for ext in ("pdf", "png"):
        plots.plot_reversal_time_course(
            curves, OUT / f"fig6_reversal_time_course.{ext}", underpowered)


def main() -> int:
    fig_model1()
    fig_glm_hmm()
    fig_belief()
    fig_decode_excess()
    fig_window()
    fig_reversal()
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
