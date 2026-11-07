# Figures for "How to run", Step 9 (one per result, as its table suggests) and
# Step 10 (equivalence plots). Reads the CSVs written by turn_dynamics.py,
# turn_dynamics_models.R, mark_not_estimable.py and subsample_checks.py; each
# figure's table view is the CSV it is drawn from. Writes PNG and PDF.
#
# Usage:
#   python make_figures.py --results turn_dynamics_results \
#       --checks subsample_checks_results --out figures

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

# -- palette (validated: 2 categorical slots all-pairs, 4-step ordinal ramp, white surface)
SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # length groups <=5, 6-7, 8-9, 10

NAMES = {
    "gemini3.7_flash": "Gemini 3.7 Flash", "gemma4_31b": "Gemma 4 31B", "gemma4_e2b": "Gemma 4 E2B",
    "gpt_oss_120b": "gpt-oss-120b", "llama3.1_8b": "Llama 3.1 8B", "llama3.3_70b": "Llama 3.3 70B",
    "mistral_nemo": "Mistral NeMo", "mistral_small_4_seed42_only": "Mistral Small 4",
    "phi4_mini_3.8": "Phi-4-mini", "qwen3.5_9b": "Qwen3.5 9B", "qwen3.8_27b": "Qwen3.8 27B",
    "deepseek_v4_Flash": "DeepSeek V4 Flash",
}
LENGTH_GROUPS = ["<=5", "6-7", "8-9", "10"]
LENGTH_LABELS = {"<=5": "5 turns or fewer", "6-7": "6-7 turns", "8-9": "8-9 turns", "10": "10 turns"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlesize": 9.5,
    "axes.titlecolor": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "grid.color": GRID, "grid.linewidth": 0.7,
    "grid.linestyle": "-", "legend.frameon": False, "savefig.dpi": 200, "savefig.facecolor": SURFACE,
})


def name(m):
    return NAMES.get(m, m)


def clean(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, zorder=0)
    ax.set_axisbelow(True)


def save(fig, out, stem):
    fig.savefig(os.path.join(out, f"{stem}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


def dot(ax, x, y, color, hollow=False, size=6.5, z=3):
    ax.plot(x, y, "o", ms=size, mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE,
            mew=1.4 if hollow else 1.2, zorder=z)


# -- odds-ratio dot plot (H1, emergence model) --------------------------------------------------

def or_dotplot(df, out, stem, title, xlabel, flag_col=None, flag_label=None, other_label=None, note=None):
    df = df.sort_values("or_per_turn").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(df) + 1.3))
    ax.axvline(1, color=AXIS, lw=1, zorder=1)
    for i, r in df.iterrows():
        flagged = bool(r[flag_col]) if flag_col else True
        c = BLUE if flagged else MUTED
        ax.plot([r.ci95_low, r.ci95_high], [i, i], color=c, lw=1.5, solid_capstyle="round", zorder=2)
        dot(ax, r.or_per_turn, i, c, hollow=not flagged)
    ax.set_yticks(range(len(df)), [name(m) for m in df.model])
    ax.set_xscale("log")
    ticks = [t for t in (0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3) if
             df.ci95_low.min() * 0.97 <= t <= df.ci95_high.max() * 1.03]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    clean(ax)
    if flag_col:
        ax.legend(handles=[Line2D([], [], marker="o", ls="-", color=BLUE, mfc=BLUE, mec=SURFACE, label=flag_label),
                           Line2D([], [], marker="o", ls="-", color=MUTED, mfc=SURFACE, mec=MUTED, label=other_label)],
                  loc="lower right", fontsize=8)
    if note:
        fig.text(0.01, -0.02, note, color=INK2, fontsize=8, ha="left", va="top", wrap=True)
    save(fig, out, stem)


# -- paired dots (5-vs-10 gap, pronoun use) ------------------------------------------------------

def paired_dots(df, out, stem, title, xlabel, note=None):
    df = df.sort_values("diff_pp").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(df) + 1.4))
    for i, r in df.iterrows():
        ax.plot([r.early_pct, r.late_pct], [i, i], color=AXIS, lw=1.5, zorder=1)
        dot(ax, r.early_pct, i, BLUE)
        dot(ax, r.late_pct, i, ORANGE)
    ax.set_yticks(range(len(df)), [name(m) for m in df.model])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=22)
    clean(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=BLUE, mec=SURFACE, ms=7, label="Turns 1-5"),
                       Line2D([], [], marker="o", ls="", mfc=ORANGE, mec=SURFACE, ms=7, label="Turns 6-10")],
              loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, fontsize=8, borderaxespad=0.1)
    if note:
        fig.text(0.01, -0.02, note, color=INK2, fontsize=8, ha="left", va="top", wrap=True)
    save(fig, out, stem)


# -- small multiples helpers ---------------------------------------------------------------------

def grid_axes(n, ncols=4, w=2.6, h=2.1, sharey=True, sharex=True):
    nrows = int(np.ceil((n + 1) / ncols))  # one slot kept for the legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), sharey=sharey, sharex=sharex)
    return fig, axes.ravel()


def forest_grid(res, est_col, out, stem, title, xlabel, xlim, ticks, note):
    """One panel per model, one row per index cue; filled blue = BH-flagged."""
    res = res[res.family == "index"].copy()
    cues = sorted(res.cue.unique(), key=str.lower)
    models = sorted(res.model.unique(), key=lambda m: name(m).lower())
    fig, axes = grid_axes(len(models), w=2.9, h=3.0, sharex=True, sharey=True)
    lo, hi = xlim
    for ax, m in zip(axes, models):
        d = res[res.model == m].set_index("cue")
        ax.axvline(1, color=AXIS, lw=1, zorder=1)
        for i, cue in enumerate(cues):
            if cue not in d.index:
                continue
            r = d.loc[cue]
            est, l, u = r[est_col], r.ci95_low, r.ci95_high
            usable = (r.get("estimable", True) and pd.notna(est) and 1e-3 < est < 1e3
                      and pd.notna(l) and pd.notna(u) and l > 0 and np.isfinite(u) and u / l < 1e4)
            y = len(cues) - 1 - i
            if not usable:
                ax.text(lo * 1.08, y, "n/a", color=MUTED, fontsize=6.5, va="center", ha="left")
                continue
            c = BLUE if r.flag_bh else MUTED
            ax.plot([max(l, lo), min(u, hi)], [y, y], color=c, lw=1.4, solid_capstyle="round", zorder=2)
            if l < lo:
                ax.plot(lo, y, marker="<", ms=4, color=c, zorder=2)
            if u > hi:
                ax.plot(hi, y, marker=">", ms=4, color=c, zorder=2)
            if lo < est < hi:
                dot(ax, est, y, c, hollow=not r.flag_bh, size=5.5)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_title(name(m), loc="left")
        ax.set_yticks(range(len(cues)), [c for c in reversed(cues)], fontsize=7.5)
        clean(ax)
    for ax in axes[len(models):]:
        ax.axis("off")
    axes[len(models)].legend(
        handles=[Line2D([], [], marker="o", ls="-", color=BLUE, mfc=BLUE, mec=SURFACE, label="Flagged (BH, q = .05)"),
                 Line2D([], [], marker="o", ls="-", color=MUTED, mfc=SURFACE, mec=MUTED, label="Not flagged"),
                 Line2D([], [], ls="", marker="$n/a$", ms=12, color=MUTED, label="No estimate (cue rare or absent)")],
        loc="center", fontsize=8)
    for ax in axes[-4:]:
        if ax.has_data():
            ax.set_xlabel(xlabel)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.0, note, color=INK2, fontsize=8, ha="left", va="top")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    save(fig, out, stem)


# -- figures -------------------------------------------------------------------------------------

def fig_h1(r, out):
    h = r("h1_primary.csv")
    or_dotplot(h, out, "fig01_h1_or_per_turn",
               "H1: odds ratio per turn of the 13-cue index (95% CI)",
               "Odds ratio per turn (log scale; < 1 = fewer cues later in the dialogue)",
               note="Mixed model per LLM, conditional on dialogue length. H1 (one-sided increase, Holm) is supported "
                    "in 0 of 11 models. Source: h1_primary.csv.")


def fig_gap(r, out):
    g = r("five_vs_ten.csv")
    paired_dots(g[g.measure == "index_13"], out, "fig02_five_vs_ten_index",
                "How much a 5-turn benchmark misses: 13-cue index rate, same dialogues",
                "Cues present / cues rated (%)",
                note="Dialogues with all 5 early turns rated and at least one late turn; per-dialogue rates averaged. "
                     "Sorted by the late-minus-early gap. Source: five_vs_ten.csv (measure = index_13), "
                     "with bootstrap CIs of the gap.")


def fig_linearity(r, out):
    t = r("rates_by_turn_length.csv")
    t = t[t.measure == "index_13"]
    models = sorted(t.model.unique(), key=lambda m: name(m).lower())
    fig, axes = grid_axes(len(models))
    for ax, m in zip(axes, models):
        for g, c in zip(LENGTH_GROUPS, RAMP):
            x = t[(t.model == m) & (t.length_group == g) & (t.valid > 0)].sort_values("turn")
            ax.plot(x.turn, x.pct, color=c, lw=1.6, solid_capstyle="round", solid_joinstyle="round")
        ax.set_title(name(m), loc="left")
        ax.set_xticks([1, 5, 10])
        clean(ax, grid="y")
    for ax in axes[len(models):]:
        ax.axis("off")
    axes[len(models)].legend(handles=[Line2D([], [], color=c, lw=2, label=LENGTH_LABELS[g]) for g, c in zip(LENGTH_GROUPS, RAMP)],
                             title="Dialogue length", title_fontsize=8, loc="center", fontsize=8)
    for ax in axes[:: 4]:
        ax.set_ylabel("13-cue index rate (%)")
    for ax in axes[-4:]:
        if ax.has_data():
            ax.set_xlabel("Turn")
    fig.suptitle("Linearity check: 13-cue index rate by turn, one line per dialogue length", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.text(0.01, 0.0, "Lines falling within a length group = within-dialogue decrease; flat lines stacked by length "
             "= selection. Source: rates_by_turn_length.csv (measure = index_13).", color=INK2, fontsize=8, va="top")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    save(fig, out, "fig03_linearity_by_length")


def fig_cues(r, out):
    forest_grid(r("cue_trends_reported.csv"), "or_per_turn", out, "fig04_cue_trends",
                "Q3: which cues change - odds ratio per turn for each index cue (95% CI)",
                "OR per turn (log)", (0.08, 5), [0.1, 0.25, 0.5, 1, 2, 4],
                "One mixed model per LLM and cue. Filled = flagged by BH within the model (q = .05); arrows = interval "
                "beyond the axis; n/a = no variation or not estimable. Source: cue_trends_reported.csv.")


def fig_emergence_model(r, out):
    e = r("emergence_model.csv")
    e["flag"] = e.p_bh < 0.05
    or_dotplot(e, out, "fig05_emergence_model",
               "Q1: odds ratio per turn of a first appearance (95% CI)",
               "Odds ratio per turn (log scale; > 1 = first appearances more likely later)",
               flag_col="flag", flag_label="Flagged (BH across models, q = .05)", other_label="Not flagged",
               note="First-appearance model over the at-risk turns (2-10) of the 13 index cues; cues never appearing "
                    "in a model are left out of its fit. Source: emergence_model.csv.")


def fig_first_appearance(r, out):
    e = r("emergence_by_turn.csv")
    e = e[e.turn >= 2].groupby(["model", "turn"])[["new", "at_risk"]].sum().reset_index()
    e["rate"] = e.new / e.at_risk * 100
    models = sorted(e.model.unique(), key=lambda m: name(m).lower())
    fig, axes = grid_axes(len(models))
    for ax, m in zip(axes, models):
        x = e[e.model == m]
        ax.axvline(5.5, color=AXIS, lw=0.8, zorder=1)
        ax.plot(x.turn, x.rate, color=BLUE, lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.set_title(name(m), loc="left")
        ax.set_xticks([2, 5, 6, 10])
        clean(ax, grid="y")
    for ax in axes[len(models):]:
        ax.axis("off")
    axes[len(models)].text(0.0, 0.5, "First appearances / cue-dialogues at risk,\npooled over the 13 index cues.\n"
                           "Vertical rule: end of the 5-turn window.", color=INK2, fontsize=8, va="center")
    for ax in axes[:: 4]:
        ax.set_ylabel("First-appearance rate (%)")
    for ax in axes[-4:]:
        if ax.has_data():
            ax.set_xlabel("Turn")
    fig.suptitle("Q1: first-appearance rate by turn (turn 1 is starting prevalence, not shown)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.text(0.01, 0.0, "Descriptive. Source: emergence_by_turn.csv (summed over cues).", color=INK2, fontsize=8, va="top")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    save(fig, out, "fig06_first_appearance_by_turn")


def fig_persistence(r, out):
    forest_grid(r("persistence_model_reported.csv"), "or_carry_over", out, "fig07_persistence",
                "Q2: carry-over - odds ratio of a cue at turn t given it at t-1 (95% CI)",
                "Carry-over OR (log)", (0.1, 50), [0.1, 1, 10, 50],
                "Adjusted for the turn-1 value, turn, length and the dialogue's tendency. Filled = flagged by BH within "
                "the model (q = .05); arrows = interval beyond the axis; n/a = no estimate. "
                "Source: persistence_model_reported.csv.")


def fig_pronoun(r, out):
    g = r("five_vs_ten.csv")
    paired_dots(g[g.measure == "personal pronoun use"], out, "fig08_pronoun_use",
                "Personal pronoun use (descriptive): rate in turns 1-5 and 6-10, same dialogues",
                "Messages with personal pronoun use (%)",
                note="Regex-based cue, kept out of the 13-cue index. Sorted by the late-minus-early gap. "
                     "Sources: five_vs_ten.csv (measure = personal pronoun use), pronoun_summary.csv.")


LABEL_SPOTS = [  # offset (points), alignment; the far ones get a leader line
    (5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"), (-5, -3, "right", "top"),
    (0, 7, "center", "bottom"), (0, -7, "center", "top"),
    (14, 12, "left", "bottom"), (14, -12, "left", "top"), (-14, 12, "right", "bottom"), (-14, -12, "right", "top"),
    (0, 20, "center", "bottom"), (0, -20, "center", "top"), (24, 0, "left", "center"), (-24, 0, "right", "center"),
    (26, 22, "left", "bottom"), (-26, 22, "right", "bottom"), (26, -22, "left", "top"), (-26, -22, "right", "top"),
    (0, 32, "center", "bottom"), (0, -32, "center", "top"), (40, 8, "left", "center"), (-40, 8, "right", "center"),
]


def place_labels(ax, xs, ys, labels):
    """Direct labels that avoid each other and the points: the first free spot
    from LABEL_SPOTS, a leader line for the far ones. Call after the layout is final."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    s = fig.dpi / 72
    pts = ax.transData.transform(np.c_[xs, ys])
    # dot plus a clear zone, so no label ends right beside another model's dot
    dots = [Bbox.from_bounds(px - 11 * s, py - 9 * s, 22 * s, 18 * s) for px, py in pts]
    frame = ax.get_window_extent(rend)

    def area(b):
        return 0.0 if b is None else max(b.width, 0) * max(b.height, 0)

    placed = []
    for i in np.argsort(-pts[:, 1]):
        others = [d for j, d in enumerate(dots) if j != i and np.hypot(*(pts[j] - pts[i])) > 3 * s]  # coincident dots read as one
        best = None  # (clash, spot index)
        for k, (dx, dy, ha, va) in enumerate(LABEL_SPOTS):
            t = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=6.5, color=INK2)
            bb = t.get_window_extent(rend)
            t.remove()
            inside = Bbox.intersection(bb, frame)
            clash = (sum(area(Bbox.intersection(bb, b)) for b in placed + others)
                     + area(bb) - area(inside))
            if best is None or clash < best[0]:
                best = (clash, k)
            if clash == 0:
                break
        dx, dy, ha, va = LABEL_SPOTS[best[1]]
        t = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                        fontsize=6.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6, shrinkA=1, shrinkB=4) if best[1] >= 6 else None)
        placed.append(t.get_window_extent(rend))


def fig_between(r, out):
    ml = r("model_level.csv").merge(r("h1_primary.csv")[["model", "or_per_turn"]], on="model")
    outcomes = [("index_rate_all_turns_pct", "13-cue rate, all turns (%)"),
                ("index_rate_early_pct", "13-cue rate, turns 1-5 (%)"),
                ("index_gap_pp", "5-vs-10 gap (points)"),
                ("late_emergence_pct", "New cue after turn 5 (% of dialogues)"),
                ("pronoun_rate_pct", "Pronoun use, all turns (%)"),
                ("or_per_turn", "H1 odds ratio per turn")]
    preds = [("release_date", "Release date", "fig09a_between_models_release_date"),
             ("params_active_b", "Active parameters (B, log scale)", "fig09b_between_models_size")]
    for pcol, plabel, stem in preds:
        x_all = pd.to_datetime(ml.release_date) if pcol == "release_date" else ml[pcol]
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
        to_label = []
        for ax, (ocol, olabel) in zip(axes.ravel(), outcomes):
            ok = x_all.notna() & ml[ocol].notna()
            x, y = x_all[ok], ml.loc[ok, ocol]
            rank_x = (x.astype("int64") if pcol == "release_date" else x).rank()
            rho = rank_x.corr(y.rank())
            ax.plot(x, y, "o", ms=6.5, mfc=BLUE, mec=SURFACE, mew=1.2, zorder=3)
            ax.margins(x=0.18, y=0.14)
            xnum = mdates.date2num(x) if pcol == "release_date" else x.to_numpy(float)
            to_label.append((ax, xnum, y.to_numpy(float), [name(m) for m in ml.loc[ok, "model"]]))
            if pcol != "release_date":
                ax.set_xscale("log")
                ax.set_xticks([2, 5, 10, 30, 100], ["2", "5", "10", "30", "100"])
                ax.minorticks_off()
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.set_title(f"{olabel}\nSpearman ρ = {rho:.2f}, n = {int(ok.sum())}", loc="left", fontsize=8.5)
            ax.set_xlabel(plabel)
            clean(ax, grid="both")
        fig.suptitle(f"Between models: each outcome against {plabel.split(' (')[0].lower()} (descriptive, no p-values)",
                     x=0.01, ha="left", fontsize=11, color=INK)
        fig.text(0.01, 0.0, "About 11 models, several from the same family; any correlation is fragile. "
                 + ("Gemini 3.7 Flash has no public parameter count. " if pcol != "release_date" else "")
                 + "Sources: model_level.csv, model_level_correlations.csv, h1_primary.csv.",
                 color=INK2, fontsize=8, va="top")
        fig.tight_layout(rect=(0, 0.01, 1, 0.96))
        for args in to_label:
            place_labels(*args)
        save(fig, out, stem)


def fig_step10(checks, out):
    c1 = pd.read_csv(os.path.join(checks, "subsample_vs_rest.csv"))
    c2 = pd.read_csv(os.path.join(checks, "test_vs_seed42_paired.csv")).rename(columns={"mean_paired_diff": "diff"})
    stats = [("rate_all_pct", "Rate, all turns"), ("rate_early_pct", "Rate, turns 1-5"), ("gap_pp", "5-vs-10 gap")]
    order = ["index_13"] + sorted([m for m in c1.measure.unique() if m not in ("index_13", "dialogue")], key=str.lower)
    label = {"index_13": "13-cue index"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.2), sharey=True, sharex=True)
    for row, (res, rlabel) in enumerate([(c1, "Check 1: seed42 subsample vs rest of all960 (862)"),
                                         (c2, "Check 2: test run vs seed42 run, paired by scenario")]):
        for col, (st, slabel) in enumerate(stats):
            ax = axes[row, col]
            d = res[res.statistic == st].set_index("measure")
            m = d.margin.iloc[0]
            ax.axvspan(-m, m, color="#f0efec", zorder=0)
            ax.axvline(0, color=AXIS, lw=1, zorder=1)
            for i, meas in enumerate(order):
                if meas not in d.index:
                    continue
                r = d.loc[meas]
                y = len(order) - 1 - i
                c = BLUE if r.equivalent_within_margin else MUTED
                ax.plot([r.ci90_low, r.ci90_high], [y, y], color=c, lw=1.4, solid_capstyle="round", zorder=2)
                dot(ax, r["diff"], y, c, hollow=not r.equivalent_within_margin, size=5.5)
            ax.set_yticks(range(len(order)), [label.get(o, o) for o in reversed(order)], fontsize=7.5)
            ax.set_title(f"{slabel}" if row == 0 else "", loc="left")
            if col == 0:
                ax.annotate(rlabel, (0, 1.06 if row == 0 else 1.02), xycoords="axes fraction", fontsize=9,
                            color=INK, ha="left", va="bottom", fontweight="semibold")
            if row == 1:
                ax.set_xlabel("Difference (percentage points), 90% CI")
            clean(ax)
    fig.legend(handles=[Line2D([], [], marker="o", ls="-", color=BLUE, mfc=BLUE, mec=SURFACE, label="Equivalent within ±5 points"),
                        Line2D([], [], marker="o", ls="-", color=MUTED, mfc=SURFACE, mec=MUTED, label="Not shown equivalent"),
                        Line2D([], [], color="#f0efec", lw=8, label="Equivalence margin")],
               loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8)
    fig.suptitle("Step 10: Mistral Small 4 checks - differences with 90% intervals against the ±5-point margin",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, -0.06, "Equivalent = the 90% interval lies inside the margin (two one-sided tests at .05). Not shown "
             "equivalent does not mean different. Sources: subsample_vs_rest.csv, test_vs_seed42_paired.csv.",
             color=INK2, fontsize=8, va="top")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, out, "fig10_step10_equivalence_cues")

    # dialogue-level measures: each on its own axis and margin
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.2))
    for ax, (st, slabel, unit) in zip(axes, [("length_turns", "Dialogue length", "turns"),
                                             ("reached_full_pct", "Reached 10 turns", "percentage points")]):
        rows = [(c1[(c1.measure == "dialogue") & (c1.statistic == st)].iloc[0], "Check 1"),
                (c2[(c2.measure == "dialogue") & (c2.statistic == st)].iloc[0], "Check 2")]
        m = rows[0][0].margin
        ax.axvspan(-m, m, color="#f0efec", zorder=0)
        ax.axvline(0, color=AXIS, lw=1, zorder=1)
        for y, (r, lab) in zip([1, 0], rows):
            c = BLUE if r.equivalent_within_margin else MUTED
            ax.plot([r.ci90_low, r.ci90_high], [y, y], color=c, lw=1.4, solid_capstyle="round", zorder=2)
            dot(ax, r["diff"], y, c, hollow=not r.equivalent_within_margin, size=5.5)
        ax.set_yticks([1, 0], ["Check 1: subsample vs rest", "Check 2: test vs seed42"])
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(f"{slabel} (margin ±{m:g} {unit.split()[0] if unit == 'turns' else 'points'})", loc="left")
        ax.set_xlabel(f"Difference ({unit}), 90% CI")
        clean(ax)
    fig.tight_layout()
    save(fig, out, "fig11_step10_equivalence_dialogue")


# -- exploratory rerun: with and without each dialogue's final turn ------------------------------

def paired_ci(a, b, est, lo, hi, out, stem, title, xlabel, ref, log, note, ylabels=None):
    """Two estimates per model (a: all rated turns, b: without the final turn), sorted by a."""
    a = a.sort_values(est)
    b = b.reindex(a.index)
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(a) + 1.5))
    ax.axvline(ref, color=AXIS, lw=1, zorder=1)
    for i, m in enumerate(a.index):
        for df, c, dy in ((a, BLUE, 0.17), (b, ORANGE, -0.17)):
            ax.plot([df.at[m, lo], df.at[m, hi]], [i + dy, i + dy], color=c, lw=1.5, solid_capstyle="round", zorder=2)
            dot(ax, df.at[m, est], i + dy, c, size=6)
    ax.set_yticks(range(len(a)), ylabels(a.index) if ylabels else [name(m) for m in a.index])
    if log:
        ax.set_xscale("log")
        ticks = [t for t in (0.85, 0.9, 0.95, 1, 1.05, 1.1) if
                 min(a[lo].min(), b[lo].min()) * 0.98 <= t <= max(a[hi].max(), b[hi].max()) * 1.02]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=22)
    clean(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", color=BLUE, mfc=BLUE, mec=SURFACE, label="All rated turns"),
                       Line2D([], [], marker="o", ls="-", color=ORANGE, mfc=ORANGE, mec=SURFACE,
                              label="Without each dialogue's final turn")],
              loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, fontsize=8, borderaxespad=0.1)
    fig.text(0.01, -0.02, note, color=INK2, fontsize=8, ha="left", va="top", wrap=True)
    save(fig, out, stem)


def fig_compare(read, read_alt, out):
    h0, h1 = read("h1_primary.csv").set_index("model"), read_alt("h1_primary.csv").set_index("model")
    paired_ci(h0, h1, "or_per_turn", "ci95_low", "ci95_high", out, "fig12_h1_without_final_turn",
              "H1 odds ratio per turn, with and without each dialogue's final turn (95% CI)",
              "Odds ratio per turn (log scale; < 1 = fewer cues later in the dialogue)", 1, True,
              "Exploratory rerun added after the results: the final message is usually a closing exchange. "
              "Same mixed model and dialogues, each one turn shorter. Sources: h1_primary.csv of "
              "turn_dynamics_results and turn_dynamics_drop_final_turn.")
    g0 = read("five_vs_ten.csv").query("measure == 'index_13'").set_index("model")
    g1 = read_alt("five_vs_ten.csv").query("measure == 'index_13'").set_index("model")
    paired_ci(g0, g1, "diff_pp", "ci95_low_pp", "ci95_high_pp", out, "fig13_gap_without_final_turn",
              "5-vs-10 gap of the 13-cue index, with and without each dialogue's final turn (95% CI)",
              "Turns 6-10 minus turns 1-5 (percentage points)", 0, False,
              "Labels give the dialogues compared (all turns / without final turn): dialogues that ended at turn 6 "
              "lose their only late turn and drop out. Sources: five_vs_ten.csv (measure = index_13) of both runs.",
              ylabels=lambda idx: [f"{name(m)} ({g0.at[m, 'n_dialogues']} / {g1.at[m, 'n_dialogues']})" for m in idx])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Draw the Step 9 and Step 10 figures.")
    p.add_argument("--results", default="turn_dynamics_results")
    p.add_argument("--checks", default="subsample_checks_results")
    p.add_argument("--compare", default=None, help="Results without each dialogue's final turn (exploratory)")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    read = lambda f: pd.read_csv(os.path.join(a.results, f), float_precision="round_trip")  # noqa: E731
    for f in (fig_h1, fig_gap, fig_linearity, fig_cues, fig_emergence_model, fig_first_appearance,
              fig_persistence, fig_pronoun, fig_between):
        f(read, a.out)
    fig_step10(a.checks, a.out)
    if a.compare:
        fig_compare(read, lambda f: pd.read_csv(os.path.join(a.compare, f), float_precision="round_trip"), a.out)
