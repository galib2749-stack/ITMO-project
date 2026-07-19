"""Regenerate presentation-only chart images with the X5-themed dark palette,
from already-cached artifacts (no retraining/recomputation of models needed):
  - data/processed/holdout_scores_*.npy, holdout_target.npy, holdout_treatment.npy
  - artifacts/qini_curves.json

Outputs go to reports/figures/presentation_*.png (kept separate from the
notebook's own default-matplotlib-style figures, which stay untouched).
"""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from theme import BG, CARD_BG, MODEL_COLORS, MODEL_LABELS, GRAY_TEXT, WHITE, LINE_GRID, FONT  # noqa: E402

PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"
OUT_DIR = "reports/figures"

MODEL_ORDER = ["x_learner_catboost", "t_learner_catboost", "transformer_two_head",
               "random_targeting", "response_catboost"]


def style_axes(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(LINE_GRID)
    ax.tick_params(colors=GRAY_TEXT, labelsize=11)
    ax.xaxis.label.set_color(GRAY_TEXT)
    ax.yaxis.label.set_color(GRAY_TEXT)
    ax.title.set_color(WHITE)
    ax.grid(True, color=LINE_GRID, linewidth=0.6, alpha=0.6)


def plot_qini_curves():
    with open(f"{ARTIFACTS_DIR}/qini_curves.json") as f:
        curves = json.load(f)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for name in MODEL_ORDER:
        c = curves[name]
        ax.plot(c["fractions"], c["qini"], label=MODEL_LABELS[name],
                color=MODEL_COLORS[name], linewidth=2.4)
    ax.plot(curves[MODEL_ORDER[0]]["fractions"], curves[MODEL_ORDER[0]]["random"],
            linestyle="--", color=GRAY_TEXT, linewidth=1.6, label="Случайный порядок")

    ax.set_xlabel("Доля населения (по убыванию uplift-скора)", fontname=FONT)
    ax.set_ylabel("Qini(k)", fontname=FONT)
    ax.set_title("Qini-кривые — отложенная выборка, n = 40 011", fontname=FONT, fontsize=14, fontweight="bold")
    style_axes(ax, fig)
    leg = ax.legend(facecolor=CARD_BG, edgecolor=LINE_GRID, labelcolor=WHITE, fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/presentation_qini_curves.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/presentation_qini_curves.png")


def expected_value_curve(y_true, treatment, score, margin, cost, n_points=100):
    order = np.argsort(-score)
    y_sorted, t_sorted = y_true[order], treatment[order]
    n = len(y_true)
    fractions = np.linspace(1 / n_points, 1.0, n_points)
    values = []
    for frac in fractions:
        k = int(frac * n)
        yk, tk = y_sorted[:k], t_sorted[:k]
        n_t, n_c = (tk == 1).sum(), (tk == 0).sum()
        rate_t = yk[tk == 1].mean() if n_t > 0 else 0.0
        rate_c = yk[tk == 0].mean() if n_c > 0 else 0.0
        incremental_conversions = (rate_t - rate_c) * k
        values.append(incremental_conversions * margin - k * cost)
    return fractions, np.array(values)


def plot_business_curves():
    y = np.load(f"{PROCESSED_DIR}/holdout_target.npy")
    t = np.load(f"{PROCESSED_DIR}/holdout_treatment.npy")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for name in MODEL_ORDER:
        score = np.load(f"{PROCESSED_DIR}/holdout_scores_{name}.npy")
        fracs, evs = expected_value_curve(y, t, score, margin=300.0, cost=5.0)
        ax.plot(fracs, evs, label=MODEL_LABELS[name], color=MODEL_COLORS[name], linewidth=2.4)

    ax.set_xlabel("Доля охваченной аудитории", fontname=FONT)
    ax.set_ylabel("Сценарный ожидаемый эффект, ₽ (иллюстративно)", fontname=FONT)
    ax.set_title("Сценарный ожидаемый эффект (маржа 300₽, SMS 5₽)", fontname=FONT, fontsize=14, fontweight="bold")
    style_axes(ax, fig)
    ax.legend(facecolor=CARD_BG, edgecolor=LINE_GRID, labelcolor=WHITE, fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/presentation_business_ev_curves.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/presentation_business_ev_curves.png")


if __name__ == "__main__":
    plot_qini_curves()
    plot_business_curves()
