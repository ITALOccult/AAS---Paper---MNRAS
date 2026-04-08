#!/usr/bin/env python3
"""
Legge benchmark_results/uncertainty.csv
Genera figures/fig_uncertainty.pdf
Risponde all'osservazione referee punto 6.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Percorsi ────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA = ROOT.parent.parent.parent / \
       "examples/benchmark_results/uncertainty.csv"
OUT  = ROOT / "figures/fig_uncertainty.pdf"

# ── Stile ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "axes.linewidth":  0.8,
    "lines.linewidth": 1.0,
})

AU_TO_KM    = 1.495978707e8
COMPONENTS  = ["sigma_AT_au", "sigma_CT_au", "sigma_R_au"]
LABELS      = ["Along-track", "Cross-track", "Radial"]
COLORS_COMP = ["#1f77b4", "#d62728", "#2ca02c"]
STYLE       = {"CovProp": "-", "MonteCarlo": "--"}
ALPHA_MC    = 0.7


def plot_method(ax, df_method, method_label, ls):
    for col, label, color in zip(COMPONENTS, LABELS, COLORS_COMP):
        if col not in df_method.columns:
            continue
        sigma_km = df_method[col] * AU_TO_KM
        ax.semilogy(df_method["t_days"], sigma_km,
                    color=color, ls=ls,
                    alpha=ALPHA_MC if "Monte" in method_label else 1.0,
                    label=f"{label} ({method_label})")


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"CSV non trovato: {DATA}")

    df = pd.read_csv(DATA)

    fig, ax = plt.subplots(figsize=(3.5, 3.2),
                           constrained_layout=True)

    for method in ["CovProp", "MonteCarlo"]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        plot_method(ax, sub, method, STYLE[method])

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"$1\sigma$ position uncertainty (km)")
    ax.set_title("Apophis — 30-day uncertainty propagation",
                 fontsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)

    # Legenda compatta
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              fontsize=7, framealpha=0.8,
              loc="upper left", ncol=1)

    # Annotazione accordo CovProp / MC
    ax.text(0.97, 0.05,
            "Solid: STM covariance\nDashed: Monte Carlo $N=500$",
            transform=ax.transAxes,
            fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.8))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Salvato: {OUT}")


if __name__ == "__main__":
    main()
