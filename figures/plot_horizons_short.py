#!/usr/bin/env python3
"""
Legge benchmark_results/horizons_short.csv
Genera figures/fig_horizons_short.pdf
Risponde alle osservazioni referee punti 2 e 5.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Percorsi ────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent
DATA   = ROOT.parent.parent.parent / \
         "examples/benchmark_results/horizons_short.csv"
OUT    = ROOT / "figures/fig_horizons_short.pdf"

# ── Stile AAS journals ───────────────────────────────────
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

COLORS  = {"AAS": "#1f77b4", "SABA4": "#d62728", "RKF7(8)": "#2ca02c"}
STYLES  = {"AAS": "-",       "SABA4": "--",       "RKF7(8)": "-."}
MARKERS = {"AAS": "o",       "SABA4": "s",        "RKF7(8)": "^"}

TARGET_AU   = 1.67e-11   # 2.5 μm in AU
TARGET_LABEL = "2.5 μm (AstDyn target)"

ASTEROIDS = ["Ceres", "Apophis", "Phaethon", "Baruffetti"]
ECC       = {"Ceres": 0.0784, "Apophis": 0.191,
             "Phaethon": 0.890, "Baruffetti": 0.045}
PANEL     = {a: f"({chr(97+i)}) {a}  $e={ECC[a]}$"
             for i, a in enumerate(ASTEROIDS)}


def plot_asteroid(ax, df_ast, asteroid):
    for integ in ["AAS", "SABA4", "RKF7(8)"]:
        d = df_ast[df_ast["integrator"] == integ].sort_values("t_days")
        if d.empty:
            continue
        ax.semilogy(d["t_days"], d["delta_r_au"],
                    color=COLORS[integ],
                    ls=STYLES[integ],
                    marker=MARKERS[integ],
                    markersize=3,
                    markevery=5,
                    label=integ)
    # Linea target
    ax.axhline(TARGET_AU, color="0.5", ls=":", lw=0.8)
    ax.text(0.5, TARGET_AU * 2.0,
            TARGET_LABEL,
            transform=ax.get_yaxis_transform(),
            fontsize=7, color="0.5", va="bottom")
    ax.set_title(PANEL[asteroid], fontsize=8, pad=3)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"$\delta r$ (AU)")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(framealpha=0.8, loc="upper left")


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"CSV non trovato: {DATA}")

    df = pd.read_csv(DATA)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6),
                             constrained_layout=True)
    axes = axes.flatten()

    for ax, asteroid in zip(axes, ASTEROIDS):
        sub = df[df["asteroid"] == asteroid]
        if sub.empty:
            ax.set_visible(False)
            continue
        plot_asteroid(ax, sub, asteroid)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Salvato: {OUT}")


if __name__ == "__main__":
    main()
