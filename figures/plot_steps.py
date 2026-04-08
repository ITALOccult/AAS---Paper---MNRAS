#!/usr/bin/env python3
"""
scripts/plot_steps.py
Figura 3: Distribuzione dei passi Δt vs r per AAS su Apophis.
Mostra anche Δt vs t per evidenziare il clustering al perielio.

CSV atteso:
    t_days, dt_days, r_au, integrator
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 150,
    "axes.linewidth": 0.8, "lines.linewidth": 0.8,
})


def plot_step_distribution(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    d  = df[df["integrator"] == "AAS"]

    fig = plt.figure(figsize=(7.0, 2.8), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Pannello sinistro: Δt vs t
    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(d["t_days"], d["dt_days"], ".", ms=1.5,
                 color="#1f77b4", alpha=0.4, rasterized=True)
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel(r"$\Delta t$ (days)")
    ax1.set_title("(a) Step size vs time", fontsize=8)
    ax1.grid(True, which="both", lw=0.3, alpha=0.4)

    # Pannello destro: Δt vs r con guida r^{3/2}
    ax2 = fig.add_subplot(gs[1])
    ax2.loglog(d["r_au"], d["dt_days"], ".", ms=1.5,
               color="#1f77b4", alpha=0.4, rasterized=True)

    # Guida r^{3/2}
    r_ref = np.logspace(np.log10(d["r_au"].min()),
                        np.log10(d["r_au"].max()), 50)
    # Calibra sull'intervallo mediano
    idx   = len(d) // 2
    scale = d["dt_days"].median() / d["r_au"].median()**1.5
    ax2.loglog(r_ref, scale * r_ref**1.5, "k--", lw=0.9, alpha=0.7)
    ax2.text(0.6, 0.15, r"$\propto r^{3/2}$",
             transform=ax2.transAxes, fontsize=8, color="0.4")

    ax2.set_xlabel(r"Heliocentric distance $r$ (AU)")
    ax2.set_ylabel(r"$\Delta t$ (days)")
    ax2.set_title("(b) Step size vs distance", fontsize=8)
    ax2.grid(True, which="both", lw=0.3, alpha=0.4)

    fig.suptitle("AAS integrator — Apophis (1 yr)", fontsize=8, y=1.01)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
