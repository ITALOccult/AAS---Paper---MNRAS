#!/usr/bin/env python3
"""
Legge benchmark_results/long_term_tests.csv
Genera 6 figure per i test a lungo termine.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT   = Path(__file__).parent.parent
DATA   = ROOT.parent.parent.parent / "IOccultLibrary/astdyn/examples/benchmark_results/long_term_tests.csv"
LYAP   = ROOT.parent.parent.parent / "IOccultLibrary/astdyn/examples/benchmark_results/lyapunov_series.csv"
OUTDIR = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7,
    "figure.dpi": 150, "axes.linewidth": 0.8,
    "lines.linewidth": 0.9,
})

COLORS  = {"AAS": "#1f77b4", "SABA4": "#ff7f0e", "RKF78": "#d62728"}
MARKERS = {"AAS": "o",       "SABA4": "s",        "RKF78": "^"}
CATS    = {
    "NEA":      ["Apophis", "Icarus", "Phaethon"],
    "Trojan":   ["Achilles", "Patroclus", "Hektor"],
    "Resonant": ["Hilda", "Thule", "Griqua"],
    "TNO":      ["Pluto", "Eris", "Sedna"],
}
CAT_T   = {"NEA": 50, "Trojan": 1000, "Resonant": 500, "TNO": 10000}


def bar_panel(ax, df, test_name, ylabel, title, log=True):
    sub = df[df["test"] == test_name]
    if sub.empty:
        ax.set_visible(False)
        return
    asteroids = sub["asteroid"].unique()
    x = np.arange(len(asteroids))
    w = 0.25
    for i, integ in enumerate(["AAS", "SABA4", "RKF78"]):
        vals = [sub[(sub["asteroid"] == a) &
                    (sub["integrator"] == integ)]["value"]
                .values[0] if not sub[(sub["asteroid"] == a) &
                (sub["integrator"] == integ)].empty else np.nan
                for a in asteroids]
        ax.bar(x + i*w, vals, w, label=integ,
               color=COLORS[integ], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([a[:6] for a in asteroids],
                       rotation=30, ha="right", fontsize=7)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)


def fig_energy(df):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6),
                             constrained_layout=True)
    axes = axes.flatten()
    for ax, (cat, asts) in zip(axes, CATS.items()):
        sub = df[(df["test"] == "energy") &
                 (df["asteroid"].isin(asts))]
        x = np.arange(len(asts))
        w = 0.25
        for i, integ in enumerate(["AAS", "SABA4", "RKF78"]):
            vals = [sub[(sub["asteroid"] == a) &
                        (sub["integrator"] == integ)]["value"]
                    .values[0] if not sub[
                    (sub["asteroid"] == a) &
                    (sub["integrator"] == integ)].empty
                    else np.nan for a in asts]
            ax.bar(x + i*w, vals, w, label=integ,
                   color=COLORS[integ], alpha=0.85)
        ax.set_yscale("log")
        ax.set_xticks(x + w)
        ax.set_xticklabels(asts, rotation=20,
                           ha="right", fontsize=7)
        ax.set_ylabel(r"$|\Delta H/H_0|$")
        ax.set_title(f"{cat} — T = {CAT_T[cat]} yr",
                     fontsize=8)
        ax.axhline(1e-11, color="black", ls=":", lw=0.7, label="target")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    p = OUTDIR / "fig_lt_energy.pdf"
    fig.savefig(p, bbox_inches="tight")
    print(f"Salvato: {p}")


def fig_reversibility(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             constrained_layout=True)
    for ax, test, ylabel in zip(
        axes,
        ["reversibility_r", "reversibility_v"],
        [r"$\epsilon_r = |\Delta r|/|r_0|$",
         r"$\epsilon_v = |\Delta v|/|v_0|$"]
    ):
        sub = df[df["test"] == test]
        asts = sub["asteroid"].unique()
        x = np.arange(len(asts))
        w = 0.25
        for i, integ in enumerate(["AAS", "SABA4", "RKF78"]):
            vals = [sub[(sub["asteroid"] == a) &
                        (sub["integrator"] == integ)]["value"]
                    .values[0] if not sub[
                    (sub["asteroid"] == a) &
                    (sub["integrator"] == integ)].empty
                    else np.nan for a in asts]
            ax.bar(x + i*w, vals, w, label=integ,
                   color=COLORS[integ], alpha=0.85)
        ax.set_yscale("log")
        ax.set_xticks(x + w)
        ax.set_xticklabels(asts, rotation=20,
                           ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title("Time-reversal error", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    p = OUTDIR / "fig_lt_reversibility.pdf"
    fig.savefig(p, bbox_inches="tight")
    print(f"Salvato: {p}")


def fig_secular(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             constrained_layout=True)
    for ax, test, ylabel, log in zip(
        axes,
        ["secular_slope", "secular_R2"],
        [r"$|a|$ (yr$^{-1}$)", r"$R^2$"],
        [True, False]
    ):
        sub = df[df["test"] == test].copy()
        if test == "secular_slope":
            sub["value"] = sub["value"].abs()
        asts = sub["asteroid"].unique()
        x = np.arange(len(asts))
        w = 0.25
        for i, integ in enumerate(["AAS", "SABA4", "RKF78"]):
            vals = [sub[(sub["asteroid"] == a) &
                        (sub["integrator"] == integ)]["value"]
                    .values[0] if not sub[
                    (sub["asteroid"] == a) &
                    (sub["integrator"] == integ)].empty
                    else np.nan for a in asts]
            ax.bar(x + i*w, vals, w, label=integ,
                   color=COLORS[integ], alpha=0.85)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(x + w)
        ax.set_xticklabels([a[:6] for a in asts],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title("Secular drift", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    if test == "secular_R2":
        axes[1].axhline(0.95, color="gray", ls="--",
                        lw=0.8, label="$R^2=0.95$")
    p = OUTDIR / "fig_lt_secular.pdf"
    fig.savefig(p, bbox_inches="tight")
    print(f"Salvato: {p}")


def fig_lyapunov(df):
    sub = df[df["test"] == "lyapunov"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0),
                           constrained_layout=True)
    asts = sub["asteroid"].unique()
    x = np.arange(len(asts))
    w = 0.25
    for i, integ in enumerate(["AAS", "SABA4", "RKF78"]):
        vals = [sub[(sub["asteroid"] == a) &
                    (sub["integrator"] == integ)]["value"]
                .values[0] if not sub[
                (sub["asteroid"] == a) &
                (sub["integrator"] == integ)].empty
                else np.nan for a in asts]
        ax.bar(x + i*w, vals, w, label=integ,
               color=COLORS[integ], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x + w)
    ax.set_xticklabels(asts, rotation=30,
                       ha="right", fontsize=7)
    ax.set_ylabel(r"mLCE (yr$^{-1}$)")
    ax.set_title("Maximum Lyapunov Characteristic Exponent",
                 fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    p = OUTDIR / "fig_lt_lyapunov.pdf"
    fig.savefig(p, bbox_inches="tight")
    print(f"Salvato: {p}")


def fig_jacobi(df):
    sub = df[df["test"] == "jacobi"]
    if sub.empty:
        print("Nessun dato Jacobi")
        return
    fig, ax = plt.subplots(figsize=(3.5, 3.0),
                           constrained_layout=True)
    asts = ["Achilles", "Patroclus", "Hektor"]
    x = np.arange(len(asts))
    w = 0.25
    for i, integ in enumerate(["AAS", "SABA4", "RKF78"]):
        vals = [sub[(sub["asteroid"] == a) &
                    (sub["integrator"] == integ)]["value"]
                .values[0] if not sub[
                (sub["asteroid"] == a) &
                (sub["integrator"] == integ)].empty
                else np.nan for a in asts]
        ax.bar(x + i*w, vals, w, label=integ,
               color=COLORS[integ], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x + w)
    ax.set_xticklabels(asts, fontsize=8)
    ax.set_ylabel(r"$\Delta C_J / C_{J,0}$")
    ax.set_title("Jacobi constant — Trojans (1000 yr)",
                 fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    p = OUTDIR / "fig_lt_jacobi.pdf"
    fig.savefig(p, bbox_inches="tight")
    print(f"Salvato: {p}")


def fig_lyapunov_convergence():
    if not LYAP.exists():
        print("lyapunov_series.csv non trovato")
        return
    df = pd.read_csv(LYAP)
    asts_show = ["Apophis", "Griqua", "Achilles"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8),
                             constrained_layout=True)
    for ax, ast in zip(axes, asts_show):
        sub = df[df["asteroid"] == ast]
        for integ in ["AAS", "SABA4", "RKF78"]:
            d = sub[sub["integrator"] == integ]
            if d.empty:
                continue
            cumulative = d["lambda_i"].expanding().mean()
            ax.semilogy(d["interval_index"],
                        cumulative.abs(),
                        color=COLORS[integ],
                        label=integ, lw=0.9)
        ax.set_xlabel("Interval index")
        ax.set_ylabel(r"Running mean mLCE (yr$^{-1}$)")
        ax.set_title(ast, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
    p = OUTDIR / "fig_lt_lyapunov_convergence.pdf"
    fig.savefig(p, bbox_inches="tight")
    print(f"Salvato: {p}")


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"CSV non trovato: {DATA}")
    df = pd.read_csv(DATA)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_energy(df)
    fig_reversibility(df)
    fig_secular(df)
    fig_lyapunov(df)
    fig_jacobi(df)
    fig_lyapunov_convergence()
    print("Tutte le figure generate.")


if __name__ == "__main__":
    main()
