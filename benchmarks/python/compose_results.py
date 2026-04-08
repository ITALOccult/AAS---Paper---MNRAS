#!/usr/bin/env python3
"""
compose_results.py
Carica tutti i CSV benchmark e produce figure di confronto
a 4 integratori (AAS, SABA4, RKF7(8), IAS15) per il paper.

Formati CSV gestiti:
  short_term_astdyn.csv   — wide, AAS only
  long_term_astdyn.csv    — wide, AAS only
  long_term_tests.csv     — long, AAS + SABA4 + RKF78
  horizons_short.csv      — long, AAS + SABA4 + RKF78
  efficiency.csv          — long, AAS + SABA4 + RKF78
  short_term_ias15.csv    — long, IAS15
  long_term_ias15_*.csv   — long, IAS15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/Users/michelebigi/Documents/Develop/ASTDYN/paper/AAS/AAS-Integrator-Paper")
DATA = ROOT / "benchmark"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

AU_TO_M = 1.495978707e11

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi":      150,
    "axes.linewidth":  0.8,
    "lines.linewidth": 0.9,
})

COLORS  = {"AAS": "#1f77b4", "SABA4": "#ff7f0e",
           "RKF7(8)": "#d62728", "IAS15": "#2ca02c"}
MARKERS = {"AAS": "o", "SABA4": "s", "RKF7(8)": "^", "IAS15": "D"}
INTEGRATORS = ["AAS", "SABA4", "RKF7(8)", "IAS15"]

FAMILIES = {
    "NEA":      ["Apophis", "Icarus", "Phaethon"],
    "Trojan":   ["Achilles", "Patroclus", "Hektor"],
    "Resonant": ["Hilda", "Thule", "Griqua"],
    "TNO":      ["Pluto", "Eris", "Sedna"],
}
ASTS_ORDERED = [a for fam in FAMILIES.values() for a in fam]

T_LONG = {
    "Apophis": 50.0,   "Icarus": 50.0,     "Phaethon": 50.0,
    "Achilles": 1000.0,"Patroclus": 1000.0,"Hektor": 1000.0,
    "Hilda": 500.0,    "Thule": 500.0,     "Griqua": 500.0,
    "Pluto": 10000.0,  "Eris": 10000.0,    "Sedna": 10000.0,
}

# Mappa nomi test AstDyn wide → nomi standard
TEST_MAP_LONG = {
    "energy":           "energy_final",
    "angular_momentum": "angular_momentum_final",
    "reversibility":    "reversibility_r",
    "lyapunov":         "lyapunov_mLCE",
    "jacobi":           "jacobi_final",
}
# Mappa nomi test AstDyn wide (short_term)
TEST_MAP_SHORT = {
    "dE_E_max":        "energy_final",
    "reversibility_r": "reversibility_r",
    "secular_slope":   "secular_slope",
    "dL_L_max":        "angular_momentum_final",
    "lyapunov_yr":     "lyapunov_mLCE",
    "dCJ_CJ":          "jacobi_final",
}

INTEG_MAP = {"RKF78": "RKF7(8)"}


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Assicura le colonne standard e normalizza i nomi."""
    for c in ["asteroid","integrator","test","T_yr","value","unit","note"]:
        if c not in df.columns:
            df[c] = np.nan
    df["integrator"] = df["integrator"].replace(INTEG_MAP)
    return df[["asteroid","integrator","test","T_yr","value","unit","note"]]


# ── Convertitori formato wide ─────────────────────────────────────────

def astdyn_short_to_long(path: Path) -> pd.DataFrame:
    """short_term_astdyn.csv → long (AAS)."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        ast = r["asteroid"]
        if "dr_full_m" in r and not pd.isna(r["dr_full_m"]):
            rows.append({"asteroid": ast, "integrator": "AAS",
                         "test": "horizons_full",
                         "T_yr": 30.0/365.25,
                         "value": float(r["dr_full_m"]) / AU_TO_M,
                         "unit": "AU", "note": "t=30d"})
        if "dE_E_30d" in r and not pd.isna(r["dE_E_30d"]):
            rows.append({"asteroid": ast, "integrator": "AAS",
                         "test": "energy_30d",
                         "T_yr": 30.0/365.25,
                         "value": float(r["dE_E_30d"]),
                         "unit": "dimensionless", "note": ""})
        if "stm_error" in r and not pd.isna(r["stm_error"]):
            rows.append({"asteroid": ast, "integrator": "AAS",
                         "test": "stm_frobenius",
                         "T_yr": 30.0/365.25,
                         "value": float(r["stm_error"]),
                         "unit": "dimensionless", "note": "t=30d"})
    return pd.DataFrame(rows)


def astdyn_long_to_long(path: Path) -> pd.DataFrame:
    """long_term_astdyn.csv → long (AAS)."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        ast = r["asteroid"]
        T   = T_LONG.get(ast, 0.0)
        for col, test in [("dE_E_max",        "energy_final"),
                          ("reversibility_r", "reversibility_r"),
                          ("secular_slope",   "secular_slope"),
                          ("dL_L_max",        "angular_momentum_final"),
                          ("lyapunov_yr",     "lyapunov_mLCE"),
                          ("dCJ_CJ",          "jacobi_final")]:
            val = r.get(col, np.nan)
            if not pd.isna(val):
                rows.append({"asteroid": ast, "integrator": "AAS",
                             "test": test, "T_yr": T,
                             "value": float(val),
                             "unit": "dimensionless"
                                     if test != "secular_slope" else "1/yr",
                             "note": ""})
    return pd.DataFrame(rows)


# ── Caricamento unificato ─────────────────────────────────────────────

def load_all() -> pd.DataFrame:
    frames = []

    # ── AstDyn wide ──────────────────────────────────────────────────
    p = DATA / "short_term_astdyn.csv"
    if p.exists():
        df = astdyn_short_to_long(p)
        frames.append(df)
        print(f"  Caricato: short_term_astdyn.csv  → {len(df)} righe")
    else:
        print("  [skip] short_term_astdyn.csv")

    p = DATA / "long_term_astdyn.csv"
    if p.exists():
        df = astdyn_long_to_long(p)
        frames.append(df)
        print(f"  Caricato: long_term_astdyn.csv   → {len(df)} righe")
    else:
        print("  [skip] long_term_astdyn.csv")

    # ── long_term_tests.csv — AAS + SABA4 + RKF78 ───────────────────
    p = DATA / "long_term_tests.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["test"] = df["test"].replace(TEST_MAP_LONG)
        df = normalise(df)
        frames.append(df)
        print(f"  Caricato: long_term_tests.csv    ({len(df)} righe)")
    else:
        print("  [skip] long_term_tests.csv")

    # ── horizons_short.csv — AAS + SABA4 + RKF78 ────────────────────
    p = DATA / "horizons_short.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df.rename(columns={"delta_r_au": "value"})
        if "t_days" in df.columns:
            df["T_yr"] = df["t_days"] / 365.25
            df["note"] = df["t_days"].apply(lambda t: f"t={t:.0f}d")
        df["test"] = "horizons_full"
        df["unit"] = "AU"
        df = normalise(df)
        frames.append(df)
        print(f"  Caricato: horizons_short.csv     ({len(df)} righe)")
    else:
        print("  [skip] horizons_short.csv")

    # ── IAS15 CSV ────────────────────────────────────────────────────
    for fname in ["short_term_ias15.csv",
                  "long_term_ias15.csv",
                  "long_term_ias15_NEA.csv",
                  "long_term_ias15_Trojan.csv",
                  "long_term_ias15_Resonant.csv"]:
        p = DATA / fname
        if p.exists():
            df = normalise(pd.read_csv(p))
            frames.append(df)
            print(f"  Caricato: {fname}  ({len(df)} righe)")
        else:
            print(f"  [skip] {fname}")

    if not frames:
        raise FileNotFoundError("Nessun CSV trovato in " + str(DATA))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["integrator","asteroid","test"])
    combined = combined.drop_duplicates(
        subset=["asteroid","integrator","test","T_yr"], keep="last")

    print(f"\nTotale righe: {len(combined)}")
    print(f"Integratori: {sorted(combined['integrator'].unique())}")
    print(f"Test: {sorted(combined['test'].unique())}\n")
    return combined


# ── Utility plot ─────────────────────────────────────────────────────

def bar_plot(ax, df, test, asts, ylabel, title, log=True):
    sub = df[df["test"] == test]
    x   = np.arange(len(asts))
    w   = 0.2
    plotted = False
    for i, integ in enumerate(INTEGRATORS):
        vals = []
        for ast in asts:
            row = sub[(sub["asteroid"] == ast) &
                      (sub["integrator"] == integ)]
            vals.append(float(row["value"].values[0])
                        if not row.empty else np.nan)
        if all(np.isnan(v) for v in vals):
            continue
        ax.bar(x + i*w, vals, w,
               label=integ, color=COLORS[integ], alpha=0.85)
        plotted = True
    if log and plotted:
        ax.set_yscale("log")
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(asts, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)


# ── Figure ───────────────────────────────────────────────────────────

def fig_horizons(df):
    sub = df[df["test"] == "horizons_full"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6),
                             constrained_layout=True)
    axes = axes.flatten()
    for ax, (family, asts) in zip(axes, FAMILIES.items()):
        for integ in INTEGRATORS:
            s = sub[(sub["integrator"] == integ) &
                    (sub["asteroid"].isin(asts))]
            if s.empty:
                continue
            t_list, vals = [], []
            for t_yr, grp in s.groupby("T_yr"):
                t_list.append(t_yr * 365.25)
                vals.append(grp["value"].mean())
            ax.semilogy(t_list, vals,
                        color=COLORS[integ], marker=MARKERS[integ],
                        markersize=4, label=integ)
        ax.axhline(1.67e-11, color="gray", ls=":", lw=0.7,
                   label="2.5 μm")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(r"$\delta r$ (AU)")
        ax.set_title(family, fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(fontsize=6)
    fig.suptitle("Horizons residuals — full force model", fontsize=9)
    out = FIGS / "fig_horizons_4integrators.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Salvato: {out.name}")
    plt.close(fig)


def fig_energy_final(df):
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    bar_plot(ax, df, "energy_final", ASTS_ORDERED,
             r"$|\Delta H/H_0|$ (final)",
             "Final energy error — long-term benchmarks")
    ax.axhline(1e-11, color="gray", ls="--", lw=0.7,
               label=r"$10^{-11}$")
    ax.legend(fontsize=7)
    out = FIGS / "fig_energy_4integrators.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Salvato: {out.name}")
    plt.close(fig)


def fig_reversibility(df):
    rev_asts = ["Apophis", "Achilles", "Pluto"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             constrained_layout=True)
    for ax, test, ylabel in zip(
        axes,
        ["reversibility_r", "reversibility_v"],
        [r"$\epsilon_r$", r"$\epsilon_v$"]
    ):
        bar_plot(ax, df, test, rev_asts, ylabel, "Time-reversal error")
    out = FIGS / "fig_reversibility_4integrators.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Salvato: {out.name}")
    plt.close(fig)


def fig_lyapunov(df):
    sub = df[df["test"] == "lyapunov_mLCE"].copy()
    sub = sub.copy()
    sub["value"] = sub["value"].abs()
    fig, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)
    x = np.arange(len(ASTS_ORDERED))
    w = 0.2
    for i, integ in enumerate(INTEGRATORS):
        vals = []
        for ast in ASTS_ORDERED:
            row = sub[(sub["asteroid"] == ast) &
                      (sub["integrator"] == integ)]
            vals.append(float(row["value"].values[0])
                        if not row.empty else np.nan)
        if all(np.isnan(v) for v in vals):
            continue
        ax.bar(x + i*w, vals, w,
               label=integ, color=COLORS[integ], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(ASTS_ORDERED, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"mLCE (yr$^{-1}$)")
    ax.set_title("Maximum Lyapunov exponent", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    out = FIGS / "fig_lyapunov_4integrators.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Salvato: {out.name}")
    plt.close(fig)


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
        bar_plot(ax, sub, test,
                 [a[:6] for a in ASTS_ORDERED],
                 ylabel, "Secular drift", log=log)
        if not log:
            ax.axhline(0.95, color="gray", ls="--", lw=0.8,
                       label="$R^2=0.95$")
            ax.legend(fontsize=7)
    out = FIGS / "fig_secular_4integrators.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Salvato: {out.name}")
    plt.close(fig)


def fig_jacobi(df):
    trojans = ["Achilles", "Patroclus", "Hektor"]
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    bar_plot(ax, df, "jacobi_final", trojans,
             r"$\Delta C_J/C_{J,0}$",
             "Jacobi constant — Trojans (1000 yr)")
    out = FIGS / "fig_jacobi_4integrators.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Salvato: {out.name}")
    plt.close(fig)


def fig_angular_momentum(df):
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    bar_plot(ax, df, "angular_momentum_final", ASTS_ORDERED,
             r"$|\Delta L/L_0|$ (final)",
             "Angular momentum conservation — long-term")
    out = FIGS / "fig_angular_momentum_4integrators.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Salvato: {out.name}")
    plt.close(fig)


# ── Tabella riepilogo ─────────────────────────────────────────────────

def print_summary(df):
    tests = [
        ("energy_final",           "Final |ΔH/H₀|"),
        ("reversibility_r",        "ε_r"),
        ("lyapunov_mLCE",          "mLCE [1/yr]"),
        ("secular_R2",             "Secular R²"),
        ("angular_momentum_final", "|ΔL/L₀|"),
        ("jacobi_final",           "ΔCJ/CJ0"),
    ]
    asts_show = ["Apophis", "Phaethon", "Achilles", "Griqua", "Sedna"]
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Asteroid':<12} {'Test':<25}", end="")
    for integ in INTEGRATORS:
        print(f" {integ:>12}", end="")
    print()
    print("-"*80)
    for ast in asts_show:
        for test, label in tests:
            sub = df[(df["asteroid"] == ast) & (df["test"] == test)]
            if sub.empty:
                continue
            print(f"{ast:<12} {label:<25}", end="")
            for integ in INTEGRATORS:
                row = sub[sub["integrator"] == integ]
                if not row.empty:
                    print(f" {float(row['value'].values[0]):>12.2e}", end="")
                else:
                    print(f" {'N/A':>12}", end="")
            print()


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Caricamento CSV...")
    df = load_all()

    print("Generazione figure...")
    fig_horizons(df)
    fig_energy_final(df)
    fig_reversibility(df)
    fig_lyapunov(df)
    fig_secular(df)
    fig_jacobi(df)
    fig_angular_momentum(df)
    print_summary(df)

    print(f"\nFigure in: {FIGS}")

