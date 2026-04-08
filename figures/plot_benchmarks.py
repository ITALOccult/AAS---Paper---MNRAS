"""
plot_benchmarks.py — Generate paper figures from AAS benchmark CSV files.

Input:  .../IOccultLibrary/astdyn/examples/benchmark_results/*.csv
Output: .../paper/AAS/AAS-Integrator-Paper/figures/fig_*.pdf
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = pathlib.Path(
    "/Users/michelebigi/Documents/Develop/ASTDYN"
    "/IOccultLibrary/astdyn/examples/benchmark_results"
)
FIGURES    = pathlib.Path(
    "/Users/michelebigi/Documents/Develop/ASTDYN"
    "/paper/AAS/AAS-Integrator-Paper/figures"
)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.linewidth":   0.8,
    "lines.linewidth":  1.0,
})

COLORS  = {"AAS": "#1f77b4", "SABA4": "#d62728", "RKF78": "#2ca02c"}
MARKERS = {"AAS": "o",       "SABA4": "s",        "RKF78": "^"}

ECCENTRICITIES = {
    "Ceres":      0.0784,
    "Apophis":    0.1914,
    "Phaethon":   0.8902,
    "Baruffetti": 0.0503,
}

DAYS_PER_YEAR = 365.25


# ── Figure 1 — Energy conservation ───────────────────────────────────────────

def plot_energy_panel(ax, df_ast, label, ecc):
    """Log-log precision vs |ΔE/E| for one asteroid."""
    for integ in ("AAS", "SABA4", "RKF78"):
        sub = df_ast[df_ast.integrator == integ]
        if sub.empty:
            continue
        ax.loglog(sub.precision, sub.delta_E_over_E,
                  color=COLORS[integ], marker=MARKERS[integ],
                  ms=4, label=integ, linestyle="-")

    # Slope-4 guide line (α = 0.5 → slope 4 on log-log)
    x_guide = np.logspace(np.log10(df_ast.precision.min()),
                          np.log10(df_ast.precision.max()), 30)
    y_ref   = df_ast[df_ast.integrator == "AAS"].delta_E_over_E.median()
    x_ref   = df_ast[df_ast.integrator == "AAS"].precision.median()
    ax.loglog(x_guide, y_ref * (x_guide / x_ref)**4,
              "k--", lw=0.8, label=r"slope 4")

    ax.set_xlabel("precision / step [AU/day or –]")
    ax.set_ylabel(r"$|\Delta E/E|$")
    ax.set_title(f"({label}) {list(ECCENTRICITIES.keys())[ord(label)-ord('a')]}"
                 f"  $e={ecc:.3f}$")
    ax.legend(fontsize=7, framealpha=0.6)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)


def fig_energy(df):
    asteroids = df.asteroid.unique()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    panel_labels = ["a", "b", "c", "d"]
    for i, (ax, name) in enumerate(zip(axes.flat, asteroids)):
        ecc = ECCENTRICITIES.get(name, 0.0)
        plot_energy_panel(ax, df[df.asteroid == name], panel_labels[i], ecc)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_energy_vs_precision.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_energy_vs_precision.pdf")


# ── Figure 2 — Divergence time ────────────────────────────────────────────────

def fig_divergence(df):
    asteroids = df.asteroid.unique()[:2]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    rng = np.random.default_rng(42)

    T_MAX_YR = 2.0  # integration horizon

    for ax, name in zip(axes, asteroids):
        sub = df[df.asteroid == name]
        handles = []
        for integ in ("AAS", "SABA4"):
            rows = sub[sub.integrator == integ]
            if rows.empty:
                continue
            diverged  = rows[rows.t_divergence_days > 0]
            censored  = rows[rows.t_divergence_days < 0]

            jitter_d = rng.uniform(-0.15, 0.15, len(diverged))
            jitter_c = rng.uniform(-0.15, 0.15, len(censored))

            # Diverged: scatter dots
            if not diverged.empty:
                ax.scatter(diverged.perturb_index + jitter_d,
                           diverged.t_divergence_days / DAYS_PER_YEAR,
                           color=COLORS[integ], marker=MARKERS[integ],
                           s=18, zorder=3, label=integ)

            # Right-censored: downward-pointing triangles at T_MAX
            if not censored.empty:
                sc = ax.scatter(censored.perturb_index + jitter_c,
                                [T_MAX_YR] * len(censored),
                                color=COLORS[integ], marker="v",
                                s=28, zorder=3,
                                label=f"{integ} (censored)")
                handles.append(sc)

        # Dashed line at T_MAX
        ax.axhline(T_MAX_YR, color="0.5", ls="--", lw=0.8)
        ax.text(8.4, T_MAX_YR + 0.04, "2 yr", fontsize=7, color="0.5", va="bottom")

        ax.set_xlim(0.5, 8.5)
        ax.set_ylim(0, T_MAX_YR * 1.25)
        ax.set_xlabel("perturbation index $k$")
        ax.set_ylabel(r"$\hat{\tau}_D$ [yr]")
        ax.set_title(name)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.grid(True, lw=0.3, alpha=0.5)

        # Annotation if all censored
        all_censored = (df[df.asteroid == name].t_divergence_days < 0).all()
        if all_censored:
            ax.text(0.5, 0.45, "No divergence\nwithin 2 yr",
                    transform=ax.transAxes, fontsize=8, color="0.35",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_divergence_time.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_divergence_time.pdf")


# ── Figure 3 — Step distribution ─────────────────────────────────────────────

def fig_step_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Panel (a): Δt vs t
    ax = axes[0]
    ax.semilogy(df.t_days, df.dt_days,
                color=COLORS["AAS"], ms=1.5, alpha=0.4,
                marker=".", linestyle="none", rasterized=True)
    ax.set_xlabel("$t$ [days]")
    ax.set_ylabel(r"$\Delta t$ [days]")
    ax.set_title("(a) Step size vs time — Apophis")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)

    # Panel (b): Δt vs r (log-log) + r^{3/2} guide
    ax = axes[1]
    r_vals  = df.r_au.values
    dt_vals = df.dt_days.values
    ax.loglog(r_vals, dt_vals,
              color=COLORS["AAS"], ms=1.5, alpha=0.4,
              marker=".", linestyle="none", rasterized=True)

    # r^{3/2} guide calibrated on median
    r_med  = np.median(r_vals)
    dt_med = np.median(dt_vals)
    r_line = np.logspace(np.log10(r_vals.min()), np.log10(r_vals.max()), 50)
    ax.loglog(r_line, dt_med * (r_line / r_med)**1.5,
              "k--", lw=0.8, label=r"$r^{3/2}$")
    ax.set_xlabel("$r$ [AU]")
    ax.set_ylabel(r"$\Delta t$ [days]")
    ax.set_title(r"(b) Step size vs $r$")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_step_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_step_distribution.pdf")


# ── Figure 4 — Shadow Hamiltonian ────────────────────────────────────────────

def fig_shadow(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for integ, ls in [("AAS", "-"), ("SABA4", "--")]:
        sub = df[df.integrator == integ]
        t_yr = sub.t_days / DAYS_PER_YEAR
        axes[0].semilogy(t_yr, sub.delta_H_physical.clip(1e-20),
                         color=COLORS[integ], ls=ls, label=integ)
        axes[1].semilogy(t_yr, sub.delta_H_shadow.clip(1e-20),
                         color=COLORS[integ], ls=ls, label=integ)

    for ax, title in zip(axes, [r"$|\Delta H/H_0|$", r"$|\Delta \tilde{H}/\tilde{H}_0|$"]):
        ax.set_xlabel("$t$ [yr]")
        ax.set_ylabel(title)
        ax.set_title(title + "  — Ceres")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_shadow_hamiltonian.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_shadow_hamiltonian.pdf")


# ── Figure 5 — STM accuracy ───────────────────────────────────────────────────

def fig_stm(df):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    sub = df[df.method == "AAS_analytic"]
    ax.semilogy(sub.t_days, sub.stm_error_frobenius.clip(1e-20),
                color=COLORS["AAS"], marker="o", ms=3, label="AAS analytic")
    ax.set_xlabel("$t$ [days]")
    ax.set_ylabel(r"$\|\Phi_\mathrm{AAS} - \Phi_\mathrm{num}\|_F$")
    ax.set_title("STM accuracy — Apophis (30 days)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_stm_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_stm_accuracy.pdf")


# ── Figure 6 — Horizons validation ───────────────────────────────────────────

def fig_horizons(df):
    asteroids = df.asteroid.unique()
    n = len(asteroids)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.0), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, asteroids):
        sub = df[df.asteroid == name]
        for integ in ("AAS", "SABA4", "RKF78"):
            rows = sub[sub.integrator == integ]
            if rows.empty:
                continue
            ax.semilogy(rows.t_days / DAYS_PER_YEAR, rows.dr_km,
                        color=COLORS[integ], marker=MARKERS[integ],
                        ms=4, label=integ, linestyle="-")
        ax.set_xlabel("$t$ [yr]")
        ax.set_ylabel("$|\\Delta r|$ [km]")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_horizons_validation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_horizons_validation.pdf")


# ── Figure 7 — Efficiency ─────────────────────────────────────────────────────

def fig_efficiency(df):
    asteroids = df.asteroid.unique()
    n = len(asteroids)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, asteroids):
        sub = df[df.asteroid == name]
        for integ in ("AAS", "SABA4", "RKF78"):
            rows = sub[sub.integrator == integ].sort_values("n_func_evals")
            if rows.empty:
                continue
            ax.loglog(rows.n_func_evals, rows.delta_E_over_E,
                      color=COLORS[integ], marker=MARKERS[integ],
                      ms=4, label=integ, linestyle="-")
        ax.set_xlabel("function evaluations")
        ax.set_ylabel(r"$|\Delta E/E|$")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_efficiency.pdf")


# ── Figure 8 — Short-term Horizons residuals (A6) ─────────────────────────────

def fig_horizons_short(df):
    asteroids = df.asteroid.unique()
    n = len(asteroids)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.0), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, asteroids):
        sub = df[df.asteroid == name]
        for integ in ("AAS", "SABA4", "RKF78"):
            rows = sub[sub.integrator == integ].sort_values("t_days")
            if rows.empty:
                continue
            ax.semilogy(rows.t_days, rows.delta_r_au,
                        color=COLORS[integ], marker=MARKERS[integ],
                        ms=3, label=integ, linestyle="-")
        ax.set_xlabel("$t$ [days]")
        ax.set_ylabel(r"$|\Delta r|$ [AU]")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_horizons_short.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_horizons_short.pdf")


# ── Figure 9 — Uncertainty propagation (A7) ──────────────────────────────────

def fig_uncertainty(df):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=False)
    components = [("sigma_AT_au", "Along-track $\\sigma_{AT}$"),
                  ("sigma_CT_au", "Cross-track $\\sigma_{CT}$"),
                  ("sigma_R_au",  "Radial $\\sigma_R$")]
    for ax, (col, title) in zip(axes, components):
        for method in df.method.unique():
            sub = df[df.method == method].sort_values("t_days")
            ax.semilogy(sub.t_days, sub[col],
                        label=method, linestyle="-", marker="o", ms=3)
        ax.set_xlabel("$t$ [days]")
        ax.set_ylabel("[AU]")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_uncertainty.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_uncertainty.pdf")


# ── Long-term helpers ────────────────────────────────────────────────────────

CATEGORIES = {
    "NEA/PHA":  ["Apophis", "Icarus", "Phaethon"],
    "Trojan":   ["Achilles", "Patroclus", "Hektor"],
    "Resonant": ["Hilda", "Thule", "Griqua"],
    "TNO":      ["Pluto", "Eris", "Sedna"],
}
ASTEROID_ORDER = [a for names in CATEGORIES.values() for a in names]
INTEGRATORS_LT = ["AAS", "SABA4", "RKF78"]


def _grouped_bar(ax, df, test, ylabel, take_abs=False, semilogy=True):
    """Grouped bar chart over all 12 long-term objects, coloured by integrator."""
    n_ast = len(ASTEROID_ORDER)
    width = 0.25
    offsets = np.array([-width, 0.0, width])
    xs = np.arange(n_ast)
    for i, integ in enumerate(INTEGRATORS_LT):
        vals = []
        for ast in ASTEROID_ORDER:
            row = df[(df.asteroid == ast) & (df.integrator == integ) & (df.test == test)]
            v = row.value.values[0] if not row.empty else np.nan
            if take_abs:
                v = abs(v)
            vals.append(v)
        vals = np.array(vals, dtype=float)
        ax.bar(xs + offsets[i], vals, width=width * 0.9,
               color=COLORS[integ], label=integ, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(ASTEROID_ORDER, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    if semilogy:
        ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    cat_names = list(CATEGORIES.keys())
    boundaries = [0, 3, 6, 9, 12]
    for j, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        mid = (lo + hi - 1) / 2
        ax.axvspan(lo - 0.5, hi - 0.5, alpha=0.06 if j % 2 == 0 else 0.0,
                   color="steelblue", zorder=0)
        ax.text(mid, 1.02, cat_names[j], transform=ax.get_xaxis_transform(),
                fontsize=7, ha="center", style="italic", color="0.4")


# ── Figure L1 — Long-term energy conservation ────────────────────────────────

def fig_lt_energy(df):
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    _grouped_bar(ax, df, "energy",
                 r"$|\Delta H/H_0|$", take_abs=False, semilogy=True)
    ax.set_title("Final relative energy error (long-term)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_lt_energy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_lt_energy.pdf")


# ── Figure L2 — Time-reversal symmetry ───────────────────────────────────────

def fig_lt_reversibility(df):
    targets = ["Apophis", "Achilles", "Pluto"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    xs = np.arange(len(targets))
    width = 0.25
    for ax, test, label in [
        (axes[0], "reversibility_r", r"$\epsilon_r$"),
        (axes[1], "reversibility_v", r"$\epsilon_v$"),
    ]:
        for i, integ in enumerate(INTEGRATORS_LT):
            vals = []
            for a in targets:
                row = df[(df.asteroid == a) & (df.integrator == integ) & (df.test == test)]
                vals.append(row.value.values[0] if not row.empty else np.nan)
            ax.bar(xs + (i - 1) * width, vals, width=width * 0.9,
                   color=COLORS[integ], label=integ, alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(targets)
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    fig.suptitle("Time-reversal round-trip errors", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_lt_reversibility.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_lt_reversibility.pdf")


# ── Figure L3 — Secular drift analysis ───────────────────────────────────────

def fig_lt_secular(df):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8))
    _grouped_bar(axes[0], df, "secular_slope",
                 r"$|a|$ [yr$^{-1}$]", take_abs=True, semilogy=True)
    axes[0].set_title("Linear regression slope $|a|$")
    _grouped_bar(axes[1], df, "secular_R2",
                 r"$R^2$", take_abs=False, semilogy=False)
    axes[1].axhline(0.95, color="0.4", ls="--", lw=0.8, label="$R^2=0.95$")
    axes[1].set_ylim(0, 1.12)
    axes[1].set_title("Coefficient of determination $R^2$")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_lt_secular.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_lt_secular.pdf")


# ── Figure L4 — Maximum Lyapunov exponent ────────────────────────────────────

def fig_lt_lyapunov(df):
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    _grouped_bar(ax, df, "lyapunov",
                 r"$\lambda$ [yr$^{-1}$]", take_abs=False, semilogy=True)
    ax.set_title("Maximum Lyapunov characteristic exponent")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_lt_lyapunov.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_lt_lyapunov.pdf")


# ── Figure L5 — Lyapunov convergence ─────────────────────────────────────────

def fig_lt_lyapunov_convergence(df):
    targets = ["Apophis", "Griqua", "Achilles"]
    titles  = ["Apophis (regular, NEA)", "Griqua (chaotic, Resonant)",
               "Achilles (Trojan)"]
    conv_int = ["AAS", "RKF78"]
    ls_map   = {"AAS": "-", "RKF78": "--"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=False)
    for ax, ast, title in zip(axes, targets, titles):
        for integ in conv_int:
            sub = df[(df.asteroid == ast) & (df.integrator == integ)].sort_values("interval_index")
            if sub.empty:
                continue
            running = sub.lambda_i.cumsum().values / np.arange(1, len(sub) + 1)
            ax.semilogy(sub.interval_index.values, running,
                        color=COLORS[integ], ls=ls_map[integ], label=integ, lw=1.0)
        ax.set_xlabel("interval index")
        ax.set_ylabel(r"$\langle\lambda\rangle$ [yr$^{-1}$]")
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_lt_lyapunov_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_lt_lyapunov_convergence.pdf")


# ── Figure L6 — Jacobi constant — Jupiter Trojans ────────────────────────────

def fig_lt_jacobi(df):
    trojans = ["Achilles", "Patroclus", "Hektor"]
    xs = np.arange(len(trojans))
    width = 0.25
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for i, integ in enumerate(INTEGRATORS_LT):
        vals = []
        for a in trojans:
            row = df[(df.asteroid == a) & (df.integrator == integ) & (df.test == "jacobi")]
            vals.append(row.value.values[0] if not row.empty else np.nan)
        ax.bar(xs + (i - 1) * width, vals, width=width * 0.9,
               color=COLORS[integ], label=integ, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(trojans)
    ax.set_ylabel(r"Final $C_J$ [dimensionless]")
    ax.set_title("Jacobi constant — Jupiter Trojans (1000 yr)")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_lt_jacobi.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_lt_jacobi.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

def load(name):
    p = BASE / name
    if not p.exists():
        print(f"  MISSING: {p}")
        return None
    return pd.read_csv(p)


def main():
    print(f"Reading from:  {BASE}")
    print(f"Writing to:    {FIGURES}")

    df_energy    = load("energy_vs_precision.csv")
    df_div       = load("divergence_time.csv")
    df_step      = load("step_distribution.csv")
    df_shadow    = load("shadow_hamiltonian.csv")
    df_stm       = load("stm_accuracy.csv")
    df_horizons  = load("horizons_validation.csv")
    df_eff       = load("efficiency.csv")
    df_hzn_short = load("horizons_short.csv")
    df_uncert    = load("uncertainty.csv")
    df_lt        = load("long_term_tests.csv")
    df_lys       = load("lyapunov_series.csv")

    if df_energy    is not None: fig_energy(df_energy)
    if df_div       is not None: fig_divergence(df_div)
    if df_step      is not None: fig_step_distribution(df_step)
    if df_shadow    is not None: fig_shadow(df_shadow)
    if df_stm       is not None: fig_stm(df_stm)
    if df_horizons  is not None: fig_horizons(df_horizons)
    if df_eff       is not None: fig_efficiency(df_eff)
    if df_hzn_short is not None: fig_horizons_short(df_hzn_short)
    if df_uncert    is not None: fig_uncertainty(df_uncert)
    if df_lt        is not None: fig_lt_energy(df_lt)
    if df_lt        is not None: fig_lt_reversibility(df_lt)
    if df_lt        is not None: fig_lt_secular(df_lt)
    if df_lt        is not None: fig_lt_lyapunov(df_lt)
    if df_lys       is not None: fig_lt_lyapunov_convergence(df_lys)
    if df_lt        is not None: fig_lt_jacobi(df_lt)

    print("Done.")


if __name__ == "__main__":
    main()
