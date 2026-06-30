"""Coverage validation for PLR confidence intervals.

For a given injected signal chi_true, coverage measures what fraction of
toys produce intervals that actually contain the true value.  This follows
the methodology in Figure 11 of arXiv:1902.11297.

The implementation reuses the Neyman threshold (q0) from a prior sensitivity
run rather than recomputing it.
"""

import gc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Prevent macOS from aborting on fork() after Cocoa/ObjC framework init.
if sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from qualiphide import format_mass, format_chi
from qualiphide.config import load_config, load_derived_data
from qualiphide.toymc import generate_toymc, load_channel_data
from qualiphide.minimization import minimize_unconstrained
from qualiphide.sensitivity import _bisect_all_toys, load_q0_interpolator


def run_coverage(name):
    """Run coverage validation for a given configuration.

    Parameters
    ----------
    name : str
        Config name (without ``.yaml`` extension).  Used to locate the YAML
        file and as the output directory name.
    """
    cfg = load_config(name)
    out = cfg.output_dir

    if cfg.chi_true_coverage is None or len(cfg.chi_true_coverage) == 0:
        raise ValueError(
            "chi_true_coverage is missing or empty in the YAML config. "
            "Add a list of chi values under 'chi_true_coverage' to run "
            "coverage mode."
        )

    # Use m_test_coverage if provided, otherwise fall back to full m_test
    m_test = cfg.m_test_coverage if cfg.m_test_coverage is not None else cfg.m_test

    # Verify all q0 threshold files exist before starting
    missing = []
    for m in m_test:
        q0_path = f"{out}/fit_results/q0_{format_mass(m)}.csv"
        if not os.path.exists(q0_path):
            missing.append(f"  {q0_path}  (m = {m*1e3:.2f} meV)")
    if missing:
        raise FileNotFoundError(
            "Neyman threshold (q0) files are missing. "
            "Run sensitivity first.\n" + "\n".join(missing)
        )

    os.makedirs(f"{out}/coverage", exist_ok=True)
    channel_data = load_channel_data()

    # Load existing coverage results for resuming
    results_path = f"{out}/coverage/coverage_results.csv"
    existing = _load_existing_results(results_path)

    mass_bar = tqdm(m_test, desc="Mass points", unit="mass")
    for m in mass_bar:
        mass_bar.set_postfix(m=f"{m*1e3:.1f} meV")
        interp_data, res_data = load_derived_data(out, m)

        q0_path = f"{out}/fit_results/q0_{format_mass(m)}.csv"
        q0_func = load_q0_interpolator(out, m)

        # Read q0 file to get the chi range used in the sensitivity run.
        # The q0 file contains chi=0 from the null hypothesis fit; we
        # exclude it when computing q0_chi_min solely to avoid 0/4 = 0
        # in the chi_lo calculation below.  This does NOT limit the
        # bisection search range — chi_lo is set to
        # min(q0_chi_min, chi_true) / 4, so if chi_true is smaller than
        # the sensitivity grid the bisection will search well below it.
        # The q0 interpolator (load_q0_interpolator) flat-extrapolates
        # outside the file's chi range, so the threshold remains valid.
        q0_data = np.loadtxt(q0_path, delimiter=",", skiprows=1)
        q0_chi_all = q0_data[:, 0]
        q0_chi_min = q0_chi_all[q0_chi_all > 0].min()
        q0_chi_max = q0_chi_all.max()

        chi_bar = tqdm(
            cfg.chi_true_coverage, desc="  Coverage chi",
            unit="chi", leave=False,
        )
        for chi_true in chi_bar:
            chi_bar.set_postfix(chi=f"{chi_true:.2e}")

            # Check if already computed
            if _is_completed(existing, m, chi_true):
                continue

            # Bisection bracket: extend beyond both the sensitivity grid
            # and chi_true so the crossing is always interior to the
            # bracket.  The /4 and *4 factors provide margin in log-space.
            chi_lo = min(q0_chi_min, chi_true) / 4
            chi_hi = max(q0_chi_max, chi_true) * 4
            chi_bounds = (chi_lo, chi_hi)

            # Generate signal-injected ToyMC
            toy = generate_toymc(
                cfg.n_toy, chi_true, m, cfg.physics,
                interp_data, res_data, cfg.cutoffs,
                channel_data=channel_data,
            )

            # Unconstrained fit
            nll_max, chi_hat, _ = minimize_unconstrained(
                toy, cfg.physics, m, chi_true,
                interp_data, res_data, cfg.cutoffs, cfg.fit_bounds,
            )

            # Bisection for LL and UL
            LL, UL, _ = _bisect_all_toys(
                toy, cfg.n_toy, m, cfg.physics, chi_bounds,
                interp_data, res_data, cfg.cutoffs,
                nll_max, q0_func, cfg.fit_bounds,
                chi_hat=chi_hat,
            )

            # Save per-toy limits
            toy_path = (
                f"{out}/coverage/"
                f"coverage_{format_mass(m)}_{format_chi(chi_true)}.csv"
            )
            with open(toy_path, "w") as f:
                f.write("lower_limit,upper_limit\n")
                for ll, ul in zip(LL, UL):
                    f.write(f"{ll},{ul}\n")

            # Compute coverage
            analysis_cov = float(np.mean(UL > chi_true))
            analysis_unc = np.sqrt(
                analysis_cov * (1 - analysis_cov) / cfg.n_toy
            )
            interval_cov = float(
                np.mean((LL <= chi_true) & (chi_true <= UL))
            )
            interval_unc = np.sqrt(
                interval_cov * (1 - interval_cov) / cfg.n_toy
            )

            # Append to results CSV
            write_header = not os.path.exists(results_path)
            with open(results_path, "a") as f:
                if write_header:
                    f.write(
                        "mass_eV,chi_true,n_toy,"
                        "analysis_coverage,analysis_unc,"
                        "interval_coverage,interval_unc\n"
                    )
                f.write(
                    f"{m},{chi_true},{cfg.n_toy},"
                    f"{analysis_cov},{analysis_unc},"
                    f"{interval_cov},{interval_unc}\n"
                )

            # Update existing set for resume tracking
            existing.add((m, chi_true))

            del toy
            gc.collect()

        # Plot coverage vs chi for this mass
        _plot_coverage_vs_chi(out, m)


def _load_existing_results(results_path):
    """Load (mass, chi_true) pairs from an existing coverage_results.csv."""
    completed = set()
    if os.path.exists(results_path):
        data = np.loadtxt(results_path, delimiter=",", skiprows=1)
        if data.size == 0:
            return completed
        if data.ndim == 1:
            data = data.reshape(1, -1)
        for row in data:
            completed.add((row[0], row[1]))
    return completed


def _is_completed(existing, m, chi_true):
    """Check whether (m, chi_true) is already in the completed set."""
    for em, ec in existing:
        if np.isclose(em, m, rtol=0, atol=1e-15) and np.isclose(
            ec, chi_true, rtol=0, atol=1e-15
        ):
            return True
    return False


def _plot_coverage_vs_chi(name, m):
    """Plot coverage vs chi_true for a given mass, with error bars."""
    results_path = f"{out}/coverage/coverage_results.csv"
    if not os.path.exists(results_path):
        return
    data = np.loadtxt(results_path, delimiter=",", skiprows=1)
    if data.size == 0:
        return
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Filter to this mass
    mask = np.isclose(data[:, 0], m, rtol=0, atol=1e-15)
    if not np.any(mask):
        return
    rows = data[mask]
    if len(rows) < 1:
        return

    chi_vals = rows[:, 1]
    order = np.argsort(chi_vals)
    chi_vals = chi_vals[order]

    analysis_cov = rows[order, 3]
    analysis_unc = rows[order, 4]
    interval_cov = rows[order, 5]
    interval_unc = rows[order, 6]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.errorbar(
        chi_vals, analysis_cov, yerr=analysis_unc,
        fmt="o-", capsize=3, label="Analysis coverage (UL > truth)",
    )
    ax.errorbar(
        chi_vals, interval_cov, yerr=interval_unc,
        fmt="s-", capsize=3, label="Interval coverage (LL < truth < UL)",
    )
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90% CL")
    ax.set_xscale("log")
    ax.set_xlabel("$\\chi_{\\mathrm{true}}$")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Coverage, $m_{{\\mathrm{{test}}}}$ = {m * 1000:.2f} meV"
    )
    ax.legend(fontsize=8)
    plt.savefig(
        f"{out}/coverage/coverage_vs_chi_{format_mass(m)}.png", dpi=150,
    )
    plt.close()
