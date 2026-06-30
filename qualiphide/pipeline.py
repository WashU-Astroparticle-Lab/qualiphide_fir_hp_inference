"""Full inference pipeline orchestration.

For each mass in the test grid, this module:
1. Generates null ToyMC (chi=0) and computes fit results / discovery threshold.
2. For each nonzero chi: generates signal ToyMC, computes fit results, and evaluates
   the 3-sigma discovery probability.
3. Dynamically extends the chi grid when discovery probability hasn't yet reached 1.
4. Computes sensitivity (PLR upper limits) and plots Brazil-band / heatmap figures.
5. Supports resuming — checks for existing output files and skips completed steps.
"""

import glob
import os
import sys
import numpy as np
import gc
from tqdm import tqdm

# Prevent macOS from aborting on fork() after Cocoa/ObjC framework init.
# This must be set before any multiprocessing fork happens.
if sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from qualiphide import format_mass, format_chi
from qualiphide.config import load_config, load_derived_data
from qualiphide.toymc import generate_toymc, load_toymc, load_channel_data
from qualiphide.minimization import (
    compute_fit_results, plot_fit_results, round_chi,
)
from qualiphide.sensitivity import compute_sensitivity, plot_sensitivity_band
from qualiphide.discovery import (
    compute_discovery_probability, plot_discovery_heatmap,
    save_discovery_thresholds,
)


def load_discovery_results(csv_path):
    """Load rows from the three_sigma_discovery.csv results file.

    Parameters
    ----------
    csv_path : str
        Path to ``three_sigma_discovery.csv``.

    Returns
    -------
    list[list]
        Each inner list is ``[m, chi, q_disc, probability]``.
    """
    if not os.path.exists(csv_path):
        return []
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.size == 0:
        return []
    if data.shape == (4,):
        data = np.array([data])
    return [list(row) for row in data]


def find_completed_chi_values(out, m, chi_values):
    """Return the subset of *chi_values* that already have fit result files.

    Parameters
    ----------
    out : str
        Output directory name.
    m : float
        Hidden photon mass (eV).
    chi_values : np.ndarray
        Candidate chi values.

    Returns
    -------
    np.ndarray
        Chi values with existing fit result CSVs.
    """
    fit_dir = f"{out}/fit_results"
    if not os.path.isdir(fit_dir):
        return np.array([], dtype=float)
    available = []
    for c in chi_values:
        fit_path = f"{fit_dir}/fit_results_{format_mass(m)}_{round_chi(c)}.csv"
        if os.path.exists(fit_path):
            available.append(c)
    return np.array(available, dtype=float)


def get_stored_discovery_probability(m_c_q0_prob, m, c, atol=1e-15):
    """Look up a previously computed discovery probability.

    Parameters
    ----------
    m_c_q0_prob : list[list]
        Rows from ``three_sigma_discovery.csv``.
    m : float
        Mass (eV).
    c : float
        Chi value.
    atol : float
        Tolerance for chi comparison.

    Returns
    -------
    float or None
        Stored probability, or ``None`` if not found.
    """
    for row in m_c_q0_prob:
        if row[0] == m and np.isclose(row[1], c, atol=atol):
            return row[3]
    return None


def build_sensitivity_chi_bounds(chi_true, m_0_c_stored):
    """Determine chi bracket for the sensitivity bisection.

    Parameters
    ----------
    chi_true : np.ndarray
        Chi grid from config (includes 0).
    m_0_c_stored : list[list]
        Three-sigma rows for the current mass.

    Returns
    -------
    tuple of float
        ``(chi_lo, chi_hi)`` bracket for the bisection search.
    """
    chi_pos = chi_true[chi_true > 1e-15]
    chi_lo = float(np.min(chi_pos)) if chi_pos.size else 1e-20
    chi_prob1 = [
        float(row[1])
        for row in m_0_c_stored
        if np.isclose(row[3], 1.0, rtol=0, atol=1e-9)
    ]
    if chi_prob1:
        chi_hi = min(chi_prob1)
    else:
        max_disc = max(row[3] for row in m_0_c_stored)
        chi_hi = min(
            float(row[1]) for row in m_0_c_stored if row[3] == max_disc
        )
    return (chi_lo / 4, 4 * chi_hi)


def run_pipeline(name, keep_toymc=False):
    """Run the full inference pipeline for a given configuration.

    Parameters
    ----------
    name : str
        Config name (without ``.yaml`` extension).
    keep_toymc : bool
        If True, save all ToyMC datasets instead of compacting to 50 per file.
    """
    cfg = load_config(name)
    out = cfg.output_dir
    for subdir in ["", "/ToyMC", "/fit_results", "/sensitivity", "/three_sigma_discovery"]:
        os.makedirs(f"{out}{subdir}", exist_ok=True)

    chi_true_full = cfg.chi_true

    # Load background channels CSV once (avoids repeated file I/O)
    channel_data = load_channel_data()

    m_done = 0
    mass_bar = tqdm(cfg.m_test, desc="Mass points", unit="mass")
    for m in mass_bar:
        mass_bar.set_postfix(m=f"{m*1e3:.1f} meV")
        interp_data, res_data = load_derived_data(out, m)

        # Base: every chi in the yaml grid with chi <= 1e-11
        chi_true = chi_true_full[chi_true_full <= 1e-11].astype(float)
        take_more_idx = len(chi_true)
        # Add any chi for this m from three_sigma_discovery.csv not already in chi_true
        three_sigma_discovery_csv = f"{out}/three_sigma_discovery/three_sigma_discovery.csv"
        if os.path.exists(three_sigma_discovery_csv):
            for row in load_discovery_results(three_sigma_discovery_csv):
                if np.isclose(row[0], m, rtol=0, atol=1e-15):
                    c_csv = round_chi(row[1])
                    if not np.any(
                        np.isclose(chi_true, c_csv, rtol=0, atol=1e-14)
                    ):
                        chi_true = np.append(chi_true, c_csv)
            chi_true = np.sort(chi_true)
        sens_path = f"{out}/sensitivity/sensitivity_{format_mass(m)}.csv"
        if os.path.exists(sens_path):
            try:
                existing = np.loadtxt(sens_path, delimiter=",", skiprows=1)
                if existing.ndim == 2 and len(existing) == cfg.n_toy:
                    plot_fit_results(m, chi_true, out)
                    m_done += 1
                    plot_sensitivity_band(cfg.m_test[:m_done:], "log", out)
                    plot_discovery_heatmap(
                        cfg.m_test[:m_done:],
                        chi_true_full[1::],
                        "log",
                        out,
                    )
                    save_discovery_thresholds(out)
                    continue
            except Exception:
                pass
        null_file = f"{out}/ToyMC/ToyMC_{format_mass(m)}_0.0.npy"
        if os.path.exists(null_file):
            null_toy = load_toymc(null_file)
            if len(null_toy) < cfg.n_toy:
                # Compacted ToyMC means sensitivity already completed for
                # a prior mass that rounds to the same format_mass string.
                # Skip rather than recompute — the results are already on
                # disk.
                m_done += 1
                continue
        else:
            null_toy = generate_toymc(
                cfg.n_toy, 0.0, m, cfg.physics,
                interp_data, res_data, cfg.cutoffs,
                channel_data=channel_data,
            )
            np.save(null_file, np.array(null_toy, dtype=object))
        q_disc, nll_max_null = compute_fit_results(
            null_toy, 0.0, m, cfg.physics,
            out, interp_data, res_data, cfg.cutoffs, cfg.fit_bounds,
        )
        chi_bar = tqdm(chi_true_full[1::], desc="  Chi scan", unit="chi", leave=False)
        for c in chi_bar:
            chi_bar.set_postfix(chi=f"{c:.2e}")
            m_c_q0_prob = []
            m_c_stored_c = []
            if os.path.exists(f"{out}/three_sigma_discovery/three_sigma_discovery.csv"):
                data = np.loadtxt(
                    f"{out}/three_sigma_discovery/three_sigma_discovery.csv", delimiter=",", skiprows=1
                )
                if data.shape == (4,):
                    data = [data]
                for row in data:
                    m_c_q0_prob.append(list(row))
                m_c_stored = [
                    [m_c_q0_prob[i][0], round_chi(m_c_q0_prob[i][1])]
                    for i in range(len(m_c_q0_prob))
                ]
                m_c_stored_c = np.array(
                    [mc for mc in m_c_stored if m in mc]
                )
                if len(m_c_stored_c) != 0:
                    m_c_stored_c = m_c_stored_c[:, 1]
            tolerance = 10 ** (int(str(c)[-3:]) - 1)
            if len(m_c_q0_prob) != 0 or c not in chi_true:
                if c not in chi_true:
                    if not np.any(
                        np.isclose(m_c_stored_c, c, atol=tolerance)
                    ):
                        if np.max(chi_true) != np.max(chi_true_full):
                            m_c_q0_prob.append(
                                [m, round_chi(c), q_disc, 1]
                            )
                        else:
                            m_c_q0_prob.append(
                                [m, round_chi(c), q_disc, 0]
                            )
                    m_c_q0_prob = sorted(
                        m_c_q0_prob, key=lambda x: [x[0], x[1]]
                    )
                    with open(
                        f"{out}/three_sigma_discovery/three_sigma_discovery.csv", "w"
                    ) as f:
                        f.write("mass_eV,chi,q_disc,probability\n")
                        for row in m_c_q0_prob:
                            f.write(
                                f"{row[0]},{row[1]},{row[2]},{row[3]}\n"
                            )
                    continue
                elif np.any(
                    np.isclose(m_c_stored_c, c, atol=tolerance)
                ):
                    continue
            full_toy = generate_toymc(
                cfg.n_toy, c, m, cfg.physics,
                interp_data, res_data, cfg.cutoffs,
                channel_data=channel_data,
            )
            debug_file = f"{out}/ToyMC/ToyMC_{format_mass(m)}_{c}.npy"
            if keep_toymc:
                np.save(debug_file, np.array(full_toy, dtype=object))
            else:
                n_keep = min(50, len(full_toy))
                np.save(debug_file, np.array(full_toy[:n_keep], dtype=object))
            _, nll_max_full = compute_fit_results(
                full_toy, c, m, cfg.physics,
                out, interp_data, res_data, cfg.cutoffs, cfg.fit_bounds,
            )
            prob = compute_discovery_probability(
                full_toy,
                cfg.n_toy,
                c,
                m,
                chi_true[1::],
                cfg.physics,
                out,
                q_disc,
                interp_data,
                res_data,
                cfg.cutoffs,
                nll_max_full,
                cfg.fit_bounds,
            )
            del full_toy
            if (
                c == chi_true[-1]
                and prob != 1
                and c != chi_true_full[-1]
            ):
                chi_true = np.array(
                    list(chi_true) + [chi_true_full[take_more_idx]]
                )
                take_more_idx += 1
            gc.collect()
        plot_fit_results(m, chi_true, out)
        m_c_q0_prob = []
        data = np.loadtxt(
            f"{out}/three_sigma_discovery/three_sigma_discovery.csv", delimiter=",", skiprows=1
        )
        if data.shape == (4,):
            data = [data]
        for row in data:
            m_c_q0_prob.append(list(row))
        m_0_c_stored = [mc for mc in m_c_q0_prob if mc[0] == m]
        chi_bounds = build_sensitivity_chi_bounds(chi_true, m_0_c_stored)
        compute_sensitivity(
            null_toy,
            cfg.n_toy,
            m,
            cfg.physics,
            out,
            chi_bounds,
            interp_data,
            res_data,
            cfg.cutoffs,
            nll_max_null,
            cfg.fit_bounds,
        )
        if not keep_toymc:
            n_keep = min(50, len(null_toy))
            np.save(null_file, np.array(null_toy[:n_keep], dtype=object))
        del null_toy
        # Clean up nll_max .npy files for this mass — no longer needed
        for nll_file in glob.glob(f"{out}/fit_results/nll_max_{format_mass(m)}_*.npy"):
            os.remove(nll_file)
        m_done += 1
        plot_sensitivity_band(cfg.m_test[:m_done:], "log", out)
        plot_discovery_heatmap(
            cfg.m_test[:m_done:], chi_true_full[1::], "log", out
        )
        save_discovery_thresholds(out)
        gc.collect()
