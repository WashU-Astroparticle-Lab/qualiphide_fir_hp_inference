"""Three-sigma discovery probability computation.

For each (mass, chi) point, evaluates the fraction of signal toys whose
test statistic *q* exceeds the null discovery threshold *q_disc*
(the 99.865th percentile of the null q distribution).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from qualiphide import format_mass
from qualiphide.minimization import minimize_constrained, round_chi
import gc


def compute_discovery_probability(ToyMC_full, n_toy, c, m, chi_true, physics, name,
                                  q_disc, interp_data, res_data, cutoffs, nll_max_full,
                                  fit_bounds=None):
    """Compute the 3-sigma discovery probability for a (mass, chi) point.

    Parameters
    ----------
    ToyMC_full : list
        Signal toy data for this (mass, chi).
    n_toy : int
        Number of toys.
    c : float
        Kinetic mixing parameter chi.
    m : float
        Hidden photon mass (eV).
    chi_true : np.ndarray
        Non-zero chi grid (used to check if previous chi had prob=1).
    physics : PhysicsConstants
        Experiment constants.
    name : str
        Output directory name.
    q_disc : float
        Discovery threshold from the null q distribution.
    interp_data : InterpolationData
        Interpolated data products.
    res_data : ResolutionData
        Resolution model.
    cutoffs : EnergyCutoffs
        Energy boundaries.
    nll_max_full : np.ndarray
        Unconstrained NLL for the signal toys.

    Returns
    -------
    float
        Discovery probability (fraction of toys with q > q_disc).
    """
    c_index = list(chi_true).index(c)
    os.makedirs(f"{name}/three_sigma_discovery", exist_ok=True)
    data = []
    m_c_q0_prob = []
    if os.path.exists(f"{name}/three_sigma_discovery/three_sigma_discovery.csv"):
        data = np.loadtxt(
            f"{name}/three_sigma_discovery/three_sigma_discovery.csv", delimiter=",", skiprows=1
        )
        if data.shape == (4,):
            data = [data]
        for row in data:
            m_c_q0_prob.append(list(row))
    m_c_stored = [
        [m_c_q0_prob[i][0], round_chi(m_c_q0_prob[i][1])]
        for i in range(len(m_c_q0_prob))
    ]
    if len(m_c_q0_prob) != 0:
        if [m, round_chi(c)] in m_c_stored:
            gc.collect()
            return 1.0
    if c_index != 0:
        if [m, round_chi(chi_true[c_index - 1])] in m_c_stored:
            m_c_prev = m_c_stored.index(
                [m, round_chi(chi_true[c_index - 1])]
            )
            if m_c_q0_prob[m_c_prev][3] == 1:
                m_c_q0_prob.append([m, round_chi(c), q_disc, 1])
                m_c_q0_prob = sorted(
                    m_c_q0_prob, key=lambda x: [x[0], x[1]]
                )
                with open(
                    f"{name}/three_sigma_discovery/three_sigma_discovery.csv", "w"
                ) as f:
                    f.write("mass_eV,chi,q_disc,probability\n")
                    for term in m_c_q0_prob:
                        f.write(
                            f"{term[0]},{term[1]},{term[2]},{term[3]}\n"
                        )
                gc.collect()
                return 1.0
    nll_out_full = minimize_constrained(
        ToyMC_full, physics, m, 0,
        interp_data, res_data, cutoffs, fit_bounds=fit_bounds,
    )[0]
    q_full = 2 * (nll_out_full - nll_max_full)
    q_full = np.sort(q_full)[::-1]
    q_above = 0
    while q_full[q_above] >= q_disc:
        q_above += 1
        if q_above == n_toy:
            break
    prob = q_above / n_toy
    if prob > 0.99:
        prob = 1
    elif prob < 0.01:
        prob = 0
    m_c_q0_prob.append([m, round_chi(c), q_disc, prob])
    m_c_q0_prob = sorted(m_c_q0_prob, key=lambda x: [x[0], x[1]])
    with open(f"{name}/three_sigma_discovery/three_sigma_discovery.csv", "w") as f:
        f.write("mass_eV,chi,q_disc,probability\n")
        for term in m_c_q0_prob:
            f.write(f"{term[0]},{term[1]},{term[2]},{term[3]}\n")
    gc.collect()
    return prob


def save_discovery_thresholds(name, thresholds=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Find chi values at given discovery probability thresholds for each mass.

    For each mass, interpolates chi vs discovery probability to find the chi
    value at which each threshold is crossed. Results are saved to
    ``<name>/three_sigma_discovery/discovery_thresholds.csv``.

    Parameters
    ----------
    name : str
        Output directory name.
    thresholds : tuple of float
        Discovery probability thresholds to evaluate.
    """
    csv_path = f"{name}/three_sigma_discovery/three_sigma_discovery.csv"
    if not os.path.exists(csv_path):
        return
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.size == 0:
        return
    if data.ndim == 1:
        data = np.array([data])

    # Group by mass
    masses = np.unique(data[:, 0])
    rows = []
    for m in masses:
        mask = data[:, 0] == m
        chi_vals = data[mask, 1]
        prob_vals = data[mask, 3]
        # Sort by chi
        order = np.argsort(chi_vals)
        chi_vals = chi_vals[order]
        prob_vals = prob_vals[order]

        row = [m]
        for thr in thresholds:
            chi_at_thr = np.nan
            # Find first crossing where probability >= threshold
            for i in range(len(prob_vals) - 1):
                p0, p1 = prob_vals[i], prob_vals[i + 1]
                if p0 <= thr <= p1 and p1 > p0:
                    # Linear interpolation in log-chi space
                    frac = (thr - p0) / (p1 - p0)
                    log_chi = np.log10(chi_vals[i]) + frac * (
                        np.log10(chi_vals[i + 1]) - np.log10(chi_vals[i])
                    )
                    chi_at_thr = 10 ** log_chi
                    break
            row.append(chi_at_thr)
        rows.append(row)

    out_path = f"{name}/three_sigma_discovery/discovery_thresholds.csv"
    header = "mass_eV," + ",".join(f"chi_{int(t*100)}pct" for t in thresholds)
    with open(out_path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def plot_discovery_heatmap(m_test, chi_true, plot_scale, name):
    """Plot the discovery probability heatmap.

    Parameters
    ----------
    m_test : np.ndarray
        Array of test masses (eV).
    chi_true : np.ndarray
        Non-zero chi grid.
    plot_scale : str
        Axis scale (``"log"`` or ``"linear"``).
    name : str
        Output directory name.
    """
    sig_data = np.loadtxt(
        f"{name}/three_sigma_discovery/three_sigma_discovery.csv", delimiter=",", skiprows=1
    )
    m_c_q0_prob = sorted(
        [s for s in sig_data], key=lambda x: [x[0], x[1]]
    )
    with open(f"{name}/three_sigma_discovery/three_sigma_discovery.csv", "w") as f:
        f.write("mass_eV,chi,q_disc,probability\n")
        for term in m_c_q0_prob:
            f.write(f"{term[0]},{term[1]},{term[2]},{term[3]}\n")
    m_c_q0_prob = [mcqp for mcqp in m_c_q0_prob if mcqp[0] in m_test]
    m_c_q0_prob_T = np.array(m_c_q0_prob).T
    prob_list = m_c_q0_prob_T[3]
    median = []
    for m in m_test:
        limits_m = np.loadtxt(
            f"{name}/sensitivity/sensitivity_{format_mass(m)}.csv", delimiter=",", skiprows=1
        )
        limits_m = [u for u in limits_m[:, 1] if u != 0]
        median.append(np.percentile(limits_m, 50))
    three_sigma_discovery_array = np.array(prob_list).reshape(
        (len(m_test), len(chi_true))
    ).T

    _, ax = plt.subplots()
    im = ax.pcolormesh(
        m_test,
        chi_true,
        three_sigma_discovery_array,
        cmap="hot",
        vmin=0,
        vmax=1,
        shading="nearest",
    )
    ax.plot(m_test, median, color="cyan", linewidth=1.5, label="Median sensitivity")
    ax.legend(fontsize=8, loc="upper left")
    plt.ylim(chi_true[0], chi_true[-1])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Discovery probability")
    plt.xlabel("$m_{\\gamma'}$ (eV)")
    plt.ylabel("$\\chi$")
    plt.title("$>3\\sigma$ Discovery Probability")
    m_tick_labels = [f"{m:.3g}" for m in m_test[1::2]]
    chi_tick_labels = [f"{c:.3g}" for c in chi_true[1::2]]
    plt.xticks(m_test[1::2], m_tick_labels)
    plt.yticks(chi_true[1::2], chi_tick_labels)
    plt.xscale(plot_scale)
    plt.yscale(plot_scale)
    plt.savefig(f"{name}/three_sigma_discovery.png")
    plt.close()
