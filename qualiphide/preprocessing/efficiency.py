"""Detector efficiency interpolation from raw calibration data.

Reads ``total_efficiency_sr3pt2.csv`` and interpolates the full efficiency,
aperture efficiency (eta_a), and remaining efficiency components onto a
uniform energy grid.  Results are written to ``<name>/eta_full.csv``,
``<name>/eta_a.csv``, and ``<name>/eta_rest.csv``.
"""

import numpy as np
import matplotlib.pyplot as plt

from qualiphide import DATA_DIR


def interpolate_efficiency(name, cutoff, maximum):
    """Interpolate detector efficiency curves and save to CSV.

    Parameters
    ----------
    name : str
        Output directory name.
    cutoff : float
        Lower energy cutoff (eV).
    maximum : float
        Upper energy cutoff (eV).
    """
    final_energies = np.arange(0, maximum + 1e-5, 1e-5)
    eta_full = np.loadtxt(
        str(DATA_DIR / "total_efficiency_sr3pt2.csv"), delimiter=",", skiprows=2
    )
    init_data = []
    for row in eta_full:
        init_data.append([row[0], row[2], row[5], row[8] * row[10] * row[11]])
    init_data = np.transpose(sorted(init_data, key=lambda x: x[0]))
    init_energies = init_data[0]
    eta_full_median = np.interp(
        final_energies, init_energies / 1000, init_data[1]
    )
    eta_a_median = np.interp(
        final_energies, init_energies / 1000, init_data[2]
    )
    eta_rest_median = np.interp(
        final_energies, init_energies / 1000, init_data[3]
    )
    mask = (final_energies >= cutoff) & (final_energies <= maximum)
    eta_full_median = np.where(mask, eta_full_median, 0)
    eta_a_median = np.where(mask, eta_a_median, 0)
    eta_rest_median = np.where(mask, eta_rest_median, 0)

    plt.plot(final_energies, eta_full_median, label="full eta")
    plt.plot(final_energies, eta_a_median, label="eta_a")
    plt.plot(final_energies, eta_rest_median, label="rest of eta")
    plt.legend()
    plt.grid()
    plt.xlabel("Energy (eV)")
    plt.savefig(f"{name}/eta_all.png")
    plt.close()

    with open(f"{name}/eta_full.csv", "w") as f:
        f.write("energy_eV,eta_full\n")
        for i, a in enumerate(eta_full_median):
            f.write(f"{final_energies[i]},{a}\n")

    with open(f"{name}/eta_a.csv", "w") as f:
        f.write("energy_eV,eta_a\n")
        for i, b in enumerate(eta_a_median):
            f.write(f"{final_energies[i]},{b}\n")

    with open(f"{name}/eta_rest.csv", "w") as f:
        f.write("energy_eV,eta_rest\n")
        for i, c in enumerate(eta_rest_median):
            f.write(f"{final_energies[i]},{c}\n")
