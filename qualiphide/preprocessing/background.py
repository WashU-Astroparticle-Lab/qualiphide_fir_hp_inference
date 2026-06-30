"""Background spectra processing and PDF construction.

Reads per-channel background spectra from ``sr3pt2_bkg_channels_Hz_meV_remade.csv``,
computes 16th/50th/84th percentile envelopes, flattens above the ``split`` energy,
and splits into below/above ``split_mu_b`` normalised PDFs.  Results are written to
``<name>/f_b_below.csv`` and ``<name>/f_b_above.csv``.
"""

import numpy as np
import matplotlib.pyplot as plt

from qualiphide import DATA_DIR


def process_background_spectra(name, cutoff, split_mu_b, split, maximum):
    """Process background channel spectra and write normalised PDFs.

    Parameters
    ----------
    name : str
        Output directory name.
    cutoff : float
        Lower energy cutoff (eV).
    split_mu_b : float
        Energy separating below/above background regions (eV).
    split : float
        Energy above which the PDF is flattened (eV).
    maximum : float
        Upper energy cutoff (eV).
    """
    data = np.loadtxt(
        str(DATA_DIR / "sr3pt2_bkg_channels_Hz_meV_remade.csv"),
        delimiter=",",
        skiprows=1,
    )

    energies = data[:, 1] / 1000
    final_energies = np.arange(0.0, maximum + 1e-5, 1e-5)
    full_channels = np.transpose(
        [
            data[:, c]
            for c in range(2, len(data[0]))
            if np.trapezoid(data[:, c], x=energies) != 0.0
        ]
    )
    pct_84, pct_50, pct_16 = [], [], []
    for i, row in enumerate(full_channels):
        pct_84.append(np.percentile(row, 84))
        pct_50.append(np.percentile(row, 50))
        pct_16.append(np.percentile(row, 16))
    mask = (final_energies >= cutoff) & (final_energies <= maximum)
    pct_84 = np.where(mask, np.interp(final_energies, energies, pct_84), 0)
    pct_50 = np.where(mask, np.interp(final_energies, energies, pct_50), 0)
    pct_16 = np.where(mask, np.interp(final_energies, energies, pct_16), 0)
    norm = np.trapezoid(pct_50, x=final_energies)
    pct_84, pct_50, pct_16 = (
        np.array(pct_84) / norm,
        np.array(pct_50) / norm,
        np.array(pct_16) / norm,
    )
    pcts = [pct_16, pct_50, pct_84]
    mask_flat = final_energies >= split
    for i, p in enumerate(pcts):
        p_flat = np.where(mask_flat, p, 0)
        p_not_flat = np.where(~mask_flat, p, 0)
        N_flat = np.trapezoid(p_flat, x=final_energies)
        weights = np.gradient(final_energies)
        flat_value = N_flat / np.sum(weights[mask_flat])
        p_flat = np.where(mask_flat, flat_value, 0)
        pcts[i] = p_flat + p_not_flat
    pcts_below, pcts_above = [], []
    for p in pcts:
        mask_below = final_energies < split_mu_b
        mask_above = ~mask_below

        p_below = np.where(mask_below, p, 0.0)
        p_above = np.where(mask_above, p, 0.0)

        p_below /= np.trapezoid(p_below, x=final_energies)
        p_above /= np.trapezoid(p_above, x=final_energies)

        pcts_below.append(p_below)
        pcts_above.append(p_above)

    with open(f"{name}/f_b_below.csv", "w") as f:
        f.write("energy_eV,pct16,pct50,pct84\n")
        for i, e in enumerate(final_energies):
            f.write(
                f"{e},{pcts_below[0][i]},{pcts_below[1][i]},{pcts_below[2][i]}"
            )
            if i != len(final_energies) - 1:
                f.write("\n")
    plt.plot(final_energies, pcts_below[0], color="black")
    plt.plot(final_energies, pcts_below[1], color="black")
    plt.plot(final_energies, pcts_below[2], color="black")
    plt.fill_between(
        final_energies, pcts_below[0], pcts_below[2], color="yellow"
    )
    plt.xlabel("E (eV)")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.grid()
    plt.savefig(f"{name}/f_b_below.png")
    plt.close()

    with open(f"{name}/f_b_above.csv", "w") as f:
        f.write("energy_eV,pct16,pct50,pct84\n")
        for i, e in enumerate(final_energies):
            f.write(
                f"{e},{pcts_above[0][i]},{pcts_above[1][i]},{pcts_above[2][i]}"
            )
            if i != len(final_energies) - 1:
                f.write("\n")
    plt.plot(final_energies, pcts_above[0], color="black")
    plt.plot(final_energies, pcts_above[1], color="black")
    plt.plot(final_energies, pcts_above[2], color="black")
    plt.fill_between(
        final_energies, pcts_above[0], pcts_above[2], color="yellow"
    )
    plt.xlabel("E (eV)")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.grid()
    plt.savefig(f"{name}/f_b_above.png")
    plt.close()
