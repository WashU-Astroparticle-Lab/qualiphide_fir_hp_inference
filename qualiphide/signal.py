"""Signal probability density function for hidden photon dark matter.

The signal is modelled as a Gaussian peaked at the hidden photon mass *m*,
smeared by the energy resolution, and weighted by the aperture efficiency
curve eta_a.  Above the ``split`` energy the PDF is flattened (uniform)
while preserving the integrated probability.
"""

import numpy as np


def gaussian(x, mean, std):
    """Normalised Gaussian evaluated at *x*."""
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )


def generate_signal_pdf(mass, cutoff, split, maximum, resolution, energies, eta_a):
    """Build the signal PDF f_s for a given mass and resolution.

    Parameters
    ----------
    mass : float
        Hidden photon mass (eV).
    cutoff : float
        Lower energy cutoff (eV).
    split : float
        Energy above which the PDF is flattened (eV).
    maximum : float
        Upper energy cutoff (eV).
    resolution : float
        Energy resolution sigma (eV).
    energies : np.ndarray
        Energy grid (eV).
    eta_a : np.ndarray
        Aperture efficiency evaluated on *energies*.

    Returns
    -------
    np.ndarray
        Normalised signal PDF on the energy grid.
    """
    mask = (energies >= cutoff) & (energies <= maximum)

    f_s_orig = gaussian(energies, mass, resolution)
    f_s = f_s_orig * eta_a
    f_s = np.where(mask, f_s, 0)
    N_full = np.trapezoid(f_s, x=energies)
    f_s /= N_full

    f_s_above = np.where(energies >= split, f_s, 0)
    N_above = np.trapezoid(f_s_above, x=energies)
    f_s_above[energies >= split] = N_above / np.trapezoid(
        np.ones_like(energies[energies > split]), x=energies[energies > split]
    )
    f_s_below = np.where(energies < split, f_s, 0)
    return f_s_above + f_s_below
