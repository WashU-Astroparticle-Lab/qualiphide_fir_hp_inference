"""Toy Monte Carlo pseudo-experiment generation.

Each toy samples nuisance parameters (eta, mu_b, r), generates signal
events from the signal PDF, and background events from randomly selected
channel combinations, then combines them into a single observed energy array.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from typing import NamedTuple

import qualiphide
from qualiphide import DATA_DIR
from qualiphide.signal import generate_signal_pdf

# Module-level dict populated by the pool initializer in each worker.
_T = {}


class ToyData(NamedTuple):
    """Data for a single pseudo-experiment.

    Parameters
    ----------
    observed_energies : np.ndarray
        Sorted array of observed event energies (eV).
    eta : float
        Sampled detector efficiency.
    mu_b_below : float
        Sampled expected background count below split.
    mu_b_above : float
        Sampled expected background count above split.
    r : float
        Sampled resolution nuisance parameter (standard-normal units).
    n_signal : int
        Poisson-sampled number of signal events.
    n_bkg_below : int
        Poisson-sampled number of background events below split.
    n_bkg_above : int
        Poisson-sampled number of background events above split.
    bkg_channel_indices : tuple
        Indices of the 4 background channels selected for this toy.
    """

    observed_energies: np.ndarray
    eta: float
    mu_b_below: float
    mu_b_above: float
    r: float
    n_signal: int
    n_bkg_below: int
    n_bkg_above: int
    bkg_channel_indices: tuple


def load_channel_data():
    """Load the raw background channel CSV once.

    Returns
    -------
    np.ndarray
        Raw 2-D array from ``sr3pt2_bkg_channels_Hz_meV_remade.csv``.
    """
    channel_path = DATA_DIR / "sr3pt2_bkg_channels_Hz_meV_remade.csv"
    return np.loadtxt(str(channel_path), delimiter=",", skiprows=1)


def _process_channels(channel_data, energies):
    """Interpolate raw channel spectra onto the analysis energy grid.

    Parameters
    ----------
    channel_data : np.ndarray
        Raw CSV data (index, energies_meV, ch0, ch1, ...).
    energies : np.ndarray
        Analysis energy grid (eV).

    Returns
    -------
    np.ndarray
        Shape ``(n_channels, len(energies))`` normalised channel PDFs.
    """
    channel_energies = channel_data[:, 1] * 1e-3  # meV -> eV
    raw_channels = channel_data[:, 2:].T

    channels = []
    for ch in raw_channels:
        interp = np.interp(energies, channel_energies, ch, left=0, right=0)
        norm = np.trapezoid(interp, x=energies)
        if norm > 1e-12:
            channels.append(interp / norm)

    return np.array(channels)


def _init_toymc_worker(channels, energies, eta_a, mass, cutoff, split,
                       maximum, split_mu_b):
    """Initialise a ToyMC worker with shared read-only data."""
    _T["channels"] = channels
    _T["energies"] = energies
    _T["eta_a"] = eta_a
    _T["mass"] = mass
    _T["cutoff"] = cutoff
    _T["split"] = split
    _T["maximum"] = maximum
    _T["split_mu_b"] = split_mu_b


def build_background_pdfs(channel_indices, channels, energies, cutoff, split,
                          maximum, split_mu_b):
    """Build below/above background PDFs from a set of channel indices.

    Averages the selected channel spectra, applies energy cutoffs, flattens
    above ``split``, and splits into below/above ``split_mu_b`` normalised PDFs.

    Parameters
    ----------
    channel_indices : array_like
        Indices into *channels* selecting which channels to average.
    channels : np.ndarray
        Shape ``(n_channels, len(energies))`` normalised channel PDFs
        (as returned by :func:`_process_channels`).
    energies : np.ndarray
        Analysis energy grid (eV).
    cutoff : float
        Lower energy cutoff (eV).
    split : float
        Energy above which the PDF is flattened (eV).
    maximum : float
        Upper energy cutoff (eV).
    split_mu_b : float
        Energy separating below/above background regions (eV).

    Returns
    -------
    energies : np.ndarray
        The energy grid the PDFs are defined on (eV).
    f_b_below : np.ndarray
        Normalised background PDF below ``split_mu_b``.
    f_b_above : np.ndarray
        Normalised background PDF above ``split_mu_b``.
    """
    f_b = np.mean(channels[list(channel_indices)], axis=0)

    # Apply cutoff
    mask = (energies >= cutoff) & (energies <= maximum)
    f_b = np.where(mask, f_b, 0.0)

    # Renormalize
    norm = np.trapezoid(f_b, x=energies)
    if norm > 1e-12:
        f_b /= norm

    # Flatten above split
    mask_flat = energies >= split
    f_flat = np.where(mask_flat, f_b, 0)
    f_not_flat = np.where(~mask_flat, f_b, 0)

    N_flat = np.trapezoid(f_flat, x=energies)
    weights = np.gradient(energies)
    flat_value = N_flat / np.sum(weights[mask_flat])

    f_flat = np.where(mask_flat, flat_value, 0)
    f_b = f_flat + f_not_flat

    # Split into below / above
    mask_below = energies < split_mu_b
    mask_above = ~mask_below

    f_b_below = np.where(mask_below, f_b, 0.0)
    f_b_above = np.where(mask_above, f_b, 0.0)

    # Renormalize each
    norm_below = np.trapezoid(f_b_below, x=energies)
    norm_above = np.trapezoid(f_b_above, x=energies)

    if norm_below > 1e-12:
        f_b_below /= norm_below
    if norm_above > 1e-12:
        f_b_above /= norm_above

    return energies, f_b_below, f_b_above


def _generate_single_toy(args):
    """Generate one pseudo-experiment (called by multiprocessing workers)."""
    (eta_i, mu_b_below_i, mu_b_above_i,
     N_s, N_b_below, N_b_above,
     r_i, resolution, i) = args

    energies = _T["energies"]
    channels = _T["channels"]
    mass = _T["mass"]
    cutoff = _T["cutoff"]
    split = _T["split"]
    maximum = _T["maximum"]
    eta_a = _T["eta_a"]
    split_mu_b = _T["split_mu_b"]

    # --- SIGNAL ---
    f_s = generate_signal_pdf(mass, cutoff, split, maximum, resolution, energies, eta_a)
    cdf_s = cumulative_trapezoid(f_s, x=energies, initial=0)
    E_i_s = np.interp(np.random.random(size=N_s), cdf_s, energies)

    # --- BACKGROUND: mean of 4 random channels ---
    idx = np.random.choice(len(channels), size=4, replace=False)
    _, f_b_below, f_b_above = build_background_pdfs(
        idx, channels, energies, cutoff, split, maximum, split_mu_b
    )

    # --- Build CDFs ---
    cdf_b_below = cumulative_trapezoid(f_b_below, x=energies, initial=0)
    cdf_b_above = cumulative_trapezoid(f_b_above, x=energies, initial=0)

    # --- Sample ---
    E_i_b_below = np.interp(
        np.random.random(size=N_b_below), cdf_b_below, energies
    )
    E_i_b_above = np.interp(
        np.random.random(size=N_b_above), cdf_b_above, energies
    )

    # --- Combine ---
    E_i = np.concatenate((E_i_s, E_i_b_below, E_i_b_above))
    E_i = np.sort(E_i)

    return (i, ToyData(E_i, eta_i, mu_b_below_i, mu_b_above_i, r_i,
                       N_s, N_b_below, N_b_above, tuple(idx)))


def generate_toymc(n_toys, c, m, physics, interp_data, res_data, cutoffs,
                   channel_data=None):
    """Generate *n_toys* pseudo-experiments for a given (chi, mass) point.

    Parameters
    ----------
    n_toys : int
        Number of pseudo-experiments to generate.
    c : float
        Kinetic mixing parameter chi.
    m : float
        Hidden photon mass (eV).
    physics : PhysicsConstants
        Experiment constants.
    interp_data : InterpolationData
        Interpolated efficiency, background PDFs, and rate distributions.
    res_data : ResolutionData
        Energy resolution model.
    cutoffs : EnergyCutoffs
        Energy boundaries.
    channel_data : np.ndarray, optional
        Pre-loaded raw channel CSV data (from :func:`load_channel_data`).
        If *None*, the CSV is loaded from disk (backward-compatible).

    Returns
    -------
    list[ToyData]
        List of pseudo-experiments ordered by index.
    """
    energies = interp_data.energies
    eta_used = interp_data.eta_used
    eta_a = interp_data.eta_a
    mu_b_data_below = interp_data.mu_b_data_below
    mu_b_data_above = interp_data.mu_b_data_above

    # --- LOAD / PROCESS CHANNELS ---
    if channel_data is None:
        channel_data = load_channel_data()
    channels = _process_channels(channel_data, energies)

    # --- Background rate distributions ---
    cdf_mu_b_below = cumulative_trapezoid(
        mu_b_data_below[1], x=mu_b_data_below[0], initial=0
    )
    cdf_mu_b_above = cumulative_trapezoid(
        mu_b_data_above[1], x=mu_b_data_above[0], initial=0
    )

    # --- Expected counts ---
    N_gamma_det = physics.runtime * (c / physics.chi_N_m_factor) ** 2 * (1 / m)

    eta = np.maximum(
        0,
        np.minimum(
            1,
            np.random.normal(
                eta_used, physics.sigma_factor_eta * eta_used, n_toys
            ),
        ),
    )

    mu_b_below = np.interp(
        np.random.random(size=n_toys), cdf_mu_b_below, mu_b_data_below[0]
    )
    mu_b_above = np.interp(
        np.random.random(size=n_toys), cdf_mu_b_above, mu_b_data_above[0]
    )

    r = np.random.normal(0, 1, n_toys)
    resolution = np.maximum(
        1e-6,
        np.where(
            r < 0,
            res_data.median + r * res_data.delta_low,
            res_data.median + r * res_data.delta_up,
        ),
    )

    N_s = np.random.poisson(eta * N_gamma_det / physics.N_KID)
    N_b_below = np.random.poisson(mu_b_below)
    N_b_above = np.random.poisson(mu_b_above)

    # --- Pack lightweight per-toy arguments ---
    args = [
        (
            eta[i],
            mu_b_below[i],
            mu_b_above[i],
            N_s[i],
            N_b_below[i],
            N_b_above[i],
            r[i],
            resolution[i],
            i,
        )
        for i in range(n_toys)
    ]

    # --- Run multiprocessing with initializer pattern ---
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=qualiphide.N_WORKERS,
        initializer=_init_toymc_worker,
        initargs=(channels, energies, eta_a, m,
                  cutoffs.cutoff, cutoffs.split, cutoffs.maximum,
                  cutoffs.split_mu_b),
    ) as pool:
        toys = pool.map(_generate_single_toy, args)

    # Sort by original index
    toys.sort(key=lambda x: x[0])
    toy_array = [t[1] for t in toys]

    return toy_array


def toydata_from_energies(energies):
    """Wrap an observed energy array into a ToyData list for compute_q_values.

    Use this for real (non-simulated) data where true nuisance parameters
    are unknown.  The nuisance fields are set to NaN and event counts to -1.

    Parameters
    ----------
    energies : array_like
        Observed event energies in eV.

    Returns
    -------
    list[ToyData]
        Single-element list ready to pass to :func:`compute_q_values`.
    """
    E = np.sort(np.asarray(energies, dtype=np.float64))
    return [ToyData(E, np.nan, np.nan, np.nan, np.nan, -1, -1, -1, ())]


def load_toymc(file):
    """Load previously saved ToyMC data from a ``.npy`` file.

    Old files may contain 5-element tuples (without event counts).
    These are converted to ``ToyData`` with event counts set to -1.

    Parameters
    ----------
    file : str
        Path to the ``.npy`` file.

    Returns
    -------
    list[ToyData]
        List of toy experiment tuples.
    """
    toy_data = np.load(file, allow_pickle=True)
    toys = toy_data.tolist()
    # Migrate old tuples to the current 9-field ToyData
    if toys and len(toys[0]) < 9:
        n = len(toys[0])
        def _migrate(t):
            pad = {5: (-1, -1, -1, ()), 8: ((),)}
            return ToyData(*t, *pad[n])
        toys = [_migrate(t) for t in toys]
    return toys
