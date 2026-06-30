"""Configuration loading, dataclasses, and preprocessing orchestration.

This module defines the core data structures used throughout the pipeline
and provides functions to load experiment configuration from YAML files and
derived data products (efficiency, background PDFs, resolution model).
"""

import os
from dataclasses import dataclass

import numpy as np
import yaml

from qualiphide import DATA_DIR
from qualiphide.preprocessing.efficiency import interpolate_efficiency
from qualiphide.preprocessing.background import process_background_spectra
from qualiphide.preprocessing.rates import fit_background_rates


@dataclass(frozen=True)
class PhysicsConstants:
    """Fundamental experiment constants derived from the YAML config.

    Parameters
    ----------
    N_KID : int
        Number of kinetic inductance detectors.
    runtime : float
        Total observation time in seconds.
    chi_N_m_factor : float
        Prefactor relating kinetic mixing chi to expected photon count:
        ``N_gamma_det = runtime * (chi / chi_N_m_factor)**2 / m``.
    sigma_factor_eta : float
        Fractional uncertainty on detector efficiency eta (1-sigma).
    """

    N_KID: int
    runtime: float
    chi_N_m_factor: float
    sigma_factor_eta: float


@dataclass(frozen=True)
class EnergyCutoffs:
    """Energy boundaries that define the analysis window.

    Parameters
    ----------
    cutoff : float
        Lower energy cutoff (eV).
    split : float
        Energy above which the signal PDF is flattened (eV).
    maximum : float
        Upper energy cutoff (eV).
    split_mu_b : float
        Energy separating "below" and "above" background regions (eV).
    """

    cutoff: float
    split: float
    maximum: float
    split_mu_b: float


@dataclass
class InterpolationData:
    """Interpolated data products for a specific mass point.

    Parameters
    ----------
    energies : np.ndarray
        Common energy grid (eV).
    eta_used : float
        Detector efficiency at the test mass.
    eta_a : np.ndarray
        Aperture efficiency curve over the energy grid.
    f_b_below : np.ndarray
        Background PDF below ``split_mu_b``.
    f_b_above : np.ndarray
        Background PDF above ``split_mu_b``.
    mu_b_data_below : list
        ``[x, pdf]`` arrays for the expected background count distribution below split.
    mu_b_data_above : list
        ``[x, pdf]`` arrays for the expected background count distribution above split.
    """

    energies: np.ndarray
    eta_used: float
    eta_a: np.ndarray
    f_b_below: np.ndarray
    f_b_above: np.ndarray
    mu_b_data_below: list
    mu_b_data_above: list


@dataclass
class ResolutionData:
    """Energy resolution model at a specific mass point.

    Parameters
    ----------
    median : float
        Median energy resolution (eV).
    delta_low : float
        Downward 1-sigma deviation from median (positive value).
    delta_up : float
        Upward 1-sigma deviation from median (positive value).
    """

    median: float
    delta_low: float
    delta_up: float


@dataclass(frozen=True)
class FitBounds:
    """Number of ancillary-likelihood sigmas used as Minuit parameter bounds.

    Parameters
    ----------
    n_sigma_eta : float
        Allowed range for eta is eta_used ± n_sigma_eta * sigma_eta, clipped to [0, 1].
    n_sigma_r : float
        Allowed range for r (standard-normal units) is [-n_sigma_r, +n_sigma_r].
    n_sigma_mu_b : float
        Allowed range for mu_b is mean ± n_sigma_mu_b * std of the ancillary PDF,
        clipped to be positive.
    """

    n_sigma_eta: float = 3.0
    n_sigma_r: float = 3.0
    n_sigma_mu_b: float = 5.0


@dataclass
class RunConfig:
    """Top-level configuration for a pipeline run.

    Parameters
    ----------
    name : str
        Config name (YAML stem, without ``.yaml``).
    output_dir : str
        Directory for pipeline outputs and preprocessed data.
    n_toy : int
        Number of Toy Monte Carlo pseudo-experiments per point.
    chi_true : np.ndarray
        Grid of kinetic mixing parameter values (includes 0).
    m_test : np.ndarray
        Grid of hidden photon masses to test (eV).
    physics : PhysicsConstants
        Experiment constants.
    cutoffs : EnergyCutoffs
        Energy boundaries.
    fit_bounds : FitBounds
        Sigma multipliers for Minuit parameter bounds.
    """

    name: str
    output_dir: str
    n_toy: int
    chi_true: np.ndarray
    m_test: np.ndarray
    physics: PhysicsConstants
    cutoffs: EnergyCutoffs
    fit_bounds: FitBounds = None
    chi_true_coverage: np.ndarray = None
    m_test_coverage: np.ndarray = None

    def __post_init__(self):
        if self.fit_bounds is None:
            object.__setattr__(self, 'fit_bounds', FitBounds())


def _parse_grid_dict(spec, decimals):
    """Parse the dict form of a grid spec into a sorted, deduplicated array.

    Accepts an optional ``logspace`` mapping (``start``, ``stop``, ``num``)
    and/or an ``extra`` list of explicit values; results are unioned.

    Parameters
    ----------
    spec : dict
        Mapping with ``logspace`` and/or ``extra`` keys.
    decimals : int
        Rounding precision applied before deduplication.
    """
    parts = []
    if "logspace" in spec:
        ls = spec["logspace"]
        parts.append(np.logspace(
            np.log10(np.float64(ls["start"])),
            np.log10(np.float64(ls["stop"])),
            num=int(ls["num"]),
        ))
    if "extra" in spec:
        parts.append(np.array([np.float64(v) for v in spec["extra"]]))
    if not parts:
        raise ValueError(
            "Grid dict form requires `logspace` and/or `extra`."
        )
    return np.unique(np.round(np.concatenate(parts), decimals=decimals))


def load_config(name: str) -> RunConfig:
    """Load experiment configuration from a YAML file and run preprocessing.

    Searches for ``configs/<name>.yaml`` first, then ``<name>.yaml`` in the
    current directory. Runs efficiency interpolation, background spectra
    processing, and background rate fitting as side effects.

    The ``chi_true`` and ``m_test`` fields accept either the legacy
    3-element ``[start, stop, num]`` logspace triple, or a dict form
    combining an optional ``logspace: {start, stop, num}`` and/or an
    ``extra: [...]`` list of explicit values (unioned, sorted, dedup'd).

    Parameters
    ----------
    name : str
        Config name (without ``.yaml`` extension).

    Returns
    -------
    RunConfig
        Fully populated configuration object.
    """
    yaml_path = os.path.join("configs", name + ".yaml")
    if not os.path.exists(yaml_path):
        yaml_path = name + ".yaml"

    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    output_dir = data.get("output_dir", name)
    os.makedirs(output_dir, exist_ok=True)

    N_KID = data["constant_params"]["N_KID"]
    A_dish = data["constant_params"]["A_dish"]
    rho_CDM = data["constant_params"]["rho_CDM"]
    alpha = np.sqrt(data["constant_params"]["alpha_sq"])
    runtime = data["constant_params"]["runtime"] * 60 * 60
    chi_N_m_factor = data["constant_params"]["chi_N_m_coeff"] * np.sqrt(
        0.3 / rho_CDM
    ) * np.sqrt(1 / A_dish) * (np.sqrt(2 / 3) / alpha)
    sigma_factor_eta = data["nuisance_param_data"]["sigma_factor_eta"]

    raw_chi = data["chi_true"]
    if isinstance(raw_chi, dict):
        chi_true = _parse_grid_dict(raw_chi, decimals=16)
    else:
        chi_true = np.round(
            np.logspace(
                np.log10(np.float64(raw_chi[0])),
                np.log10(np.float64(raw_chi[1])),
                num=int(raw_chi[2]),
            ),
            decimals=16,
        )
    if not np.any(chi_true == 0.0):
        chi_true = np.array([np.float64(0.0)] + list(chi_true))

    # Masses are rounded to the nearest 0.1 meV (1e-4 eV, decimals=4) and
    # deduplicated.  This matches the filename resolution in ``format_mass``
    # so grid points that round to the same 0.1 meV cannot produce colliding
    # filenames (which previously overwrote results and triggered a downstream
    # reshape/indexing error).
    raw_m = data["m_test"]
    if isinstance(raw_m, dict):
        m_test = _parse_grid_dict(raw_m, decimals=4)
    elif raw_m[-1] >= np.float64(1.0):
        m_test = np.unique(np.round(
            np.logspace(
                np.log10(np.float64(raw_m[0])),
                np.log10(np.float64(raw_m[1])),
                num=int(raw_m[2]),
            ),
            decimals=4,
        ))
    else:
        m_test = np.unique(
            np.round(np.array([np.float64(d) for d in raw_m]), decimals=4)
        )

    cutoffs = EnergyCutoffs(
        cutoff=data["cutoff"],
        split=data["split"],
        maximum=data["maximum"],
        split_mu_b=data["split_mu_b"],
    )

    # Skip preprocessing when output files already exist.  The derived
    # data depend only on the config parameters (cutoffs, runtime) and the
    # bundled CSV data, so they are identical across runs with the same
    # config.  Skipping avoids rewriting files that may be read by a
    # concurrent pipeline (e.g. coverage + sensitivity on the same dir).
    _efficiency_files = [f"{output_dir}/{f}" for f in
                         ("eta_full.csv", "eta_a.csv", "eta_rest.csv")]
    if not all(os.path.exists(f) for f in _efficiency_files):
        interpolate_efficiency(output_dir, cutoffs.cutoff, cutoffs.maximum)

    _background_files = [f"{output_dir}/{f}" for f in
                         ("f_b_below.csv", "f_b_above.csv")]
    if not all(os.path.exists(f) for f in _background_files):
        process_background_spectra(output_dir, cutoffs.cutoff, cutoffs.split_mu_b, cutoffs.split, cutoffs.maximum)

    _rates_files = [f"{output_dir}/{f}" for f in
                    ("mu_b_sampler_below.csv", "mu_b_sampler_above.csv")]
    if not all(os.path.exists(f) for f in _rates_files):
        fit_background_rates(output_dir, cutoffs.cutoff, cutoffs.split_mu_b, cutoffs.maximum, runtime)

    n_toy = data["energy_data"]["n_toy"]

    # Update global worker count from YAML (env var still overrides)
    import qualiphide
    max_w = data.get("max_workers", qualiphide._DEFAULT_MAX_WORKERS)
    if "QUALIPHIDE_WORKERS" not in os.environ:
        qualiphide.N_WORKERS = min(os.cpu_count() or 4, int(max_w))

    fb_data = data.get("fit_bounds", {})
    fit_bounds = FitBounds(
        n_sigma_eta=fb_data.get("n_sigma_eta", 3.0),
        n_sigma_r=fb_data.get("n_sigma_r", 3.0),
        n_sigma_mu_b=fb_data.get("n_sigma_mu_b", 5.0),
    )

    physics = PhysicsConstants(
        N_KID=N_KID,
        runtime=runtime,
        chi_N_m_factor=chi_N_m_factor,
        sigma_factor_eta=sigma_factor_eta,
    )

    chi_true_coverage = None
    raw_coverage = data.get("chi_true_coverage")
    if raw_coverage:
        chi_true_coverage = np.array([np.float64(v) for v in raw_coverage])

    m_test_coverage = None
    raw_m_coverage = data.get("m_test_coverage")
    if raw_m_coverage:
        m_test_coverage = np.array([np.float64(v) for v in raw_m_coverage])

    return RunConfig(
        name=name,
        output_dir=output_dir,
        n_toy=n_toy,
        chi_true=chi_true,
        m_test=m_test,
        physics=physics,
        cutoffs=cutoffs,
        fit_bounds=fit_bounds,
        chi_true_coverage=chi_true_coverage,
        m_test_coverage=m_test_coverage,
    )


def load_derived_data(name, m):
    """Load preprocessed data products and resolution model for a given mass.

    Parameters
    ----------
    name : str
        Config / output directory name.
    m : float
        Hidden photon mass (eV) to interpolate data for.

    Returns
    -------
    interp_data : InterpolationData
        Interpolated efficiency, background PDFs, and background rate distributions.
    res_data : ResolutionData
        Energy resolution model at mass *m*.
    """
    eta_data = np.loadtxt(f"{name}/eta_full.csv", delimiter=",", skiprows=1)
    energies, eta_func = eta_data[:, 0], eta_data[:, 1]
    eta_a_data = np.loadtxt(f"{name}/eta_a.csv", delimiter=",", skiprows=1)
    eta_a = eta_a_data[:, 1]
    f_b_below_data = np.loadtxt(f"{name}/f_b_below.csv", delimiter=",", skiprows=1)
    f_b_below = f_b_below_data[:, 2]
    f_b_above_data = np.loadtxt(f"{name}/f_b_above.csv", delimiter=",", skiprows=1)
    f_b_above = f_b_above_data[:, 2]
    mu_b_below_data = np.loadtxt(f"{name}/mu_b_sampler_below.csv", delimiter=",", skiprows=1)
    mu_b_data_below = [mu_b_below_data[:, 0], mu_b_below_data[:, 1]]
    mu_b_above_data = np.loadtxt(f"{name}/mu_b_sampler_above.csv", delimiter=",", skiprows=1)
    mu_b_data_above = [mu_b_above_data[:, 0], mu_b_above_data[:, 1]]

    resolution_csv = np.loadtxt(
        str(DATA_DIR / "resolution_model_20260211.csv"), delimiter=",", skiprows=1
    )
    E_res = resolution_csv[:, 0] / 1000
    resolution_median = resolution_csv[:, 1] / 1000
    resolution_lower = resolution_csv[:, 2] / 1000
    resolution_upper = resolution_csv[:, 3] / 1000

    eta_used = np.interp(m, energies, eta_func)
    res_med = np.interp(m, E_res, resolution_median)
    res_low = np.interp(m, E_res, resolution_lower)
    res_up = np.interp(m, E_res, resolution_upper)

    interp_data = InterpolationData(
        energies=energies,
        eta_used=eta_used,
        eta_a=eta_a,
        f_b_below=f_b_below,
        f_b_above=f_b_above,
        mu_b_data_below=mu_b_data_below,
        mu_b_data_above=mu_b_data_above,
    )

    res_data = ResolutionData(
        median=res_med,
        delta_low=res_med - res_low,
        delta_up=res_up - res_med,
    )

    return interp_data, res_data
