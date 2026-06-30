"""Profile likelihood ratio test statistic computation via iminuit.

Computes the test statistic ``q = 2 * (NLL_constrained - NLL_unconstrained)``
using batched multiprocessing workers.  A global dict ``G`` carries
pre-computed arrays and the reusable Minuit instance into each worker.
"""

import os
import numpy as np
from math import lgamma
from iminuit import Minuit
from qualiphide import format_mass
import qualiphide.likelihood as ll
import multiprocessing as mp
import gc
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import qualiphide

CHI_DECIMALS = 15
_DEFAULT_BATCH_SIZE = 40


def _batch_size(n_toys):
    """Choose batch size to fill all cores, capped at _DEFAULT_BATCH_SIZE."""
    n_workers = qualiphide.N_WORKERS
    bs = max(1, n_toys // n_workers)
    return min(bs, _DEFAULT_BATCH_SIZE)


def round_chi(c):
    """Round a chi value to ``CHI_DECIMALS`` significant digits."""
    return round(float(c), CHI_DECIMALS)


G = {}


class NLLUnconstrained:
    """Negative log-likelihood with all parameters free (denominator of PLR).

    Parameters
    ----------
    arguments : tuple
        Per-toy likelihood arguments passed to :func:`ll.log_likelihood`.
    """

    def __init__(self, arguments):
        self.arguments = arguments

    def __call__(self, eta, mu_b_below, mu_b_above, r, N_gamma_det):
        val = ll.log_likelihood(
            eta, mu_b_below, mu_b_above, r, N_gamma_det, self.arguments
        )
        return -val if np.isfinite(val) else np.inf


class NLLConstrained:
    """Negative log-likelihood with N_gamma_det fixed (numerator of PLR).

    Parameters
    ----------
    N_gamma_det : float
        Fixed expected detected signal photon count.
    arguments : tuple
        Per-toy likelihood arguments passed to :func:`ll.log_likelihood`.
    """

    def __init__(self, N_gamma_det, arguments):
        self.N_gamma_det = N_gamma_det
        self.arguments = arguments

    def __call__(self, eta, mu_b_below, mu_b_above, r):
        val = ll.log_likelihood(
            eta, mu_b_below, mu_b_above, r, self.N_gamma_det, self.arguments
        )
        return -val if np.isfinite(val) else np.inf


def _pdf_mean_std(x, pdf):
    """Compute mean and std of a distribution given as (x, pdf) arrays."""
    mean = np.trapezoid(x * pdf, x=x)
    var = np.trapezoid((x - mean) ** 2 * pdf, x=x)
    return mean, np.sqrt(max(var, 0.0))


def _populate_G(physics, interp_data, toy_data, res_data, cutoffs, fit_bounds):
    """Fill the global dict *G* with shared data used by all workers."""
    from qualiphide.config import FitBounds
    if fit_bounds is None:
        fit_bounds = FitBounds()
    G["physics"] = physics
    G["eta_used"] = interp_data.eta_used
    G["f_b_funcs"] = (interp_data.f_b_below, interp_data.f_b_above)
    G["toy_data"] = toy_data
    G["res_data"] = res_data

    G["nll"] = None

    E = interp_data.energies
    eta_a = interp_data.eta_a

    MASK_FULL = (E >= cutoffs.cutoff) & (E <= cutoffs.maximum)
    MASK_ABOVE = E >= cutoffs.split
    dE = np.diff(E)
    E_above = E[MASK_ABOVE]
    WIDTH_ABOVE = np.sum(E_above[1:] - E_above[:-1])

    G["E"] = E
    G["eta_a"] = eta_a
    G["MASK_FULL"] = MASK_FULL
    G["MASK_ABOVE"] = MASK_ABOVE
    G["dE"] = dE
    G["WIDTH_ABOVE"] = WIDTH_ABOVE
    G["mu_b_data_below"] = interp_data.mu_b_data_below
    G["mu_b_data_above"] = interp_data.mu_b_data_above

    # Precompute contiguous index ranges for numba kernels
    full_indices = np.where(MASK_FULL)[0]
    above_indices = np.where(MASK_ABOVE)[0]
    G["mf_start"] = int(full_indices[0])
    G["mf_end"] = int(full_indices[-1]) + 1
    G["ma_start"] = int(above_indices[0])
    G["ma_end"] = int(above_indices[-1]) + 1
    G["split_energy"] = cutoffs.split

    # Precompute physically motivated parameter bounds
    eta_used = interp_data.eta_used
    sigma_eta = physics.sigma_factor_eta * eta_used
    G["eta_bounds"] = (
        max(0.0, eta_used - fit_bounds.n_sigma_eta * sigma_eta),
        min(1.0, eta_used + fit_bounds.n_sigma_eta * sigma_eta),
    )
    G["r_bounds"] = (-fit_bounds.n_sigma_r, fit_bounds.n_sigma_r)

    mu_b_below_mean, mu_b_below_std = _pdf_mean_std(
        interp_data.mu_b_data_below[0], interp_data.mu_b_data_below[1]
    )
    mu_b_above_mean, mu_b_above_std = _pdf_mean_std(
        interp_data.mu_b_data_above[0], interp_data.mu_b_data_above[1]
    )
    G["mu_b_below_bounds"] = (
        max(0.0, mu_b_below_mean - fit_bounds.n_sigma_mu_b * mu_b_below_std),
        mu_b_below_mean + fit_bounds.n_sigma_mu_b * mu_b_below_std,
    )
    G["mu_b_above_bounds"] = (
        max(0.0, mu_b_above_mean - fit_bounds.n_sigma_mu_b * mu_b_above_std),
        mu_b_above_mean + fit_bounds.n_sigma_mu_b * mu_b_above_std,
    )

    # Precompute central initial guesses (independent of per-toy true values)
    G["eta_init"] = eta_used
    G["mu_b_below_init"] = mu_b_below_mean
    G["mu_b_above_init"] = mu_b_above_mean
    G["r_init"] = 0.0

    # Warmup numba JIT (first call compiles)
    _warmup_numba(E, eta_a)


def _warmup_numba(E, eta_a):
    """Trigger numba compilation during worker init so it doesn't happen during migrad."""
    try:
        ll._compute_normalization_windowed(E, eta_a, E[len(E) // 2], 0.001,
                                           G["mf_start"], G["mf_end"],
                                           G["ma_start"], G["ma_end"])
    except Exception:
        pass


def _init_worker_unconstrained(physics, interp_data, toy_data, res_data, cutoffs, fit_bounds):
    """Initialise a multiprocessing worker for unconstrained minimisation."""
    _populate_G(physics, interp_data, toy_data, res_data, cutoffs, fit_bounds)

    def nll_wrapper(eta, mu_b_below, mu_b_above, r, N_gamma_det):
        return G["nll"](eta, mu_b_below, mu_b_above, r, N_gamma_det)

    m = Minuit(
        nll_wrapper,
        eta=0.0,
        mu_b_below=1.0,
        mu_b_above=1.0,
        r=0.0,
        N_gamma_det=1.0,
    )
    m.strategy = 0
    m.print_level = 0
    m.throw_nan = False
    m.errordef = 0.5

    G["minuit"] = m


_CONSTRAINED_PARAMS = ("eta", "mu_b_below", "mu_b_above", "r")


def _init_worker_constrained(physics, interp_data, toy_data, res_data, cutoffs,
                              fit_bounds, fixed_params):
    """Initialise a multiprocessing worker for constrained minimisation."""
    _populate_G(physics, interp_data, toy_data, res_data, cutoffs, fit_bounds)
    G["fixed_params"] = set(fixed_params) if fixed_params else set()

    def nll_wrapper(eta, mu_b_below, mu_b_above, r):
        return G["nll"](eta, mu_b_below, mu_b_above, r)

    m = Minuit(
        nll_wrapper,
        eta=0.0,
        mu_b_below=1.0,
        mu_b_above=1.0,
        r=0.0,
    )
    m.strategy = 0
    m.print_level = 0
    m.throw_nan = False
    m.errordef = 0.5

    G["minuit"] = m


def minimize_unconstrained(toy_data, physics, m, chi_start,
                           interp_data, res_data, cutoffs, fit_bounds=None):
    """Run unconstrained (denominator) minimisation across all toys.

    Parameters
    ----------
    toy_data : list
        List of toy experiment tuples.
    physics : PhysicsConstants
        Experiment constants.
    m : float
        Hidden photon mass (eV).
    chi_start : float
        Starting chi for N_gamma_det initialisation.
    interp_data : InterpolationData
        Interpolated data products.
    res_data : ResolutionData
        Resolution model.
    cutoffs : EnergyCutoffs
        Energy boundaries.

    Returns
    -------
    nll_max : np.ndarray
        Best-fit NLL for each toy.
    chi_hats : np.ndarray
        Best-fit chi for each toy.
    theta_hats : np.ndarray
        Best-fit nuisance parameters for each toy.
    """
    bs = _batch_size(len(toy_data))
    worker_args = [
        (start, min(start + bs, len(toy_data)), m, chi_start)
        for start in range(0, len(toy_data), bs)
    ]

    n = len(toy_data)
    nll_max = [None] * n
    chi_hats = [None] * n
    theta_hats = [None] * n

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=qualiphide.N_WORKERS,
        initializer=_init_worker_unconstrained,
        initargs=(physics, interp_data, toy_data, res_data, cutoffs, fit_bounds),
    ) as pool:
        pbar = tqdm(total=n, desc="    Unconstrained fits", unit="toy", leave=False)
        for start, results in pool.imap_unordered(_worker_unconstrained, worker_args):
            for j, (nll, chi, theta) in enumerate(results):
                i = start + j
                nll_max[i] = nll
                chi_hats[i] = chi
                theta_hats[i] = theta
            pbar.update(len(results))
        pbar.close()
        pool.close()
        pool.join()

    return (
        np.array(nll_max),
        np.array(chi_hats),
        np.array(theta_hats),
    )


def _worker_unconstrained(args):
    """Batch worker for unconstrained fits."""
    start, end, m_test, chi_start = args
    out = []
    for i in range(start, end):
        out.append(
            _fit_single_toy_unconstrained(
                G["toy_data"][i],
                G["physics"],
                m_test,
                chi_start,
                G["f_b_funcs"],
            )
        )
    return start, out


def _fit_single_toy_unconstrained(toy_data, physics, m_test, chi_start, f_b_funcs):
    """Minimise the unconstrained NLL for a single toy.

    Returns
    -------
    fval : float
        Minimum NLL value.
    chi_hat : float
        Best-fit chi.
    theta_hat : np.ndarray
        Best-fit nuisance parameters ``[eta, mu_b_below, mu_b_above, r]``.
    """
    (E_in, eta_i, mu_b_below_i, mu_b_above_i, r_i, *_) = toy_data
    f_b_below, f_b_above = f_b_funcs
    f_b_val_below = np.interp(E_in, G["E"], f_b_below, left=0.0, right=0.0)
    f_b_val_above = np.interp(E_in, G["E"], f_b_above, left=0.0, right=0.0)
    np.clip(f_b_val_below, 1e-300, None, out=f_b_val_below)
    np.clip(f_b_val_above, 1e-300, None, out=f_b_val_above)

    # Precompute per-toy arrays (constant across migrad iterations)
    eta_a_at_Ein = np.interp(E_in, G["E"], G["eta_a"], left=0.0, right=0.0)
    G["_eta_a_Ein_cache"] = eta_a_at_Ein
    G["_above_mask_Ein"] = E_in >= G["split_energy"]
    G["_f_s_buf"] = np.empty_like(E_in)
    G["_lgamma_cache"] = lgamma(len(E_in) + 1.0)

    N_gamma_det_start = (
        physics.runtime * (chi_start / physics.chi_N_m_factor) ** 2 / m_test
    )
    arguments = (
        E_in,
        physics.N_KID,
        G["eta_used"],
        physics.sigma_factor_eta * G["eta_used"],
        m_test,
        f_b_val_below,
        f_b_val_above,
        G,
    )

    G["nll"] = NLLUnconstrained(arguments)
    m = G["minuit"]

    m.values["eta"] = G["eta_init"]
    m.values["mu_b_below"] = G["mu_b_below_init"]
    m.values["mu_b_above"] = G["mu_b_above_init"]
    m.values["r"] = G["r_init"]
    m.values["N_gamma_det"] = N_gamma_det_start

    m.limits = (
        G["eta_bounds"],
        G["mu_b_below_bounds"],
        G["mu_b_above_bounds"],
        G["r_bounds"],
        (0, None),
    )
    m.migrad()

    eta_hat = m.values["eta"]
    mu_b_below_hat = m.values["mu_b_below"]
    mu_b_above_hat = m.values["mu_b_above"]
    r_hat = m.values["r"]
    chi_hat = physics.chi_N_m_factor * np.sqrt(
        m.values["N_gamma_det"] * m_test / physics.runtime
    )

    theta_hat = np.array([eta_hat, mu_b_below_hat, mu_b_above_hat, r_hat])

    return m.fval, chi_hat, theta_hat


def minimize_constrained(toy_data, physics, m, c,
                         interp_data, res_data, cutoffs, pool=None,
                         fit_bounds=None, fixed_params=None):
    """Run constrained (numerator) minimisation across all toys.

    Parameters
    ----------
    toy_data : list
        List of toy experiment tuples.
    physics : PhysicsConstants
        Experiment constants.
    m : float
        Hidden photon mass (eV).
    c : float or np.ndarray
        Fixed chi value(s) for the constrained fit.  A scalar applies the
        same chi to every toy; an array of length ``len(toy_data)`` assigns
        a per-toy chi.
    interp_data : InterpolationData
        Interpolated data products.
    res_data : ResolutionData
        Resolution model.
    cutoffs : EnergyCutoffs
        Energy boundaries.
    pool : multiprocessing.Pool, optional
        Pre-created pool to reuse. If None, a new pool is created.
    fixed_params : set of str, optional
        Nuisance parameter names to fix at their true (ToyMC-drawn) values
        during the constrained fit.  Valid names: ``"eta"``,
        ``"mu_b_below"``, ``"mu_b_above"``, ``"r"``.  When a pool is
        provided, the pool must have been created with the same
        ``fixed_params``.

    Returns
    -------
    nll_out : np.ndarray
        Constrained NLL for each toy.
    theta_dbl_hats : np.ndarray
        Best-fit nuisance parameters under the constraint.
    """
    if fixed_params:
        bad = set(fixed_params) - set(_CONSTRAINED_PARAMS)
        if bad:
            raise ValueError(
                f"Unknown constrained parameter(s): {bad}. "
                f"Valid names: {_CONSTRAINED_PARAMS}"
            )
        if pool is not None:
            raise ValueError(
                "fixed_params cannot be passed with a pre-created pool. "
                "Pass fixed_params to create_constrained_pool() instead."
            )
    n = len(toy_data)
    c_arr = np.broadcast_to(np.asarray(c, dtype=np.float64), (n,)).copy()
    bs = _batch_size(n)
    worker_args = [
        (start, min(start + bs, n), m, c_arr[start:min(start + bs, n)])
        for start in range(0, n, bs)
    ]

    n = len(toy_data)
    nll_out = [None] * n
    theta_dbl_hats = [None] * n

    pbar = tqdm(total=n, desc="    Constrained fits", unit="toy", leave=False)
    if pool is not None:
        for start, results in pool.imap_unordered(_worker_constrained, worker_args):
            for j, (nll, theta) in enumerate(results):
                i = start + j
                nll_out[i] = nll
                theta_dbl_hats[i] = theta
            pbar.update(len(results))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=qualiphide.N_WORKERS,
            initializer=_init_worker_constrained,
            initargs=(physics, interp_data, toy_data, res_data, cutoffs,
                      fit_bounds, fixed_params),
        ) as pool_local:
            for start, results in pool_local.imap_unordered(_worker_constrained, worker_args):
                for j, (nll, theta) in enumerate(results):
                    i = start + j
                    nll_out[i] = nll
                    theta_dbl_hats[i] = theta
                pbar.update(len(results))
            pool_local.close()
            pool_local.join()
    pbar.close()
    return np.array(nll_out), np.array(theta_dbl_hats)


def _worker_constrained(args):
    """Batch worker for constrained fits."""
    start, end, m_test, chi_batch = args
    out = []
    for i in range(start, end):
        out.append(
            _fit_single_toy_constrained(
                G["toy_data"][i],
                G["physics"],
                m_test,
                float(chi_batch[i - start]),
                G["f_b_funcs"],
            )
        )
    return start, out


def _fit_single_toy_constrained(toy_data, physics, m_test, chi_test, f_b_funcs):
    """Minimise the constrained NLL for a single toy.

    Returns
    -------
    fval : float
        Minimum NLL value.
    theta_dbl_hat : np.ndarray
        Best-fit nuisance parameters ``[eta, mu_b_below, mu_b_above, r]``.
    """
    (E_in, eta_i, mu_b_below_i, mu_b_above_i, r_i, *_) = toy_data
    f_b_below, f_b_above = f_b_funcs
    f_b_val_below = np.interp(E_in, G["E"], f_b_below, left=0.0, right=0.0)
    f_b_val_above = np.interp(E_in, G["E"], f_b_above, left=0.0, right=0.0)
    np.clip(f_b_val_below, 1e-300, None, out=f_b_val_below)
    np.clip(f_b_val_above, 1e-300, None, out=f_b_val_above)

    # Precompute per-toy arrays (constant across migrad iterations)
    eta_a_at_Ein = np.interp(E_in, G["E"], G["eta_a"], left=0.0, right=0.0)
    G["_eta_a_Ein_cache"] = eta_a_at_Ein
    G["_above_mask_Ein"] = E_in >= G["split_energy"]
    G["_f_s_buf"] = np.empty_like(E_in)
    G["_lgamma_cache"] = lgamma(len(E_in) + 1.0)

    N_gamma_det_test = (
        physics.runtime * (chi_test / physics.chi_N_m_factor) ** 2 / m_test
    )
    arguments = (
        E_in,
        physics.N_KID,
        G["eta_used"],
        physics.sigma_factor_eta * G["eta_used"],
        m_test,
        f_b_val_below,
        f_b_val_above,
        G,
    )

    G["nll"] = NLLConstrained(N_gamma_det_test, arguments)
    m = G["minuit"]

    m.values["eta"] = G["eta_init"]
    m.values["mu_b_below"] = G["mu_b_below_init"]
    m.values["mu_b_above"] = G["mu_b_above_init"]
    m.values["r"] = G["r_init"]

    m.limits = (
        G["eta_bounds"],
        G["mu_b_below_bounds"],
        G["mu_b_above_bounds"],
        G["r_bounds"],
    )

    # Reset all to free, then fix requested parameters at true values
    fixed = G["fixed_params"]
    for p in _CONSTRAINED_PARAMS:
        m.fixed[p] = p in fixed

    m.migrad()

    return m.fval, np.array(
        [
            m.values["eta"],
            m.values["mu_b_below"],
            m.values["mu_b_above"],
            m.values["r"],
        ]
    )


def create_constrained_pool(physics, interp_data, toy_data, res_data, cutoffs,
                            fit_bounds=None, fixed_params=None):
    """Create a reusable multiprocessing pool for constrained minimisation.

    Parameters
    ----------
    fixed_params : set of str, optional
        Nuisance parameter names to fix at their true values.  Must match
        the ``fixed_params`` passed to :func:`minimize_constrained`.

    Returns
    -------
    pool : multiprocessing.Pool
        Initialised pool ready for constrained minimisation tasks.
    """
    ctx = mp.get_context("fork")
    return ctx.Pool(
        processes=qualiphide.N_WORKERS,
        initializer=_init_worker_constrained,
        initargs=(physics, interp_data, toy_data, res_data, cutoffs,
                  fit_bounds, fixed_params),
    )


def compute_q_values(toy_data, c, m, physics,
                     interp_data, res_data, cutoffs, fit_bounds=None,
                     fixed_params=None):
    """Run both minimisations and compute the PLR test statistic for every toy.

    Parameters
    ----------
    toy_data : list
        Toy experiment data.
    c : float
        Kinetic mixing parameter chi.
    m : float
        Hidden photon mass (eV).
    physics : PhysicsConstants
        Experiment constants.
    interp_data : InterpolationData
        Interpolated data products.
    res_data : ResolutionData
        Resolution model.
    cutoffs : EnergyCutoffs
        Energy boundaries.
    fixed_params : set of str, optional
        Nuisance parameter names to fix at their true (ToyMC-drawn) values
        during the constrained fit.  Valid names: ``"eta"``,
        ``"mu_b_below"``, ``"mu_b_above"``, ``"r"``.  Default (None) leaves
        all nuisance parameters free.

    Returns
    -------
    q_i : np.ndarray
        Test statistic for each toy.
    nll_max : np.ndarray
        Best-fit NLL values from unconstrained fit.
    chi_hat_i : np.ndarray
        Best-fit chi for each toy.
    theta_hat_i : np.ndarray
        Best-fit nuisance parameters from unconstrained fit.
    theta_dbl_hat_i : np.ndarray
        Best-fit nuisance parameters from constrained fit.
    """
    nll_max, chi_hat_i, theta_hat_i = minimize_unconstrained(
        toy_data, physics, m, max(c, 1e-14),
        interp_data, res_data, cutoffs, fit_bounds,
    )
    nll_out, theta_dbl_hat_i = minimize_constrained(
        toy_data, physics, m, c,
        interp_data, res_data, cutoffs, fit_bounds=fit_bounds,
        fixed_params=fixed_params,
    )
    q_i = 2 * (nll_out - nll_max)
    return q_i, nll_max, chi_hat_i, theta_hat_i, theta_dbl_hat_i


def save_fit_results(name, m, c, q_i, chi_hat_i, theta_hat_i,
                     theta_dbl_hat_i, toy_data, nll_max):
    """Save per-toy PLR fit results to CSV and nll_max to .npy.

    Parameters
    ----------
    name : str
        Output directory name.
    m : float
        Hidden photon mass (eV).
    c : float
        Kinetic mixing parameter chi.
    q_i : np.ndarray
        Test statistic for each toy.
    chi_hat_i : np.ndarray
        Best-fit chi for each toy.
    theta_hat_i : np.ndarray
        Best-fit nuisance parameters from unconstrained fit.
    theta_dbl_hat_i : np.ndarray
        Best-fit nuisance parameters from constrained fit.
    toy_data : list
        Toy experiment data (used for true nuisance parameter values).
    nll_max : np.ndarray
        Best-fit NLL values from unconstrained fit.
    """
    os.makedirs(f"{name}/fit_results", exist_ok=True)
    fit_path = f"{name}/fit_results/fit_results_{format_mass(m)}_{round_chi(c)}.csv"
    nll_max_path = f"{name}/fit_results/nll_max_{format_mass(m)}_{round_chi(c)}.npy"

    header = ("q,chi_hat,"
              "eta_hat,mu_b_below_hat,mu_b_above_hat,r_hat,"
              "eta_dbl_hat,mu_b_below_dbl_hat,mu_b_above_dbl_hat,r_dbl_hat,"
              "chi_true,eta_true,mu_b_below_true,mu_b_above_true,r_true,"
              "n_signal,n_bkg_below,n_bkg_above,"
              "bkg_ch0,bkg_ch1,bkg_ch2,bkg_ch3")
    lines = [header]
    for i in range(len(q_i)):
        parts = [str(q_i[i]), str(chi_hat_i[i])]
        parts.extend(str(t) for t in theta_hat_i[i])
        parts.extend(str(t2) for t2 in theta_dbl_hat_i[i])
        td = toy_data[i]
        parts.extend([str(c), str(td.eta), str(td.mu_b_below),
                      str(td.mu_b_above), str(td.r),
                      str(td.n_signal), str(td.n_bkg_below),
                      str(td.n_bkg_above)])
        parts.extend(str(ch) for ch in td.bkg_channel_indices)
        lines.append(",".join(parts))
    with open(fit_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    np.save(nll_max_path, nll_max)


def load_fit_results(name, m, c):
    """Load previously saved fit results from disk.

    Parameters
    ----------
    name : str
        Output directory name.
    m : float
        Hidden photon mass (eV).
    c : float
        Kinetic mixing parameter chi.

    Returns
    -------
    q_i : np.ndarray or None
        Test statistic array, or None if the CSV does not exist.
    nll_max : np.ndarray or None
        Unconstrained NLL array, or None if the .npy does not exist.
    """
    fit_path = f"{name}/fit_results/fit_results_{format_mass(m)}_{round_chi(c)}.csv"
    nll_max_path = f"{name}/fit_results/nll_max_{format_mass(m)}_{round_chi(c)}.npy"

    q_i = None
    nll_max = None
    if os.path.exists(fit_path):
        q_i = np.loadtxt(fit_path, delimiter=",", skiprows=1)[:, 0]
    if os.path.exists(nll_max_path):
        nll_max = np.load(nll_max_path)
    return q_i, nll_max


def compute_q_disc(q_values, n_sigma=3.0):
    """Compute the discovery threshold from a null q distribution.

    Parameters
    ----------
    q_values : np.ndarray
        Test statistic values (typically from the null hypothesis).
    n_sigma : float, optional
        Significance level in units of sigma (default 3.0).  Converted to
        a one-sided upper-tail percentile via the normal CDF:
        ``percentile = 100 * Phi(n_sigma)``.

    Returns
    -------
    float
        The q-value threshold at the requested significance level.
    """
    from scipy.stats import norm
    percentile = 100.0 * norm.cdf(n_sigma)
    return np.percentile(q_values, percentile)


def compute_local_significance(q_measured, q_null):
    """Return the local discovery significance in units of sigma.

    Compares a measured test statistic *q_measured* against the null
    distribution *q_null* obtained from ToyMC under the background-only
    hypothesis and converts the resulting p-value to a significance.

    Parameters
    ----------
    q_measured : float
        Observed test statistic value.
    q_null : np.ndarray
        Test statistic values from null-hypothesis ToyMC.

    Returns
    -------
    float
        Significance in units of sigma (one-sided).
    """
    from scipy.stats import norm
    p_value = np.mean(q_null >= q_measured)
    if p_value == 0:
        return np.inf
    return float(norm.isf(p_value))


def compute_fit_results(toy_data, c, m, physics, name,
                        interp_data, res_data, cutoffs, fit_bounds=None):
    """Compute (or load) per-toy PLR fit results for a (mass, chi) point.

    Orchestrates loading, computing, and saving fit results.

    Parameters
    ----------
    toy_data : list
        Toy experiment data.
    c : float
        Kinetic mixing parameter chi.
    m : float
        Hidden photon mass (eV).
    physics : PhysicsConstants
        Experiment constants.
    name : str
        Output directory name.
    interp_data : InterpolationData
        Interpolated data products.
    res_data : ResolutionData
        Resolution model.
    cutoffs : EnergyCutoffs
        Energy boundaries.

    Returns
    -------
    q_disc : float
        3-sigma Discovery threshold (99.865th percentile of null q distribution).
    nll_max : np.ndarray
        Best-fit NLL values from unconstrained fit.
    """
    q_i, nll_max = load_fit_results(name, m, c)

    if q_i is not None:
        # Results exist on disk; recover nll_max if missing
        if nll_max is None:
            nll_max = minimize_unconstrained(
                toy_data, physics, m, max(c, 1e-14),
                interp_data, res_data, cutoffs, fit_bounds,
            )[0]
            nll_max_path = f"{name}/fit_results/nll_max_{format_mass(m)}_{round_chi(c)}.npy"
            os.makedirs(f"{name}/fit_results", exist_ok=True)
            np.save(nll_max_path, nll_max)
        return compute_q_disc(q_i), nll_max

    q_i, nll_max, chi_hat_i, theta_hat_i, theta_dbl_hat_i = compute_q_values(
        toy_data, c, m, physics,
        interp_data, res_data, cutoffs, fit_bounds,
    )
    save_fit_results(name, m, c, q_i, chi_hat_i, theta_hat_i,
                     theta_dbl_hat_i, toy_data, nll_max)
    q_disc = compute_q_disc(q_i)
    del q_i, chi_hat_i, theta_hat_i, theta_dbl_hat_i
    gc.collect()
    return q_disc, nll_max


def plot_fit_results(m, chi_true, name):
    """Plot q-value distributions and q0 vs chi for a given mass.

    Parameters
    ----------
    m : float
        Hidden photon mass (eV).
    chi_true : np.ndarray
        Array of chi values to look for.
    name : str
        Output directory name.
    """
    import matplotlib.pyplot as plt
    fit_dir = f"{name}/fit_results"
    seen = set()
    chi_available = []
    for c in np.atleast_1d(chi_true).astype(float):
        ck = round_chi(c)
        path = f"{fit_dir}/fit_results_{format_mass(m)}_{ck}.csv"
        if os.path.exists(path) and ck not in seen:
            seen.add(ck)
            chi_available.append(ck)
    if not chi_available:
        return
    chi_true = np.asarray(chi_available, dtype=float)
    q_values = []
    q0_store = []
    for c in chi_true:
        q = []
        data = np.loadtxt(f"{fit_dir}/fit_results_{format_mass(m)}_{c}.csv", delimiter=",", skiprows=1)
        for row in data:
            if row[0] != np.inf:
                q.append(row[0])
        q_values.append(np.array(q))
    for idx, q in enumerate(q_values):
        q = np.array(sorted(list(q)))
        q0_store.append(np.percentile(q, 90))
        n_bins = 101
        bins = np.linspace(0, 10, n_bins)
        plt.hist(q, bins=bins, histtype="step", density=True,
                 label=f"$\\chi$ = {chi_true[idx]:.3g}")
    with open(f"{name}/fit_results/q0_{format_mass(m)}.csv", "w") as f:
        f.write("chi,q0\n")
        for i in range(len(chi_true)):
            f.write(f"{chi_true[i]},{q0_store[i]}\n")
    q_out = np.array(sorted(list(q_values[-1])))
    q_out=q_out[q_out > 0]
    g_asym = 1 / np.sqrt(2 * np.pi * q_out) * np.exp(-q_out / 2)
    plt.plot(q_out, g_asym, label="Asymptotic formula", color="black", linestyle="--")
    plt.legend(fontsize=8)
    plt.xlabel("$q(\\chi, \\theta)$")
    plt.xlim(-0.1, 3.1)
    plt.ylim(0.015, 5)
    plt.yscale("log")
    plt.title(f"$q$ histogram, $m$ = {m * 1000:.2f} meV")
    plt.savefig(f"{name}/q_hist_iminuit_{format_mass(m)}.png")
    plt.close()

    plt.plot(chi_true, q0_store, marker="o", markersize=3, label="$q_0$ (90th percentile)")
    plt.legend()
    plt.xlabel("$\\chi$")
    plt.ylabel("$q_0$")
    plt.xscale("log")
    plt.ylim(0, 5)
    plt.title(f"$q_0$ vs $\\chi$, $m$ = {m * 1000:.2f} meV")
    plt.savefig(f"{name}/q0_vs_chi_{format_mass(m)}.png")
    plt.close()
