"""Sensitivity (upper/lower limit) computation via profile likelihood ratio.

Finds the upper and lower limit chi for each toy by directly solving
q(chi) = q0(chi) using per-toy bisection.  Each bisection step evaluates
every toy at its own midpoint chi in a single batched
``minimize_constrained`` call, giving O(log(range/tol)) total calls
instead of a fixed grid.

The upper limit is the high-chi crossing where q rises above q0.
The lower limit is the low-chi crossing where q drops below q0 (only
exists for signal-like toys whose q(chi) curve is U-shaped).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from qualiphide import format_mass
from qualiphide.minimization import minimize_constrained, create_constrained_pool
from scipy.interpolate import interp1d
import gc
from tqdm import tqdm

BRAZIL_BAND_PERCENTILES = [97.725, 84.1345, 50, 15.8655, 2.275]


def find_completed_masses(name):
    """Return sorted array of masses (eV) that have sensitivity CSV files.

    Scans ``<name>/sensitivity/`` for files matching
    ``sensitivity_<mass_str>.csv`` and converts the mass strings back to eV.
    The mass string uses the same ``format_mass`` convention (rounded to the
    nearest 0.1 meV).

    Parameters
    ----------
    name : str
        Output directory name (matching the YAML config name).

    Returns
    -------
    np.ndarray
        Sorted array of mass values in eV for which sensitivity has been
        computed.
    """
    sens_dir = f"{name}/sensitivity"
    if not os.path.isdir(sens_dir):
        return np.array([], dtype=float)
    prefix = "sensitivity_"
    suffix = "meV.csv"
    masses = []
    for fname in os.listdir(sens_dir):
        if fname.startswith(prefix) and fname.endswith(suffix):
            try:
                m_meV = float(fname[len(prefix):-len(suffix)])
                masses.append(m_meV / 1000.0)
            except ValueError:
                continue
    return np.sort(np.array(masses, dtype=float))


def load_q0_interpolator(name, m):
    """Load the q0 threshold file and return a 1-D interpolator.

    Parameters
    ----------
    name : str
        Output directory name.
    m : float
        Hidden photon mass (eV).

    Returns
    -------
    q0_func : scipy.interpolate.interp1d
        Interpolator mapping chi -> q0 threshold.
    """
    threshold = np.loadtxt(
        f"{name}/fit_results/q0_{format_mass(m)}.csv", delimiter=",", skiprows=1
    )
    return interp1d(
        threshold[:, 0], threshold[:, 1],
        kind="linear", bounds_error=False,
        fill_value=(threshold[0][1], threshold[-1][1]),
    )

# Bisection parameters
BISECT_TOL_LOG10 = 0.005   # convergence: log10(hi/lo) < tol
BISECT_MAX_ITER = 50        # safety cap


def _run_bisection(toy, m, physics, interp_data, res_data, cutoffs,
                   nll_max, q0_func, log_lo, log_hi, converged, pool,
                   upper_crossing=True, desc="  Bisection"):
    """Core bisection loop shared by upper and lower limit searches.

    Parameters
    ----------
    log_lo, log_hi : np.ndarray
        Per-toy bracket endpoints in log10(chi) space.  Modified in place.
    converged : np.ndarray of bool
        Per-toy convergence flags.  Modified in place.
    upper_crossing : bool
        If True (default), find the upper crossing (q rises through q0):
        ``q > q0 → tighten hi``.  If False, find the lower crossing
        (q drops through q0): ``q > q0 → tighten lo``.
    desc : str
        Progress bar description.

    Returns
    -------
    eval_history : list of (np.ndarray, np.ndarray)
        ``(chi_arr, q_arr)`` for each bisection step.
    """
    eval_history = []

    bisect_bar = tqdm(
        range(BISECT_MAX_ITER), desc=desc, unit="iter", leave=False,
    )
    for _ in bisect_bar:
        n_active = int((~converged).sum())
        if n_active == 0:
            break
        bisect_bar.set_postfix(active=n_active)

        log_mid = 0.5 * (log_lo + log_hi)
        chi_mid = 10.0 ** log_mid

        nll_mid, _ = minimize_constrained(
            toy, physics, m, chi_mid,
            interp_data, res_data, cutoffs, pool=pool,
        )
        q_mid = 2.0 * (nll_mid - nll_max)
        q0_mid = q0_func(chi_mid)
        eval_history.append((chi_mid.copy(), q_mid.copy()))

        above = q_mid > q0_mid
        active = ~converged
        if upper_crossing:
            # q > q0 means crossing is below mid → tighten hi
            log_hi[above & active] = log_mid[above & active]
            log_lo[~above & active] = log_mid[~above & active]
        else:
            # q > q0 means crossing is above mid → tighten lo
            log_lo[above & active] = log_mid[above & active]
            log_hi[~above & active] = log_mid[~above & active]

        bracket_width = log_hi - log_lo
        converged |= bracket_width < BISECT_TOL_LOG10

    bisect_bar.close()
    return eval_history


def compute_upper_limit(toy_datum, m, physics, chi_bounds, interp_data,
                        res_data, cutoffs, nll_max_single, q0_func,
                        fit_bounds=None, chi_hat_single=None):
    """Compute the PLR upper limit for a single toy via bisection.

    Bisects in log10(chi) space to find the high-chi crossing where
    q(chi) = q0(chi).

    Parameters
    ----------
    toy_datum : ToyData
        A single toy experiment (one element of the toy data list).
    m : float
        Hidden photon mass (eV).
    physics : PhysicsConstants
        Experiment constants.
    chi_bounds : tuple of float
        ``(chi_lo, chi_hi)`` bracket for the bisection search.
    interp_data : InterpolationData
        Interpolated data products.
    res_data : ResolutionData
        Resolution model.
    cutoffs : EnergyCutoffs
        Energy boundaries.
    nll_max_single : float
        Unconstrained NLL value for this toy.
    q0_func : callable
        Interpolator mapping chi -> q0 threshold.
    fit_bounds : FitBounds, optional
        Parameter bounds for the minimiser.
    chi_hat_single : float, optional
        Best-fit chi from the unconstrained fit for this toy.  Used to
        set the lower bracket for U-shaped toys so the bisection starts
        on the correct (right) branch.

    Returns
    -------
    upper_limit : float
        The upper limit chi value.  Returns ``chi_hi`` if the crossing is
        above the search range (unbracketable), or ``0.0`` if below and
        no chi_hat is available to fix the bracket.
    """
    toy = [toy_datum]
    nll_max = np.array([nll_max_single])
    chi_lo_bound, chi_hi_bound = chi_bounds

    pool = create_constrained_pool(
        physics, interp_data, toy, res_data, cutoffs, fit_bounds
    )

    try:
        # Check upper bound
        nll_hi, _ = minimize_constrained(
            toy, physics, m, chi_hi_bound,
            interp_data, res_data, cutoffs, pool=pool,
        )
        q_hi = 2.0 * (nll_hi[0] - nll_max_single)
        if q_hi < q0_func(chi_hi_bound):
            return chi_hi_bound

        # Check lower bound
        nll_lo, _ = minimize_constrained(
            toy, physics, m, chi_lo_bound,
            interp_data, res_data, cutoffs, pool=pool,
        )
        q_lo = 2.0 * (nll_lo[0] - nll_max_single)
        if q_lo > q0_func(chi_lo_bound) and chi_hat_single is None:
            return 0.0

        # Bisection for upper crossing
        # Always start at chi_hat when available — for non-U-shaped toys
        # with chi_hat ≈ 0, clip keeps lo_start at chi_lo_bound (no-op).
        if chi_hat_single is not None:
            lo_start = np.clip(chi_hat_single, chi_lo_bound, chi_hi_bound)
        else:
            lo_start = chi_lo_bound
        # Defensive floor: clip already ensures lo_start >= chi_lo_bound,
        # but this guards against a zero chi_lo_bound propagating into
        # log10.  Does not restrict the search range — chi_lo_bound is
        # always positive from callers (sensitivity and coverage).
        lo_start = max(lo_start, chi_lo_bound)

        # Guard: if q(lo_start) >= q0(lo_start), the valley doesn't dip
        # below threshold → entire range excluded, UL beyond search window.
        if lo_start > chi_lo_bound:
            nll_ch, _ = minimize_constrained(
                toy, physics, m, lo_start,
                interp_data, res_data, cutoffs, pool=pool,
            )
            if 2.0 * (nll_ch[0] - nll_max_single) >= q0_func(lo_start):
                return chi_hi_bound

        log_lo = np.array([np.log10(lo_start)])
        log_hi = np.array([np.log10(chi_hi_bound)])
        converged = np.array([False])

        _run_bisection(
            toy, m, physics, interp_data, res_data, cutoffs,
            nll_max, q0_func, log_lo, log_hi, converged, pool,
            upper_crossing=True,
        )

        return 10.0 ** (0.5 * (log_lo[0] + log_hi[0]))
    finally:
        pool.close()
        pool.join()


def compute_lower_limit(toy_datum, m, physics, chi_bounds, interp_data,
                        res_data, cutoffs, nll_max_single, q0_func,
                        upper_limit=None, fit_bounds=None,
                        chi_hat_single=None):
    """Compute the PLR lower limit for a single toy via bisection.

    Bisects in log10(chi) space to find the low-chi crossing where
    q(chi) drops below q0(chi).  A lower limit only exists for
    signal-like toys whose q(chi) curve is U-shaped (q > q0 at small chi).

    Parameters
    ----------
    toy_datum : ToyData
        A single toy experiment (one element of the toy data list).
    m : float
        Hidden photon mass (eV).
    physics : PhysicsConstants
        Experiment constants.
    chi_bounds : tuple of float
        ``(chi_lo, chi_hi)`` bracket for the bisection search.
    interp_data : InterpolationData
        Interpolated data products.
    res_data : ResolutionData
        Resolution model.
    cutoffs : EnergyCutoffs
        Energy boundaries.
    nll_max_single : float
        Unconstrained NLL value for this toy.
    q0_func : callable
        Interpolator mapping chi -> q0 threshold.
    upper_limit : float, optional
        Pre-computed upper limit for this toy.  Used as a fallback upper
        bracket endpoint.  If None, ``chi_hi`` is used.
    fit_bounds : FitBounds, optional
        Parameter bounds for the minimiser.
    chi_hat_single : float, optional
        Best-fit chi from the unconstrained fit for this toy.  Used as
        the upper bracket for the LL bisection (tighter than upper_limit).
        Falls back to upper_limit if not provided.

    Returns
    -------
    lower_limit : float
        The lower limit chi value, or ``0.0`` if no lower exclusion exists
        (q < q0 at the lower bracket boundary).
    """
    toy = [toy_datum]
    nll_max = np.array([nll_max_single])
    chi_lo_bound, chi_hi_bound = chi_bounds

    # Use min(chi_hat, upper_limit) as the hi bracket when available.
    # chi_hat is at the valley minimum (tighter bracket), with upper_limit
    # as a verified fallback if chi_hat overshoots.
    if chi_hat_single is not None and upper_limit is not None and upper_limit > 0:
        chi_hi_bracket = min(chi_hat_single, upper_limit)
    elif chi_hat_single is not None:
        chi_hi_bracket = chi_hat_single
    elif upper_limit is not None and upper_limit > 0:
        chi_hi_bracket = upper_limit
    else:
        chi_hi_bracket = chi_hi_bound
    # Defensive floor: chi_hat_single can be exactly 0 (no signal found)
    # which would cause log10(0) = -inf.  Flooring at chi_lo_bound keeps
    # the bracket valid without restricting the search range — if no
    # lower limit exists, the early return at q_lo check below handles it.
    chi_hi_bracket = max(chi_hi_bracket, chi_lo_bound)

    pool = create_constrained_pool(
        physics, interp_data, toy, res_data, cutoffs, fit_bounds
    )

    try:
        # Check if lower exclusion exists: q(chi_lo) must exceed q0(chi_lo)
        nll_lo, _ = minimize_constrained(
            toy, physics, m, chi_lo_bound,
            interp_data, res_data, cutoffs, pool=pool,
        )
        q_lo = 2.0 * (nll_lo[0] - nll_max_single)
        if q_lo <= q0_func(chi_lo_bound):
            return 0.0

        # Verify upper bracket: q(chi_hi_bracket) must be below q0
        nll_hi, _ = minimize_constrained(
            toy, physics, m, chi_hi_bracket,
            interp_data, res_data, cutoffs, pool=pool,
        )
        q_hi = 2.0 * (nll_hi[0] - nll_max_single)
        if q_hi > q0_func(chi_hi_bracket):
            # No valley found — entire range excluded or bracket too narrow
            return chi_hi_bracket

        # Bisection for lower crossing
        log_lo = np.array([np.log10(chi_lo_bound)])
        log_hi = np.array([np.log10(chi_hi_bracket)])
        converged = np.array([False])

        _run_bisection(
            toy, m, physics, interp_data, res_data, cutoffs,
            nll_max, q0_func, log_lo, log_hi, converged, pool,
            upper_crossing=False,
        )

        return 10.0 ** (0.5 * (log_lo[0] + log_hi[0]))
    finally:
        pool.close()
        pool.join()


def _bisect_all_toys(toy, n_toy, m, physics, chi_bounds, interp_data,
                     res_data, cutoffs, nll_max, q0_func, fit_bounds=None,
                     chi_hat=None):
    """Run vectorized bisection across all toys simultaneously.

    Finds both upper and lower limits.  All toys are evaluated in a single
    ``minimize_constrained`` call per iteration, each at its own midpoint
    chi.

    Parameters
    ----------
    toy : list
        Toy data list.
    n_toy : int
        Number of toys.
    m : float
        Hidden photon mass (eV).
    physics : PhysicsConstants
        Experiment constants.
    chi_bounds : tuple of float
        ``(chi_lo, chi_hi)`` bracket for the bisection search.
    interp_data : InterpolationData
        Interpolated data products.
    res_data : ResolutionData
        Resolution model.
    cutoffs : EnergyCutoffs
        Energy boundaries.
    nll_max : np.ndarray
        Unconstrained NLL values for each toy.
    q0_func : callable
        Interpolator mapping chi -> q0 threshold.
    fit_bounds : FitBounds, optional
        Parameter bounds for the minimiser.
    chi_hat : np.ndarray, optional
        Per-toy best-fit chi from the unconstrained fit.  Used to set the
        lower bracket for U-shaped toys so the UL bisection starts on the
        correct (right) branch.

    Returns
    -------
    LL : np.ndarray
        Per-toy lower limit chi values (0.0 if no lower exclusion).
    UL : np.ndarray
        Per-toy upper limit chi values.
    eval_history : list of (np.ndarray, np.ndarray)
        Bisection evaluation history for diagnostic plotting.
    """
    chi_lo_bound, chi_hi_bound = chi_bounds

    # Per-toy brackets in log10 space
    log_lo = np.full(n_toy, np.log10(chi_lo_bound))
    log_hi = np.full(n_toy, np.log10(chi_hi_bound))
    converged = np.zeros(n_toy, dtype=bool)

    eval_history = []

    pool = create_constrained_pool(
        physics, interp_data, toy, res_data, cutoffs, fit_bounds
    )

    try:
        # --- Verify bracket at both ends ---
        # Upper bound: all toys at chi_hi
        nll_hi, _ = minimize_constrained(
            toy, physics, m, chi_hi_bound,
            interp_data, res_data, cutoffs, pool=pool,
        )
        q_hi = 2.0 * (nll_hi - nll_max)
        q0_hi = q0_func(chi_hi_bound)
        unbracketable = q_hi < q0_hi   # q below threshold at chi_hi → no UL
        eval_history.append((np.full(n_toy, chi_hi_bound), q_hi.copy()))

        # Lower bound: all toys at chi_lo
        nll_lo, _ = minimize_constrained(
            toy, physics, m, chi_lo_bound,
            interp_data, res_data, cutoffs, pool=pool,
        )
        q_lo = 2.0 * (nll_lo - nll_max)
        q0_lo = q0_func(chi_lo_bound)
        below_range = q_lo > q0_lo      # q above threshold at chi_lo → LL exists
        eval_history.append((np.full(n_toy, chi_lo_bound), q_lo.copy()))

        # Only unbracketable toys can skip the UL bisection.
        # below_range toys (q > q0 at chi_lo) may still have a valid upper
        # crossing if q dips below q0 in the interior (U-shaped curve), so
        # they must participate in the UL bisection.
        converged = unbracketable.copy()

        # For toys with a significant chi_hat, the q(chi) curve may be
        # U-shaped with two crossings of q0.  The UL bisection must start
        # on the right branch (chi > chi_hat).  We apply this to ALL
        # non-unbracketable toys when chi_hat is available — for normal
        # toys with chi_hat ≈ 0, np.clip keeps log_lo at chi_lo_bound
        # (no-op), while for U-shaped toys it correctly shifts the bracket.
        if chi_hat is not None:
            active = ~unbracketable
            clamped = np.clip(chi_hat, chi_lo_bound, chi_hi_bound)
            # np.maximum is defensive: clip already ensures
            # clamped >= chi_lo_bound, but this prevents log10(0) if
            # chi_lo_bound were ever zero.  Does not restrict the search
            # range — chi_lo_bound is always positive from callers.
            log_clamped = np.log10(np.maximum(clamped, chi_lo_bound))
            log_lo[active] = np.maximum(
                log_lo[active], log_clamped[active]
            )

            # Guard: verify q(chi_hat) < q0(chi_hat) at the new bracket
            # lower bound.  If the valley doesn't dip below q0 (e.g. due
            # to noisy q0 or imprecise chi_hat), the entire [chi_hat, chi_hi]
            # range is excluded and UL should be chi_hi_bound.
            shifted = active & (clamped > chi_lo_bound)
            if np.any(shifted):
                nll_ch, _ = minimize_constrained(
                    toy, physics, m, clamped,
                    interp_data, res_data, cutoffs, pool=pool,
                )
                q_ch = 2.0 * (nll_ch - nll_max)
                q0_ch = q0_func(clamped)
                fully_excluded = shifted & (q_ch >= q0_ch)
                converged[fully_excluded] = True
                # These toys have q > q0 across the whole range → UL is
                # beyond our search window; assign chi_hi_bound later.
                unbracketable[fully_excluded] = True

        # --- Upper limit bisection ---
        ul_history = _run_bisection(
            toy, m, physics, interp_data, res_data, cutoffs,
            nll_max, q0_func, log_lo, log_hi, converged, pool,
            upper_crossing=True, desc="  UL Bisection",
        )
        eval_history.extend(ul_history)

        # Compute upper limits
        UL = 10.0 ** (0.5 * (log_lo + log_hi))
        UL[unbracketable] = chi_hi_bound

        # --- Lower limit bisection ---
        # A lower limit exists only for toys where q(chi_lo) > q0(chi_lo)
        # and that are not fully excluded (q > q0 across the whole range).
        needs_ll = below_range & ~unbracketable
        LL = np.zeros(n_toy)

        if np.any(needs_ll):
            # Bracket: [chi_lo_bound, min(chi_hat, UL)] per toy.
            # chi_hat is the valley minimum where q < q0, so the left crossing
            # lies between chi_lo_bound (where q > q0) and chi_hat.
            # Use min(chi_hat, UL) so that if chi_hat overshoots or is
            # imprecise, UL (a verified q=q0 crossing) acts as a safe
            # fallback upper bracket.
            log_lo_ll = np.full(n_toy, np.log10(chi_lo_bound))
            if chi_hat is not None:
                ll_hi = np.minimum(chi_hat, UL)
                log_hi_ll = np.log10(np.maximum(ll_hi, chi_lo_bound))
            else:
                log_hi_ll = np.log10(UL)

            converged_ll = ~needs_ll

            _run_bisection(
                toy, m, physics, interp_data, res_data, cutoffs,
                nll_max, q0_func, log_lo_ll, log_hi_ll, converged_ll, pool,
                upper_crossing=False, desc="  LL Bisection",
            )

            LL[needs_ll] = 10.0 ** (
                0.5 * (log_lo_ll[needs_ll] + log_hi_ll[needs_ll])
            )
    finally:
        pool.close()
        pool.join()

    return LL, UL, eval_history


def compute_sensitivity(toy, n_toy, m, physics, name, chi_bounds,
                        interp_data, res_data, cutoffs, nll_max,
                        fit_bounds=None):
    """Compute PLR upper and lower limits via per-toy bisection.

    Each toy independently bisects to find the chi where q(chi) = q0(chi).
    All toys are evaluated in a single ``minimize_constrained`` call per
    iteration, each at its own midpoint chi.

    Parameters
    ----------
    toy : list
        Toy data list.
    n_toy : int
        Number of toys.
    m : float
        Hidden photon mass (eV).
    physics : PhysicsConstants
        Experiment constants.
    name : str
        Output directory name.
    chi_bounds : tuple of float
        ``(chi_lo, chi_hi)`` bracket for the bisection search.
    interp_data : InterpolationData
        Interpolated data products.
    res_data : ResolutionData
        Resolution model.
    cutoffs : EnergyCutoffs
        Energy boundaries.
    nll_max : np.ndarray
        Unconstrained NLL values for each toy.
    fit_bounds : FitBounds, optional
        Parameter bounds for the minimiser.
    """
    q0_func = load_q0_interpolator(name, m)

    # Load chi_hat from null fit results to fix UL bisection for U-shaped toys
    chi_hat = None
    fit_path = f"{name}/fit_results/fit_results_{format_mass(m)}_0.0.csv"
    if os.path.exists(fit_path):
        chi_hat = np.loadtxt(fit_path, delimiter=",", skiprows=1,
                             usecols=(1,))
        if len(chi_hat) != n_toy:
            chi_hat = None  # length mismatch, fall back to old behavior

    os.makedirs(f"{name}/sensitivity", exist_ok=True)

    LL, UL, eval_history = _bisect_all_toys(
        toy, n_toy, m, physics, chi_bounds, interp_data,
        res_data, cutoffs, nll_max, q0_func, fit_bounds,
        chi_hat=chi_hat,
    )

    # --- Save ---
    with open(f"{name}/sensitivity/sensitivity_{format_mass(m)}.csv", "w") as f:
        f.write("lower_limit,upper_limit\n")
        for ll, ul in zip(LL, UL):
            f.write(f"{ll},{ul}\n")

    # --- Diagnostic plots ---
    _plot_bisection_diagnostic(eval_history, q0_func, UL, n_toy, m, name)
    UL_pos = UL[UL > 0]
    if len(UL_pos) > 0:
        plt.hist(UL_pos, bins=50)
        plt.title("Upper Limits, $m_{{test}}$ = {:.2f} meV".format(m * 1000))
        plt.xlabel("$\\chi_{UL}$")
        plt.ylabel("Count")
        plt.savefig(f"{name}/sensitivity_{format_mass(m)}.png")
        plt.close()

    del toy
    gc.collect()


def _plot_bisection_diagnostic(eval_history, q0_func, UL, n_toy, m, name):
    """Plot representative q(chi) traces from bisection for diagnostic."""
    if not eval_history:
        return

    # Build per-toy (chi, q) traces sorted by chi
    toy_chis = [[] for _ in range(n_toy)]
    toy_qs = [[] for _ in range(n_toy)]
    for chi_arr, q_arr in eval_history:
        for j in range(n_toy):
            toy_chis[j].append(chi_arr[j])
            toy_qs[j].append(q_arr[j])

    # Pick representative toys by UL percentile
    UL_valid = UL.copy()
    UL_valid[UL_valid <= 0] = np.inf
    finite = UL_valid < np.inf
    if not np.any(finite):
        return

    rep_idx = []
    for r in BRAZIL_BAND_PERCENTILES:
        target = np.percentile(UL_valid[finite], r)
        rep_idx.append(int(np.argmin(np.abs(UL_valid - target))))

    _, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = [
        "$-2\\sigma$ (97.7%)", "$-1\\sigma$ (84.1%)", "Median (50%)",
        "$+1\\sigma$ (15.9%)", "$+2\\sigma$ (2.3%)",
    ]
    for idx, lbl in zip(rep_idx, labels):
        chis_j = np.array(toy_chis[idx])
        qs_j = np.array(toy_qs[idx])
        order = np.argsort(chis_j)
        ax.plot(chis_j[order], qs_j[order], "o-", markersize=3, label=lbl)

    # Plot q0 threshold
    all_chis = np.concatenate([np.array(tc) for tc in toy_chis])
    chi_range = np.logspace(
        np.log10(all_chis[all_chis > 0].min()),
        np.log10(all_chis.max()), 200,
    )
    ax.plot(
        chi_range, q0_func(chi_range),
        color="black", linestyle="dashed", label="$q_0$ threshold",
    )
    ax.set_ylim(-0.1, max(q0_func(chi_range)))
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("$\\chi$")
    ax.set_ylabel("$q$")
    ax.set_title(
        "Bisection $q(\\chi)$, $m_{{test}}$ = {:.2f} meV".format(m * 1000)
    )
    plt.savefig(f"{name}/q_parabolas_{format_mass(m)}.png")
    plt.close()


def plot_sensitivity_band(
    m_test,
    plot_scale,
    name,
    limits=None,
    save=True,
    save_data=False,
    data_filename="sensitivity_band.csv",
):
    """Plot the Brazil-band sensitivity across the mass grid.

    Parameters
    ----------
    m_test : np.ndarray
        Array of test masses (eV).
    plot_scale : str
        Axis scale (``"log"`` or ``"linear"``).
    name : str
        Output directory name.
    limits : array-like, optional
        Per-mass limit values to overlay on the band.
    save : bool, optional
        If True, save the plot as ``<name>/sensitivity_band.png``.
    save_data : bool, optional
        If True, save band data as CSV in ``<name>/sensitivity``.
    data_filename : str, optional
        CSV file name used when ``save_data`` is True.
    """
    m_test = np.asarray(m_test)
    if limits is not None:
        limits = np.asarray(limits)
        if limits.shape[0] != m_test.shape[0]:
            raise ValueError("`limits` must have the same length as `m_test`.")

    percentiles = []
    percs = BRAZIL_BAND_PERCENTILES
    for m in m_test:
        p0 = []
        UL = np.loadtxt(
            f"{name}/sensitivity/sensitivity_{format_mass(m)}.csv",
            delimiter=",",
            skiprows=1,
        )[:, 1]
        UL = [u for u in UL if u != 0]
        for p in percs:
            p0.append(np.percentile(UL, p))
        percentiles.append(p0)
    per_mass_percentiles = np.array(percentiles)
    percentiles = np.transpose(per_mass_percentiles)

    if save_data:
        os.makedirs(f"{name}/sensitivity", exist_ok=True)
        data_path = os.path.join(name, "sensitivity", data_filename)

        minus_2sigma = per_mass_percentiles[:, 0]
        minus_1sigma = per_mass_percentiles[:, 1]
        median = per_mass_percentiles[:, 2]
        plus_1sigma = per_mass_percentiles[:, 3]
        plus_2sigma = per_mass_percentiles[:, 4]
        limit_col = np.full(m_test.shape, np.nan) if limits is None else limits

        out = np.column_stack(
            [
                m_test,
                plus_2sigma,
                plus_1sigma,
                median,
                minus_1sigma,
                minus_2sigma,
                limit_col,
            ]
        )
        header = (
            "m_test,lower_2sigma,lower_1sigma,median,"
            "upper_1sigma,upper_2sigma,limit"
        )
        np.savetxt(data_path, out, delimiter=",", header=header, comments="")

    band_labels = [
        "$-2\\sigma$ (97.7%)", "$-1\\sigma$ (84.1%)", "Median (50%)",
        "$+1\\sigma$ (15.9%)", "$+2\\sigma$ (2.3%)",
    ]
    band_colors = ["C0", "C1", "C2", "C3", "C4"]
    for idx, p in enumerate(percentiles):
        plt.plot(m_test, p, label=band_labels[idx], color=band_colors[idx])
    plt.legend(fontsize=8)
    for p in range(len(percentiles) - 1):
        col = "green"
        if p in [0, 3]:
            col = "yellow"
        plt.fill_between(
            m_test, percentiles[p], percentiles[p + 1], color=col, alpha=0.5
        )
    if limits is not None:
        plt.plot(m_test, limits, color="black", label="Limit")
    plt.xlabel("$m_{\\mathrm{test}}$ (eV)")
    plt.ylabel("$\\chi_{UL}$")
    plt.title("Sensitivity Band (PLR Upper Limits)")
    plt.xscale(plot_scale)
    plt.yscale(plot_scale)
    plt.xlim(m_test[0], m_test[-1])
    if save:
        plt.savefig(f"{name}/sensitivity_band.png")
        plt.close()
    else:
        plt.show()
