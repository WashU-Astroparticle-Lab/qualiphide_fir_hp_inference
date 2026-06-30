"""Profile likelihood for the QUALIPHIDE hidden photon search.

The likelihood combines a Poisson term for the observed event count with
per-event signal/background mixture weights and Gaussian ancillary
constraints on the nuisance parameters (eta, mu_b_below, mu_b_above, r).
"""

import numpy as np
from math import log, sqrt, pi, lgamma

import numba as nb

TWO_PI = 2.0 * pi
SQRT_TWO_PI = sqrt(TWO_PI)
INV_SQRT_TWO_PI = 1.0 / SQRT_TWO_PI

# Gaussian values below exp(-0.5 * CUTOFF_NSIGMA^2) are negligible
CUTOFF_NSIGMA = 10.0  # exp(-0.5*10^2) ≈ 1.9e-22


@nb.njit(cache=True)
def _compute_normalization_windowed(E, eta_a, m, resolution, mf_start, mf_end,
                                    ma_start, ma_end):
    """Compute signal PDF normalization with a windowed integration.

    Only integrates where the Gaussian is non-negligible (within ±CUTOFF_NSIGMA*σ
    of mass m), which is typically ~15% of the full MASK_FULL range.

    Returns (N_full, flat_value).
    """
    inv_res = 1.0 / resolution
    norm_coeff = inv_res * INV_SQRT_TWO_PI
    half_width = CUTOFF_NSIGMA * resolution

    # Narrow the integration window for N_full
    e_lo = m - half_width
    e_hi = m + half_width

    # Binary search for start index (first E[i] >= e_lo within mf range)
    lo = mf_start
    hi = mf_end
    while lo < hi:
        mid = (lo + hi) // 2
        if E[mid] < e_lo:
            lo = mid + 1
        else:
            hi = mid
    win_start = lo

    # Binary search for end index (first E[i] > e_hi within mf range)
    lo = win_start
    hi = mf_end
    while lo < hi:
        mid = (lo + hi) // 2
        if E[mid] <= e_hi:
            lo = mid + 1
        else:
            hi = mid
    win_end = lo

    if win_start < mf_start:
        win_start = mf_start
    if win_end > mf_end:
        win_end = mf_end

    if win_end <= win_start + 1:
        return 0.0, 0.0

    # Trapezoid of Gaussian*eta_a over the narrow window
    N_full = 0.0
    prev_val = norm_coeff * np.exp(-0.5 * ((E[win_start] - m) * inv_res) ** 2) * eta_a[win_start]
    for i in range(win_start + 1, win_end):
        cur_val = norm_coeff * np.exp(-0.5 * ((E[i] - m) * inv_res) ** 2) * eta_a[i]
        N_full += (prev_val + cur_val) * (E[i] - E[i - 1])
        prev_val = cur_val
    N_full *= 0.5

    if N_full < 1e-300:
        return N_full, 0.0

    # Integral above split: narrow window intersected with [ma_start, ma_end)
    above_win_start = ma_start if ma_start > win_start else win_start
    above_win_end = ma_end if ma_end < win_end else win_end

    inv_N = 1.0 / N_full
    integral_above = 0.0

    if above_win_end > above_win_start + 1:
        prev_fs = norm_coeff * np.exp(-0.5 * ((E[above_win_start] - m) * inv_res) ** 2) * eta_a[above_win_start] * inv_N
        for i in range(above_win_start + 1, above_win_end):
            cur_fs = norm_coeff * np.exp(-0.5 * ((E[i] - m) * inv_res) ** 2) * eta_a[i] * inv_N
            integral_above += (prev_fs + cur_fs) * (E[i] - E[i - 1])
            prev_fs = cur_fs
        integral_above *= 0.5

    width_above = E[ma_end - 1] - E[ma_start]
    if width_above < 1e-300:
        return N_full, 0.0

    flat_value = integral_above / width_above
    return N_full, flat_value


def _generate_signal_pdf_fast(m, resolution, E, eta_a, MASK_FULL, MASK_ABOVE):
    """Compute the signal PDF on a pre-masked energy grid (optimised path).

    This is a speed-optimised version of :func:`qualiphide.signal.generate_signal_pdf`
    used inside the minimisation loop where the energy masks are precomputed.

    Parameters
    ----------
    m : float
        Hidden photon mass (eV).
    resolution : float
        Energy resolution sigma (eV).
    E : np.ndarray
        Full energy grid.
    eta_a : np.ndarray
        Aperture efficiency on *E*.
    MASK_FULL : np.ndarray
        Boolean mask for the full analysis window.
    MASK_ABOVE : np.ndarray
        Boolean mask for energies >= split.

    Returns
    -------
    np.ndarray
        Signal PDF on *E*.
    """
    f_s = np.zeros_like(E)

    f_s[MASK_FULL] = (
        1 / np.sqrt(2 * np.pi * resolution)
        * np.exp(-0.5 * ((E[MASK_FULL] - m) / resolution) ** 2)
    )
    f_s[MASK_FULL] *= eta_a[MASK_FULL]

    N_full = np.trapezoid(f_s, x=E)
    f_s /= N_full

    integral_above = np.trapezoid(f_s[MASK_ABOVE], x=E[MASK_ABOVE])
    width_above = E[MASK_ABOVE][-1] - E[MASK_ABOVE][0]

    f_s_above = np.zeros_like(f_s)
    f_s_above[MASK_ABOVE] = integral_above / width_above

    f_s_below = f_s.copy()
    f_s_below[MASK_ABOVE] = 0

    return f_s_above + f_s_below


def log_likelihood(eta, mu_b_below, mu_b_above, r, N_gamma_det, arguments):
    """Evaluate the full log-likelihood for one toy experiment.

    Parameters
    ----------
    eta : float
        Detector efficiency (nuisance).
    mu_b_below : float
        Expected background count below split (nuisance).
    mu_b_above : float
        Expected background count above split (nuisance).
    r : float
        Energy resolution nuisance parameter (standard-normal units).
    N_gamma_det : float
        Expected detected signal photon count.
    arguments : tuple
        ``(E_in, N_KID, eta_used, sigma_eta_used, m,
        f_b_val_below, f_b_val_above, G)`` — per-toy constants and the
        shared global dict *G* carrying precomputed arrays.

    Returns
    -------
    float
        Log-likelihood value (returns ``1e10`` for unphysical parameters).
    """
    (E_in, N_KID, eta_used, sigma_eta_used, m,
     f_b_val_below, f_b_val_above, G) = arguments
    E = G["E"]
    eta_a = G["eta_a"]
    mu_b_data_below = G["mu_b_data_below"]
    mu_b_data_above = G["mu_b_data_above"]

    mu_s = eta * N_gamma_det / N_KID
    mu_tot = mu_s + mu_b_below + mu_b_above

    if mu_s < 0 or mu_b_below < 0 or mu_b_above < 0 or mu_tot <= 0:
        return 1e10

    res_data = G["res_data"]
    resolution = (
        res_data.median + r * res_data.delta_low
        if r < 0
        else res_data.median + r * res_data.delta_up
    )
    if resolution < 1e-6:
        resolution = 1e-6

    # Normalization via numba on a windowed region of the grid
    mf_start = G["mf_start"]
    mf_end = G["mf_end"]
    ma_start = G["ma_start"]
    ma_end = G["ma_end"]

    N_full, flat_value = _compute_normalization_windowed(
        E, eta_a, m, resolution, mf_start, mf_end, ma_start, ma_end
    )

    if N_full < 1e-300:
        return 1e10

    # Compute signal PDF directly at E_in points using vectorized numpy
    # Precomputed: eta_a_at_Ein, above_mask_Ein (set once per toy)
    eta_a_at_Ein = G["_eta_a_Ein_cache"]
    above_mask_Ein = G["_above_mask_Ein"]
    f_s_buf = G["_f_s_buf"]

    inv_res = 1.0 / resolution
    # Gaussian * eta_a / N_full for all points
    np.subtract(E_in, m, out=f_s_buf)
    f_s_buf *= inv_res
    np.square(f_s_buf, out=f_s_buf)
    f_s_buf *= -0.5
    np.exp(f_s_buf, out=f_s_buf)
    f_s_buf *= (inv_res * INV_SQRT_TWO_PI / N_full)
    f_s_buf *= eta_a_at_Ein
    # Clamp to 1e-16
    np.maximum(f_s_buf, 1e-16, out=f_s_buf)
    # Replace above-split points with flat_value
    f_s_buf[above_mask_Ein] = flat_value if flat_value > 1e-16 else 1e-16

    # Mixture and log-sum
    f_s_buf *= mu_s
    f_s_buf += mu_b_below * f_b_val_below
    f_s_buf += mu_b_above * f_b_val_above
    np.log(f_s_buf, out=f_s_buf)
    log_sum = f_s_buf.sum() - len(E_in) * log(mu_tot)

    # Poisson term: logpmf(n, mu) = n*log(mu) - mu - lgamma(n+1)
    n = len(E_in)
    logL = n * log(mu_tot) - mu_tot - G["_lgamma_cache"]
    logL += log_sum

    anc_eta = -0.5 * ((eta - eta_used) / sigma_eta_used) ** 2

    pdf_mu_b_below = np.interp(
        mu_b_below,
        mu_b_data_below[0],
        mu_b_data_below[1],
        left=mu_b_data_below[1][0],
        right=mu_b_data_below[1][-1],
    )
    if pdf_mu_b_below <= 0:
        return 1e10
    anc_mu_b_below = log(pdf_mu_b_below)

    pdf_mu_b_above = np.interp(
        mu_b_above,
        mu_b_data_above[0],
        mu_b_data_above[1],
        left=mu_b_data_above[1][0],
        right=mu_b_data_above[1][-1],
    )
    if pdf_mu_b_above <= 0:
        return 1e10
    anc_mu_b_above = log(pdf_mu_b_above)

    anc_r = -0.5 * r * r
    logL += anc_eta + anc_mu_b_below + anc_mu_b_above + anc_r
    return logL
