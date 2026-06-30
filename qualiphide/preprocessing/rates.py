"""Background rate fitting and 4-KID convolution.

Reads per-channel background rate integrals, fits skew-normal distributions,
and convolves four single-KID distributions to obtain the combined 4-KID
expected background count PDF.  Results are written to
``<name>/mu_b_sampler_below.csv`` and ``<name>/mu_b_sampler_above.csv``.
"""

from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

from qualiphide import DATA_DIR


def fit_background_rates(name, cutoff, split_mu_b, maximum, runtime):
    """Fit background rates and compute 4-KID convolved PDF.

    Parameters
    ----------
    name : str
        Output directory name.
    cutoff : float
        Lower energy cutoff (eV).
    split_mu_b : float
        Energy separating below/above background regions (eV).
    maximum : float
        Upper energy cutoff (eV).
    runtime : float
        Observation time in seconds.
    """
    data = np.loadtxt(
        str(DATA_DIR / "sr3pt2_bkg_channels_Hz_meV_remade.csv"),
        delimiter=",",
        skiprows=1,
    )

    rates_below, rates_above = [], []
    energies = data[:, 1]
    mask_below = (energies >= 1000 * cutoff) & (energies < 1000 * split_mu_b)
    mask_above = (energies >= 1000 * split_mu_b) & (energies <= 1000 * maximum)
    for c in range(2, len(data[0])):
        values = data[:, c]
        values_below = np.where(mask_below, values, 0)
        values_above = np.where(mask_above, values, 0)
        if np.trapezoid(values, x=energies) != 0.0:
            rates_below.append(np.trapezoid(values_below, x=energies))
            rates_above.append(np.trapezoid(values_above, x=energies))

    rates_below.sort()
    _, bins, _ = plt.hist(rates_below, density=True)
    alpha, loc, scale = skewnorm.fit(rates_below)
    x = np.linspace(min(bins), max(bins), 1000)
    pdf_fitted = skewnorm.pdf(x, alpha, loc, scale)
    pdf_fitted /= np.trapezoid(pdf_fitted, x=x)
    plt.plot(x, pdf_fitted)
    plt.grid()
    plt.xlabel("Rates (Hz)")
    plt.savefig(f"{name}/rate_hist_below.png")
    plt.close()

    alpha2, loc2, scale2 = skewnorm.fit(runtime * np.array(rates_below))
    mu_b = runtime * np.linspace(min(bins), max(bins), 1000)
    pdf_fitted_2 = skewnorm.pdf(mu_b, alpha2, loc2, scale2)
    pdf_fitted_2 /= np.trapezoid(pdf_fitted_2, x=mu_b)
    dmu = mu_b[1] - mu_b[0]
    pdf = pdf_fitted_2.copy()
    for i in range(3):
        pdf = fftconvolve(pdf, pdf_fitted_2, mode="full") * dmu
    mu_b_4 = np.linspace(4 * mu_b[0], 4 * mu_b[-1], len(pdf))
    pdf /= np.trapezoid(pdf, x=mu_b_4)
    plt.plot(mu_b_4, pdf)
    plt.grid()
    plt.xlabel("mu_b")
    plt.savefig(f"{name}/mu_b_func_below.png")
    plt.close()

    with open(f"{name}/mu_b_sampler_below.csv", "w") as f:
        f.write("mu_b,pdf\n")
        for i, m in enumerate(mu_b_4):
            f.write(f"{m},{pdf[i]}\n")

    rates_above.sort()
    _, bins, _ = plt.hist(rates_above, density=True)
    alpha, loc, scale = skewnorm.fit(rates_above)
    x = np.linspace(min(bins), max(bins), 1000)
    pdf_fitted = skewnorm.pdf(x, alpha, loc, scale)
    pdf_fitted /= np.trapezoid(pdf_fitted, x=x)
    plt.plot(x, pdf_fitted)
    plt.grid()
    plt.xlabel("Rates (Hz)")
    plt.savefig(f"{name}/rate_hist_above.png")
    plt.close()

    alpha2, loc2, scale2 = skewnorm.fit(runtime * np.array(rates_above))
    mu_b = runtime * np.linspace(min(bins), max(bins), 1000)
    pdf_fitted_2 = skewnorm.pdf(mu_b, alpha2, loc2, scale2)
    pdf_fitted_2 /= np.trapezoid(pdf_fitted_2, x=mu_b)
    dmu = mu_b[1] - mu_b[0]
    pdf = pdf_fitted_2.copy()
    for i in range(3):
        pdf = fftconvolve(pdf, pdf_fitted_2, mode="full") * dmu
    mu_b_4 = np.linspace(4 * mu_b[0], 4 * mu_b[-1], len(pdf))
    pdf /= np.trapezoid(pdf, x=mu_b_4)
    plt.plot(mu_b_4, pdf)
    plt.grid()
    plt.xlabel("mu_b")
    plt.savefig(f"{name}/mu_b_func_above.png")
    plt.close()

    with open(f"{name}/mu_b_sampler_above.csv", "w") as f:
        f.write("mu_b,pdf\n")
        for i, m in enumerate(mu_b_4):
            f.write(f"{m},{pdf[i]}\n")
