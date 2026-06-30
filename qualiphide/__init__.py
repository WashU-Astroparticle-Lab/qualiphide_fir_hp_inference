"""QUALIPHIDE — Statistical inference for the QUALIPHIDE dark photon search."""

import os
import pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"

# Default worker count; updated by load_config() from YAML max_workers field.
# Can also be overridden via QUALIPHIDE_WORKERS env var.
_DEFAULT_MAX_WORKERS = 16
N_WORKERS = min(
    os.cpu_count() or 4,
    int(os.environ.get("QUALIPHIDE_WORKERS", _DEFAULT_MAX_WORKERS)),
)


def format_mass(m):
    """Format a mass value (eV) to the nearest 0.1 meV for filenames.

    Rounding to a fixed 0.1 meV resolution (rather than a fixed number of
    significant figures) keeps the filename precision uniform across the
    whole mass range, so two distinct grid points can never collapse to the
    same string and overwrite each other's results.
    """
    m_meV = m * 1000
    return f"{m_meV:.1f}meV"


def format_chi(c):
    """Format a chi value to 3 significant figures for filenames."""
    return f"{float(f'{c:.3g}')}"
