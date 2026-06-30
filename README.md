# qualiphide_fir_hp_inference

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21077055.svg)](https://doi.org/10.5281/zenodo.21077055)

Profile-likelihood inference pipeline for the QUALIPHIDE FIR hidden-photon search. The statistical methodology follows [arXiv:1902.11297](https://arxiv.org/abs/1902.11297). Generates Toy Monte Carlo pseudo-experiments and performs profile likelihood ratio (PLR) tests to compute sensitivity bands and 3σ discovery thresholds.

This repository is the canonical public release for the QUALIPHIDE FIR analysis. Development history lives in [QUALIPHIDE-Inferences](https://github.com/WashU-Astroparticle-Lab/QUALIPHIDE-Inferences); experiment data and notebooks are in [qualiphide_thz](https://github.com/WashU-Astroparticle-Lab/qualiphide_thz).

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/WashU-Astroparticle-Lab/qualiphide_fir_hp_inference.git
cd qualiphide_fir_hp_inference
pip install -e .

# Optional: Jupyter support for the tutorial notebook
pip install -e ".[notebook]"
```

## Quick Start

```bash
# Run the full pipeline (uses configs/config.yaml by default)
qualiphide

# Explicit config name also works
qualiphide config
```

For a quick test, temporarily reduce `energy_data.n_toy` and the mass grid in `configs/config.yaml`. A step-by-step walkthrough is in [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb).

## Configuration

All analysis parameters are in a single file: [`configs/config.yaml`](configs/config.yaml). Pipeline outputs are written to `results/` (set via `output_dir` in the config).

## Published results

Pre-computed summary tables from the production run are committed under `results/`:

| File | Description |
|------|-------------|
| `sensitivity_band.csv` | Brazil-band upper limits on kinetic mixing χ |
| `three_sigma_discovery.csv` | 3σ discovery probability vs. mass and χ |
| `discovery_thresholds.csv` | χ thresholds at fixed discovery probabilities |

Re-running `qualiphide` regenerates full outputs under `results/` (including per-mass sensitivity files and fit results).

## Unblinded event data

Real unblinded photon events are **not** bundled here. Download from [qualiphide_thz](https://github.com/WashU-Astroparticle-Lab/qualiphide_thz/blob/main/important/final_unblinded_all_channel_photon_events.csv) if needed for unblinding studies.

## Pipeline overview

For each test mass *m* in the grid:

1. **Null ToyMC** — pseudo-experiments at χ = 0
2. **Null q-histogram** — PLR test statistic *q*; discovery threshold *q*<sub>disc</sub>
3. **Signal loop** — signal ToyMC, discovery probability, dynamic χ grid extension
4. **Sensitivity** — PLR upper limits → Brazil-band percentiles
5. **Plotting** — sensitivity band, PLR curves, discovery heatmap

The pipeline supports **resuming** from existing output files.

## Package structure

```
configs/config.yaml          Analysis configuration
qualiphide/                  Inference package
notebooks/tutorial.ipynb     Step-by-step tutorial
results/                     Published summary CSVs (+ regenerated run outputs)
```

## Physics background

The analysis follows [arXiv:1902.11297](https://arxiv.org/abs/1902.11297). Key symbols: χ (kinetic mixing), *m* (hidden photon mass), η (efficiency), μ<sub>b</sub> (background rate), *q* (profile likelihood ratio).

## Citation

If you use this software, please cite:

```bibtex
@software{qualiphide_fir_hp_inference,
  author       = {Harris, Jacob and Yuan, Lanqing},
  title        = {{qualiphide\_fir\_hp\_inference}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.21077055},
  url          = {https://github.com/WashU-Astroparticle-Lab/qualiphide_fir_hp_inference}
}
```

See also [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
