# apres-efz-phase

Code implementing **CW-MLPR** (Coherence-Weighted Multi-Lag Phase Regression) and **MDI**
(Multi-band Decorrelation Inversion) — two amplitude-independent phase-coherence estimators that
recover vertical velocity and horizontal-decorrelation distributions in the ApRES Echo-Free Zone —
plus the scripts that generate every figure in:

> **Evidence for Coherent Phase Signal in the Antarctic Echo-Free Zone**
> (paper DOI: TBD upon publication)

An interactive demo of the full analysis pipeline is available here:
**https://huggingface.co/spaces/hstaehlin/apres-viewer**

## Install

```bash
uv sync            # or: pip install -e .
```

Dependencies are deliberately minimal: `numpy`, `scipy`, `matplotlib`, `zarr` (`make_F_airborne.py`
additionally needs `h5py` for the CReSIS MATLAB v7.3 file).

## Package layout

```
src/apres/
    velocity.py                # CW-MLPR: covariance_velocity[_profile], lag-coherence estimators
    decorrelation_inversion.py # MDI: forward_matrix, invert_pv, pv_stats
    forward_2d.py               # 2-D forward model used by both methods' synthetic-validation figures
    compressed_sensing.py       # MMV/CLEAN sparse recovery (volume-scattering evidence figure)
    io.py                       # raw ApRES .DAT loading
    superresolution.py          # matrix-pencil range super-resolution
figures/                        # one script per paper figure (see table below)
data/                           # small (<6 MB) cached intermediate results, included directly
figs/                           # figure scripts write their PDF/PNG output here
```

## Regenerating a figure

```bash
python figures/make_F_cwmlpr_synth.py
```

Each script writes its output to `figs/`. Several scripts cache expensive intermediate results
under `data/`; a small cache for each is already included so most figures reproduce immediately.
Delete the corresponding `data/*.npz` to force a from-scratch recompute.

| Figure | Script | Needs the real ApRES record? |
|---|---|---|
| `F_slope_concept.pdf` | `make_F_slope_concept.py` | No — conceptual diagram |
| `F_cwmlpr_synth.pdf` | `make_F_cwmlpr_synth.py` | No — synthetic validation only |
| `F_mdi_synth.pdf` | `make_F_mdi_synth.py` | No — synthetic validation only |
| `F_mdi_results.pdf` | `make_F_mdi_results.py` | No — ships with a precomputed cache |
| `F_site.pdf` | `make_F_site.py` | Needs `data/F0_amplitude_cache.npz` (see Data) |
| `Echogram_Mercer.pdf` | `make_F_airborne.py` | Needs the CReSIS airborne file (see Data) |
| `Fref.pdf`, `F_coherence.pdf`, `F5.pdf` | `make_paper_figures.py` | Yes — real ApRES record + raw `.DAT` files |
| `F_phase_detrend.pdf` | `make_phase_detrend_fig.py` | Yes — real ApRES record |
| `F_efz_volume_phase_vz06.pdf` | `make_F_efz_volume_phase_vz06.py` | Yes — real ApRES record |
| `F_mmv_lcurve.pdf` | `make_F_mmv_lcurve.py` | Yes — real ApRES record |

`cs_mmv_F_bands_lcurve.py` is not a figure script itself; it regenerates
`data/cs_mmv_F_bands_lcurve.npz` (the CS/MMV atom cache `make_F_mmv_lcurve.py` reads) from the real
record.

## Data

This repository intentionally does not vendor the underlying ApRES record — it's hundreds of MB
and already has a permanent home:

- **Raw ApRES data and original processing pipeline**: Siegfried, Venturelli, *et al.*, 2023,
  archived on Zenodo — data: [10.5281/zenodo.7597019](https://doi.org/10.5281/zenodo.7597019),
  code: [10.5281/zenodo.7605994](https://doi.org/10.5281/zenodo.7605994).
- **Airborne echogram**: Center for Remote Sensing of Ice Sheets (CReSIS), 2017_Antarctica_Basler
  season, frame `20171204_06_010`. Place the file at `data/airborne/Data_20171204_06_010.mat`.
- **Processed products** (`data/ImageP2_python.zarr`, `data/F0_amplitude_cache.npz`, raw `.DAT`
  files under `data/raw/`) are generated from the raw record via `apres.io` and are not included
  here due to size (up to ~700 MB); place them under `data/` using the paths referenced in each
  script. `make_paper_figures.py` will regenerate `F0_amplitude_cache.npz` automatically from
  `data/raw/*.DAT` on first run if it's missing.
- Small (<6 MB total) cached results that let most figures reproduce without the full pipeline are
  included directly under `data/`.

## License

BSD 3-Clause — see `LICENSE`.
