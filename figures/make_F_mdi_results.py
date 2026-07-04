"""
F_mdi_results — MDI *results* figure for the paper (Fig. "mdi_results").

Two panels, single column:

  (a) Clean high-resolution per-bin MDI P(v_h, z) over the whole depth
      range (20–1100 m), computed per range bin with a 10-bin depth
      window from the three FMCW sub-bands, no noise gate.
  (b) Median + 16–84 % range of the per-bin distributions vs depth.

NOTE on the velocity axis: MDI recovers the beam-sweep DECORRELATION
RATE distribution; the velocity axis uses the nominal antenna spread
σ_θ = 0.22 rad and is therefore a model-dependent scale, not a
calibrated absolute velocity (see the paper's calibration subsection).

The expensive per-bin inversion is cached to
  data/mdi_perbin_highres.npz
and reused on subsequent runs.

Output:
  figs/F_mdi_results.pdf / .png
"""

from __future__ import annotations
import sys
import time
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import zarr
import warnings
warnings.filterwarnings('ignore')

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import paper_style as ps
ps.apply_style()

from apres.decorrelation_inversion import (
    forward_matrix, invert_pv, pv_stats, LAMBDAS_DEFAULT,
)
from apres.velocity import lag_coherence_spatial
SIGMA_THETA = 0.22   # antenna beam angular std (rad), ~30 deg 3-dB beamwidth

OUT_DIR = ROOT / 'figs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / 'data' / 'mdi_perbin_highres.npz'

# ── Config ─────────────────────────────────────────────────────────────
ZARR_BASE  = ROOT / 'data' / 'ImageP2_python.zarr'
BANDS      = ('low', 'full', 'high')
DEPTH_MIN  = 20.0
DEPTH_MAX  = 1100.0
BINWIN     = 10          # depth window for the per-bin inversion (bins)
ALPHA      = 1e-2
V_GRID     = np.logspace(np.log10(0.4), np.log10(25.0), 60)
N_LAGS     = 40
MAX_LAG_D  = 104.0
BED        = 1094.0
EFZ_TOP    = 600.0


def compute_perbin():
    def band_zarr(b):
        p = (ZARR_BASE if b == 'full'
             else ZARR_BASE.parent / f'{ZARR_BASE.stem}_{b}.zarr')
        return zarr.open(str(p), mode='r')

    print('Opening band zarrs...')
    zs = {b: band_zarr(b) for b in BANDS}
    R_full = np.array(zs['full']['Rcoarse']).flatten()
    t_days = np.array(zs['full']['time_days']).flatten()
    td = t_days - t_days[0]

    rec_mask = (R_full >= DEPTH_MIN) & (R_full <= DEPTH_MAX)
    rec_idx = np.where(rec_mask)[0]
    z_bins = R_full[rec_idx]
    n_bins = len(z_bins)
    print(f'  {n_bins} range bins over {DEPTH_MIN:.0f}-{DEPTH_MAX:.0f} m')

    print('Loading band slabs...')
    lo, hi = int(rec_idx[0]), int(rec_idx[-1]) + 1
    slabs = {b: np.asarray(zs[b]['raw_complex'][lo:hi, :],
                           dtype=np.complex64) for b in BANDS}

    l_ref, _ = lag_coherence_spatial(slabs['full'][:BINWIN], td,
                                     n_out=N_LAGS, max_lag_days=MAX_LAG_D)
    A = forward_matrix(V_GRID, l_ref,
                       [LAMBDAS_DEFAULT[b] for b in BANDS],
                       sigma_theta=SIGMA_THETA)

    print(f'Inverting {n_bins} bins (BINWIN={BINWIN}, no gate)...')
    P_bin = np.full((n_bins, len(V_GRID)), np.nan, dtype=np.float32)
    med = np.full(n_bins, np.nan)
    p16 = np.full(n_bins, np.nan)
    p84 = np.full(n_bins, np.nan)
    half = (BINWIN - 1) // 2
    t0 = time.time()
    for i in range(n_bins):
        a, b2 = max(0, i - half), min(n_bins, i + half + 1)
        g = np.concatenate([
            lag_coherence_spatial(slabs[bd][a:b2, :], td,
                                  n_out=N_LAGS, max_lag_days=MAX_LAG_D)[1]
            for bd in BANDS])
        P, _, _ = invert_pv(g, A, V_GRID, alpha=ALPHA)
        if P.sum() > 0:
            P_bin[i] = P
            s = pv_stats(V_GRID, P)
            med[i], p16[i], p84[i] = s['median'], s['p16'], s['p84']
        if (i + 1) % 2000 == 0 or i + 1 == n_bins:
            print(f'  {i+1}/{n_bins}  ({time.time()-t0:.0f}s)', flush=True)

    np.savez(CACHE, z_bins=z_bins, v_grid=V_GRID, P_bin=P_bin,
             med=med, p16=p16, p84=p84)
    print(f'Cached → {CACHE.relative_to(ROOT)}')
    return z_bins, P_bin, med, p16, p84


if CACHE.exists():
    print(f'Loading cache: {CACHE.relative_to(ROOT)}')
    d = dict(np.load(CACHE))
    z_bins, P_bin = d['z_bins'], d['P_bin']
    med, p16, p84 = d['med'], d['p16'], d['p84']
else:
    z_bins, P_bin, med, p16, p84 = compute_perbin()

# ── Figure ────────────────────────────────────────────────────────────
print('Plotting...')
fig = plt.figure(figsize=(ps.FIG_W_SINGLE, 4.6))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.45, 1.0], wspace=0.10,
              left=0.145, right=0.965, top=0.95, bottom=0.10)

v_ticks = [0.5, 1, 2, 5, 10, 20]
vmax_p = float(np.nanpercentile(P_bin, 95))


def run_median(x, win):
    """NaN-tolerant running median."""
    half = win // 2
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        seg = x[max(0, i - half): i + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out

# (a) high-resolution P(v_h, z) over the whole column ─────────────────
ax = fig.add_subplot(gs[0])
im = ax.pcolormesh(V_GRID, z_bins, P_bin, shading='nearest',
                   cmap=ps.CMAP_BLUE_SEQ, vmin=0, vmax=vmax_p,
                   rasterized=True)
ax.set_xscale('log')
ax.set_xlim(0.4, 25)
ax.set_xticks(v_ticks); ax.set_xticklabels([str(v) for v in v_ticks])
ax.set_ylim(DEPTH_MAX, DEPTH_MIN)
ps.zone_lines(ax, color='black')
ax.set_xlabel('$v_h$ (m yr$^{-1}$, nominal $\\sigma_\\theta$)', labelpad=2)
ax.set_ylabel('Depth (m)', labelpad=2)
ax.set_title('$P(v_h, z)$', pad=3)
ps.panel_label(ax, 'a')

# (b) median + 16-84 % profile ────────────────────────────────────────
ax = fig.add_subplot(gs[1])
ps.zone_shade(ax, efz=True, alpha_efz=0.08, dashed_lines=False)
ok = np.isfinite(med) & np.isfinite(p16) & np.isfinite(p84)
# Smooth the per-bin curves with a ~5 m running median for display
SM = 100  # bins ≈ 5 m
med_s = run_median(med, SM)
p16_s = run_median(p16, SM)
p84_s = run_median(p84, SM)
ax.fill_betweenx(z_bins[ok], p16_s[ok], p84_s[ok], color='#3b82f6',
                 alpha=0.25, linewidth=0, zorder=2, label='16–84 %')
ax.plot(med[ok], z_bins[ok], '-', color='#93c5fd', lw=0.4, alpha=0.7,
        zorder=3, label='per-bin median')
ax.plot(med_s[ok], z_bins[ok], '-', color='#1d4ed8', lw=1.2, zorder=4,
        label='median (5 m run.\\ med.)')
ps.zone_lines(ax, color='black')
ax.set_xscale('log')
ax.set_xlim(0.4, 25)
ax.set_xticks(v_ticks); ax.set_xticklabels([str(v) for v in v_ticks])
ax.set_ylim(DEPTH_MAX, DEPTH_MIN)
ax.set_yticklabels([])
ax.set_xlabel('$v_h$ (m yr$^{-1}$, nominal $\\sigma_\\theta$)', labelpad=2)
ax.set_title('Median + IQR', pad=3)
ax.legend(fontsize=5.0, loc='lower left')
ps.light_grid(ax)
ps.panel_label(ax, 'b')

for ext in ('pdf', 'png'):
    out = OUT_DIR / f'F_mdi_results.{ext}'
    fig.savefig(out, dpi=300 if ext == 'png' else None)
    print(f'Saved {out.relative_to(ROOT)}')
