"""
F_cwmlpr_synth — CW-MLPR synthetic validation figure (replaces the
real-data method figure).

Ground-truth scenes from the 2-D forward model (apres.forward_2d):
distributed volume speckle advecting at a known vertical velocity,
plus controlled additive complex noise.  The production CW-MLPR
estimator (apres.velocity.covariance_velocity) is compared against the
unwrap-and-fit baseline (per-bin OLS on the unwrapped phase, robust
median over bins).

  (a) RMSE vs per-burst SNR at fixed v_z: unwrap+OLS degrades
      catastrophically below moderate SNR (noise-induced cycle slips);
      CW-MLPR degrades gracefully.
  (b) Recovered vs true v_z at fixed SNR: CW-MLPR is unbiased across
      the physically relevant range.

Results cached to experiments/cwmlpr_synth_validation.npz.

Output:
  figs/F_cwmlpr_synth.pdf / .png
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import paper_style as ps
ps.apply_style()

from apres.forward_2d import BackgroundSpeckle, simulate as sim2d
from apres.velocity import covariance_velocity

OUT_DIR = ROOT / 'figs'
THESIS_FIGS = ROOT / 'figs'
# Thesis mode (pass "thesis") lays the two panels side by side and writes
# to the thesis figure directory; the paper default keeps the stacked,
# single-column layout.
THESIS = 'thesis' in sys.argv[1:]
SAVE_DIR = THESIS_FIGS if THESIS else OUT_DIR
CACHE = ROOT / 'data' / 'cwmlpr_synth_validation.npz'

# ── Config ────────────────────────────────────────────────────────────
Z0 = 805.0                     # scene depth (m) — representative EFZ
D_HALF = 2.0                   # half depth window (m)
N_Z, N_T = 100, 300
T_SPAN = 312.0                 # days (matches deployment)
LAMBDAC = 0.5608
N_LAGS = 8                     # production CW-MLPR setting

# Extend to very low SNR so the CW-MLPR breakdown (where the coherent
# integration gain N_t*SNR approaches unity, per-burst SNR ~ 1/N_t) is
# visible, not just the OLS breakdown.
SNRS = np.array([0.002, 0.004, 0.008, 0.016, 0.032, 0.063, 0.125,
                 0.25, 0.5, 1, 2, 4, 8, 16, 32])
V_SNR_SWEEP = 0.6              # m/yr — typical EFZ velocity
N_TRIAL_A = 24
DECORR_DAYS = 120.0           # EFZ-like scatterer coherence time

V_TRUES = np.array([0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 5.0])
SNR_B = 4.0
N_TRIAL_B = 16

depths = np.linspace(Z0 - D_HALF, Z0 + D_HALF, N_Z)
times = np.linspace(0.0, T_SPAN, N_T)
K = LAMBDAC / (4.0 * np.pi)


def scene(v_z, seed):
    return [BackgroundSpeckle(z_min=Z0 - 12, z_max=Z0 + 12,
                              x_halfwidth=60.0, n_scatterers=800,
                              amplitude_rms=0.02, vx=0.0, vz=v_z,
                              seed=seed)]


def make_window(v_z, snr_pow, seed, decorr_days=None):
    """Forward-model speckle scene + complex noise at the given SNR.

    decorr_days: if set, multiply each depth bin by an independent
    complex random-walk phase screen with coherence time ``decorr_days``
    — an EFZ-like temporal decorrelation of the scatterer field on top
    of the deterministic beam-sweep, i.e. the Gauss-Markov-violating
    regime the EFZ actually lives in.
    """
    rng = np.random.default_rng(seed)
    S = sim2d(scene(v_z, seed), times, depths, lambdac=LAMBDAC,
              noise_sigma=0.0, rng=rng)
    if decorr_days is not None and decorr_days > 0:
        dt = np.diff(times, prepend=times[0])              # days
        step_std = np.sqrt(dt / decorr_days)               # rad per step
        dphi = rng.standard_normal(S.shape) * step_std[None, :]
        S = S * np.exp(1j * np.cumsum(dphi, axis=1))
    sig_rms = np.sqrt(np.mean(np.abs(S) ** 2))
    noise = (rng.standard_normal(S.shape)
             + 1j * rng.standard_normal(S.shape)) / np.sqrt(2.0)
    return S + (sig_rms / np.sqrt(snr_pow)) * noise


def est_cwmlpr(S):
    """Production CW-MLPR (M6 multi-lag covariance + robust median)."""
    return covariance_velocity(S, times, LAMBDAC, n_lags=N_LAGS)['best_v']


def est_unwrap_ols(S):
    """Unwrap-and-fit baseline: per-bin OLS on unwrapped phase, median."""
    t_yr = times / 365.25
    A = np.vstack([t_yr, np.ones_like(t_yr)]).T
    vs = []
    for row in S:
        phi = np.unwrap(np.angle(row))
        coef, *_ = np.linalg.lstsq(A, phi, rcond=None)
        vs.append(coef[0] * K)
    return float(np.median(vs))


def compute():
    rmse_cw = np.zeros(len(SNRS)); rmse_ols = np.zeros(len(SNRS))
    rmse_cw_dc = np.zeros(len(SNRS))
    t0 = time.time()
    print('(a) SNR sweep...')
    for i, snr in enumerate(SNRS):
        e_cw, e_ols, e_cw_dc = [], [], []
        for tr in range(N_TRIAL_A):
            S = make_window(V_SNR_SWEEP, snr, seed=100 * i + tr)
            e_cw.append(est_cwmlpr(S) - V_SNR_SWEEP)
            e_ols.append(est_unwrap_ols(S) - V_SNR_SWEEP)
            Sd = make_window(V_SNR_SWEEP, snr, seed=100 * i + tr,
                             decorr_days=DECORR_DAYS)
            e_cw_dc.append(est_cwmlpr(Sd) - V_SNR_SWEEP)
        rmse_cw[i] = np.sqrt(np.nanmean(np.square(e_cw)))
        rmse_ols[i] = np.sqrt(np.nanmean(np.square(e_ols)))
        rmse_cw_dc[i] = np.sqrt(np.nanmean(np.square(e_cw_dc)))
        print(f'  SNR={snr:>5.1f}: CW-MLPR {rmse_cw[i]:.3f}, '
              f'OLS {rmse_ols[i]:.3f}, CW+decorr {rmse_cw_dc[i]:.3f}  '
              f'({time.time()-t0:.0f}s)', flush=True)

    print('(b) v_z sweep...')
    rec_cw = np.zeros((len(V_TRUES), N_TRIAL_B))
    rec_ols = np.zeros((len(V_TRUES), N_TRIAL_B))
    for i, vt in enumerate(V_TRUES):
        for tr in range(N_TRIAL_B):
            S = make_window(vt, SNR_B, seed=9000 + 100 * i + tr)
            rec_cw[i, tr] = est_cwmlpr(S)
            rec_ols[i, tr] = est_unwrap_ols(S)
        print(f'  v={vt:>4.1f}: CW-MLPR {np.nanmean(rec_cw[i]):.2f}, '
              f'OLS {np.nanmean(rec_ols[i]):.2f}  '
              f'({time.time()-t0:.0f}s)', flush=True)

    np.savez(CACHE, snrs=SNRS, rmse_cw=rmse_cw, rmse_ols=rmse_ols,
             rmse_cw_dc=rmse_cw_dc,
             v_trues=V_TRUES, rec_cw=rec_cw, rec_ols=rec_ols)
    print(f'Cached → {CACHE.relative_to(ROOT)}')
    return rmse_cw, rmse_ols, rmse_cw_dc, rec_cw, rec_ols


if CACHE.exists() and 'rmse_cw_dc' in np.load(CACHE):
    print(f'Loading cache: {CACHE.relative_to(ROOT)}')
    d = dict(np.load(CACHE))
    rmse_cw, rmse_ols, rmse_cw_dc = d['rmse_cw'], d['rmse_ols'], d['rmse_cw_dc']
    rec_cw, rec_ols = d['rec_cw'], d['rec_ols']
else:
    rmse_cw, rmse_ols, rmse_cw_dc, rec_cw, rec_ols = compute()

# ── Figure ────────────────────────────────────────────────────────────
COL_CW, COL_OLS = '#1d4ed8', '#6b7280'
if THESIS:                              # side-by-side (a | b), full text width
    fig = plt.figure(figsize=(ps.FIG_W_DOUBLE, 2.8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28,
                  left=0.08, right=0.975, top=0.92, bottom=0.17)
else:                                   # stacked, single column (paper)
    fig = plt.figure(figsize=(ps.FIG_W_SINGLE, 4.0))
    gs = GridSpec(2, 1, figure=fig, hspace=0.48,
                  left=0.16, right=0.96, top=0.95, bottom=0.10)

# (a) RMSE vs SNR ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0])
ax.loglog(SNRS, rmse_ols, 's--', color=COL_OLS, lw=1.2, ms=3.5,
          label='unwrap + OLS')
ax.loglog(SNRS, rmse_cw, 'o-', color=COL_CW, lw=1.4, ms=3.5,
          label='CW-MLPR')
ax.loglog(SNRS, rmse_cw_dc, '^:', color='#0891b2', lw=1.2, ms=3.5,
          label=f'CW-MLPR, decorr. {DECORR_DAYS:.0f} d')
ax.set_xlabel('Per-burst SNR (power)', labelpad=2)
ax.set_ylabel('RMSE (m yr$^{-1}$)', labelpad=2)
xt = [0.002, 0.01, 0.1, 1, 10]
ax.set_xticks(xt)
ax.set_xticklabels([f'{s:g}' for s in xt])
ax.legend(fontsize=5.8, loc='lower left')
ps.light_grid(ax)
ps.panel_label(ax, 'a')

# (b) recovered vs true ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1])
lim = V_TRUES[-1] * 1.12
ax.plot([0, lim], [0, lim], ':', color='#10b981', lw=1.1, label='1:1')
ax.errorbar(V_TRUES, np.nanmean(rec_ols, axis=1),
            yerr=np.nanstd(rec_ols, axis=1), fmt='s--', color=COL_OLS,
            lw=1.0, ms=3.2, capsize=2, label='unwrap + OLS')
ax.errorbar(V_TRUES, np.nanmean(rec_cw, axis=1),
            yerr=np.nanstd(rec_cw, axis=1), fmt='o-', color=COL_CW,
            lw=1.3, ms=3.2, capsize=2, label='CW-MLPR')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('True $v_z$ (m yr$^{-1}$)', labelpad=2)
ax.set_ylabel('Recovered $v_z$ (m yr$^{-1}$)', labelpad=2)
ax.legend(fontsize=5.8, loc='upper left')
ps.light_grid(ax)
ps.panel_label(ax, 'b')

SAVE_DIR.mkdir(parents=True, exist_ok=True)
for ext in ('pdf', 'png'):
    out = SAVE_DIR / f'F_cwmlpr_synth.{ext}'
    fig.savefig(out, dpi=300 if ext == 'png' else None)
    print(f'Saved {out.relative_to(ROOT)}')
