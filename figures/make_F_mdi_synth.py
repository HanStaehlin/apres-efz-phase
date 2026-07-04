"""
F_mdi_synth — MDI synthetic reconstruction figure (replaces the
real-data method figure).

Ground-truth scenes from the 2-D forward model (apres.forward_2d):

  (a) Two-population mixture: volume speckle advecting at two known
      horizontal velocities in equal proportion.  MDI recovers a
      bimodal P(v_h) at the true velocities, whereas the conventional
      single-decay scalar estimator collapses to one effective
      velocity between the modes.
  (b) Regularisation sensitivity: the bimodal recovery persists across
      a decade of the Tikhonov weight alpha either side of the L-curve
      value, broadening but never splitting or merging.

Heavy compute (empirical kernel = one 3-band simulation per grid
velocity) cached to data/mdi_synth_validation.npz.

Output:
  figs/F_mdi_synth.pdf / .png
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
from scipy.optimize import nnls

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import paper_style as ps
ps.apply_style()

from apres.forward_2d import BackgroundSpeckle, simulate as sim2d
from apres.decorrelation_inversion import (
    forward_matrix, pv_stats, second_diff_matrix, LAMBDAS_DEFAULT,
)
from apres.velocity import lag_coherence_spatial
SIGMA_THETA = 0.22   # antenna beam angular std (rad), ~30 deg 3-dB beamwidth

OUT_DIR = ROOT / 'figs'
THESIS_FIGS = ROOT / 'figs'
# Thesis mode (pass "thesis") lays the two panels side by side and writes
# to the thesis figure directory; the paper default keeps the stacked,
# single-column layout.
THESIS = 'thesis' in sys.argv[1:]
SAVE_DIR = THESIS_FIGS if THESIS else OUT_DIR
CACHE = ROOT / 'data' / 'mdi_synth_validation.npz'

# ── Config ────────────────────────────────────────────────────────────
# Z0 = 200 m: BEAM-LIMITED regime.  At deeper Z0 the finite scatterer
# slab truncates the iso-range shell (range curvature: scatterers at
# lateral offset x sit at range ≈ z + x²/2z), making sigma_eff
# config-dominated (see experiments/forward2d_depth_consistency.py).
# At 200 m the window-truncation angle θ_w = sqrt(2·Δz/z) ≈ 0.40 rad
# exceeds σ_θ = 0.22, so the beam itself sets the decorrelation and
# the synthetic test is free of the scatterer-distribution artifact.
Z0, D_WIN, N_PTS, T_SPAN = 200.0, 4.0, 300, 365.0
N_LAGS, MAX_LAG = 40, 104.0
V_GRID = np.logspace(np.log10(0.4), np.log10(25.0), 40)
ALPHA = 1e-2
ALPHA_MIX = 2e-2               # display-panel smoothing (denoised operator)
BANDS = ('low', 'full', 'high')
COL_SEED = 12                  # seeds averaged per empirical-kernel column
V_TEST = np.array([2.0, 3.0, 5.0, 8.0, 12.0])
TEST_SEED = 4
MIX_SEED = 6
V_MIX = (3.0, 8.0)             # two-population mixture truth

depths = np.linspace(Z0 - D_WIN, Z0 + D_WIN, N_PTS)
times = np.linspace(0.0, T_SPAN, N_PTS)
xhw = max(30.0, 2.0 * Z0 * SIGMA_THETA)


def speckle(vx, seed, amplitude_rms=0.02):
    return BackgroundSpeckle(z_min=Z0 - 12, z_max=Z0 + 12, x_halfwidth=xhw,
                             n_scatterers=2000, amplitude_rms=amplitude_rms,
                             vx=vx, vz=0.0, seed=7 + seed)


def coh_stack(scene, seed):
    """3-band stacked |γ(τ)| (spatial estimator)."""
    g = []
    for b in BANDS:
        S = sim2d(scene, times, depths, lambdac=LAMBDAS_DEFAULT[b],
                  noise_sigma=0.0, rng=np.random.default_rng(seed))
        g.append(lag_coherence_spatial(S, times, n_out=N_LAGS,
                                       max_lag_days=MAX_LAG)[1])
    return np.concatenate(g)


def invert_with(A, g, alpha=ALPHA):
    """Tikhonov-regularised NNLS (same core as invert_pv)."""
    L = second_diff_matrix(len(V_GRID))
    A_aug = np.vstack([A, np.sqrt(alpha) * L])
    b_aug = np.concatenate([g, np.zeros(L.shape[0])])
    P, _ = nnls(A_aug, b_aug, maxiter=10000)
    v = np.asarray(V_GRID, float); dv = np.diff(v)
    mass = float(np.sum(0.5 * (P[:-1] + P[1:]) * dv))
    return P / mass if mass > 0 else P


def compute():
    print(f'Building empirical kernel ({len(V_GRID)} columns)...')
    t0 = time.time()
    cols = []
    for j, vj in enumerate(V_GRID):
        c = [coh_stack([speckle(vj, 1000 + j * 7 + s)], 1000 + j * 7 + s)
             for s in range(COL_SEED)]
        cols.append(np.mean(c, axis=0))
        if (j + 1) % 10 == 0:
            print(f'  {j+1}/{len(V_GRID)}  ({time.time()-t0:.0f}s)',
                  flush=True)
    A_emp = np.array(cols).T

    lags = lag_coherence_spatial(np.ones((4, N_PTS), dtype=complex), times,
                                 n_out=N_LAGS, max_lag_days=MAX_LAG)[0]
    A_ana = forward_matrix(V_GRID, lags,
                           [LAMBDAS_DEFAULT[b] for b in BANDS],
                           sigma_theta=SIGMA_THETA)

    print('Single-velocity test scenes...')
    rec_emp = np.full((len(V_TEST), TEST_SEED), np.nan)
    rec_ana = np.full((len(V_TEST), TEST_SEED), np.nan)
    for vi, vt in enumerate(V_TEST):
        for s in range(TEST_SEED):
            g = coh_stack([speckle(vt, 5000 + s)], 5000 + s)
            Pe, Pa = invert_with(A_emp, g), invert_with(A_ana, g)
            if Pe.sum() > 0:
                rec_emp[vi, s] = pv_stats(V_GRID, Pe)['median']
            if Pa.sum() > 0:
                rec_ana[vi, s] = pv_stats(V_GRID, Pa)['median']
        print(f'  v={vt:>5.1f}  emp r={np.nanmean(rec_emp[vi])/vt:.2f}  '
              f'ana r={np.nanmean(rec_ana[vi])/vt:.2f}', flush=True)

    print('Two-population mixture scene...')
    g_mix = np.mean([coh_stack([speckle(V_MIX[0], 41 + 2 * s),
                                speckle(V_MIX[1], 43 + 2 * s)],
                               6000 + s) for s in range(MIX_SEED)],
                    axis=0)
    P_mix_emp = invert_with(A_emp, g_mix, alpha=ALPHA_MIX)
    P_mix_ana = invert_with(A_ana, g_mix, alpha=ALPHA_MIX)

    # Conventional scalar estimator: fit ONE Gaussian decay (a single
    # effective velocity, the Dirac-delta assumption) to the same
    # multi-band mixture coherence.  Columns of the analytic operator are
    # exactly these single-v decays, so the best single-v fit minimises
    # ||A_ana[:, j] - g_mix|| over the grid; refine with a local
    # parabolic interpolation in log-v.
    resid = np.linalg.norm(A_ana - g_mix[:, None], axis=0)
    jbest = int(np.argmin(resid))
    if 0 < jbest < len(V_GRID) - 1:
        lv = np.log(V_GRID[jbest - 1:jbest + 2])
        y = resid[jbest - 1:jbest + 2]
        denom = (y[0] - 2 * y[1] + y[2])
        shift = 0.5 * (y[0] - y[2]) / denom if denom != 0 else 0.0
        v_eff = float(np.exp(lv[1] + shift * (lv[2] - lv[1])))
    else:
        v_eff = float(V_GRID[jbest])

    np.savez(CACHE, v_grid=V_GRID, v_test=V_TEST,
             rec_emp=rec_emp, rec_ana=rec_ana,
             P_mix_emp=P_mix_emp, P_mix_ana=P_mix_ana, v_eff=v_eff)
    print(f'Cached → {CACHE.relative_to(ROOT)}')
    print(f'  single-decay scalar fit: v_eff = {v_eff:.2f} m/yr '
          f'(truth {V_MIX[0]:.0f} & {V_MIX[1]:.0f})')
    return rec_emp, rec_ana, P_mix_emp, P_mix_ana, v_eff


if CACHE.exists():
    print(f'Loading cache: {CACHE.relative_to(ROOT)}')
    d = dict(np.load(CACHE))
    rec_emp, rec_ana = d['rec_emp'], d['rec_ana']
    P_mix_emp, P_mix_ana = d['P_mix_emp'], d['P_mix_ana']
    v_eff = float(d['v_eff'])
else:
    rec_emp, rec_ana, P_mix_emp, P_mix_ana, v_eff = compute()


# ── Regularisation-sensitivity data from the cached
#    make_F_mdi_robustness.py run ─────────────────────────────────────
ROB = ROOT / 'data' / 'mdi_robustness.npz'
dR = dict(np.load(ROB))
alpha_sweep, P_alpha = dR['alpha_sweep'], dR['P_alpha']

# ── Figure (mixture recovery | regularisation sensitivity) ────────────
COL_BEAM = '#1d4ed8'
v_ticks = [0.5, 1, 2, 5, 10, 20]
if THESIS:                              # side-by-side (a | b), full text width
    fig = plt.figure(figsize=(ps.FIG_W_DOUBLE, 2.8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.26,
                  left=0.08, right=0.975, top=0.92, bottom=0.17)
else:                                   # stacked, single column (paper)
    fig = plt.figure(figsize=(ps.FIG_W_SINGLE, 4.2))
    gs = GridSpec(2, 1, figure=fig, hspace=0.42,
                  left=0.155, right=0.96, top=0.95, bottom=0.11)

# (a) mixture recovery vs the conventional scalar estimator ───────────
ax = fig.add_subplot(gs[0])
for vt in V_MIX:
    ax.axvline(vt, color='#10b981', lw=1.1, ls=':',
               label='truth' if vt == V_MIX[0] else None)
ax.fill_between(V_GRID, 0, P_mix_ana, color=COL_BEAM, alpha=0.18, lw=0)
ax.plot(V_GRID, P_mix_ana, '-', color=COL_BEAM, lw=1.3, label='MDI $P(v_h)$')
ax.axvline(v_eff, color='#6b7280', lw=1.3, ls='-.',
           label=f'single-decay fit ({v_eff:.1f})')
ax.set_xscale('log')
ax.set_xlim(V_GRID[0], V_GRID[-1])
ax.set_xticks(v_ticks); ax.set_xticklabels([str(v) for v in v_ticks])
ax.set_ylim(bottom=0)
ax.set_xlabel('$v_h$ (m yr$^{-1}$, log)', labelpad=2)
ax.set_ylabel('$P(v_h)$', labelpad=2)
ax.legend(fontsize=5.2, loc='upper right')
ps.light_grid(ax)
ps.panel_label(ax, 'a')

# (b) regularisation (Tikhonov-alpha) sensitivity of the bimodal recovery
ax = fig.add_subplot(gs[1])
for vt in V_MIX:
    ax.axvline(vt, color='#10b981', lw=1.0, ls=':',
               label='truth' if vt == V_MIX[0] else None)
cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(alpha_sweep)))
for a, P, col in zip(alpha_sweep, P_alpha, cols):
    ax.plot(V_GRID, P, '-', color=col, lw=1.1, label=fr'$\alpha$={a:g}')
ax.set_xscale('log')
ax.set_xlim(V_GRID[0], V_GRID[-1])
ax.set_xticks(v_ticks); ax.set_xticklabels([str(v) for v in v_ticks])
ax.set_ylim(bottom=0)
ax.set_xlabel('$v_h$ (m yr$^{-1}$, log)', labelpad=2)
ax.set_ylabel('$P(v_h)$', labelpad=2)
ax.legend(fontsize=4.8, loc='upper right', ncol=2)
ps.light_grid(ax)
ps.panel_label(ax, 'b')

SAVE_DIR.mkdir(parents=True, exist_ok=True)
for ext in ('pdf', 'png'):
    out = SAVE_DIR / f'F_mdi_synth.{ext}'
    fig.savefig(out, dpi=300 if ext == 'png' else None)
    print(f'Saved {out.relative_to(ROOT)}')
