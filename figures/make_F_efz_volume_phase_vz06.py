"""
make_F_efz_volume_phase.py

Phase-version of make_F_efz_volume.py.  Same layout, but the sim rows
show the carrier-detrended phase instead of amplitude.  The real-data
row is unchanged (amplitude | raw phase | detrended phase).

Output
------
  figs/F_efz_volume_phase_vz06.pdf
  figs/F_efz_volume_phase_vz06.png
"""

import sys
sys.path.insert(0, str((__import__('pathlib').Path(__file__).parent).resolve()))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import zarr

import paper_style as ps
from apres.forward_2d import (
    BackgroundSpeckle, Layer, simulate, carrier_detrend,
    LAMBDA_C, DEFAULT_PSF_SIGMA, DEFAULT_SIGMA_THETA,
)

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = ROOT / 'data' / 'ImageP2_python.zarr'
OUT_PDF   = ROOT / 'figs' / 'F_efz_volume_phase_vz06.pdf'
OUT_PNG   = ROOT / 'figs' / 'F_efz_volume_phase_vz06.png'

D_LO, D_HI = 800.0, 820.0
DEPTH_CENTRE = 0.5 * (D_LO + D_HI)
T_SPAN_DAYS  = 312.0

# Simulator settings
V_Z          = 0.6                    # m/yr, ice settling at this depth
V_H_STRONG   = 8.0                    # m/yr — "strong" horizontal shear case
SIGMA_THETA  = DEFAULT_SIGMA_THETA    # 0.22 rad (ApRES standard)
PSF_SIGMA    = DEFAULT_PSF_SIGMA
# Additive noise is set RELATIVE to each simulated signal so the panels
# sit at a realistic EFZ-like per-bin power SNR (the EFZ runs at only a
# few dB above the thermal floor; see the SNR profile in Fig. F5).
SNR_DB       = 6.0                    # target per-bin power SNR of the sims
# Match real data resolution exactly (Rcoarse ~53 mm spacing × ~380 bins
# in 20 m, and 1878 bursts ≈ 4 h cadence across the year).
N_T_SIM      = 1878
N_Z_SIM      = 380

FIG_W      = ps.FIG_W_DOUBLE
AMP_CMAP   = ps.CMAP_AMP
PHASE_CMAP = ps.CMAP_PHASE

# Layer scene: 3 modest-brightness layers clustered near the window
# centre (closer together than the full window span).
LAYER_DEPTHS = [807.5, 810.0, 812.5]
LAYER_A_PER_M = 0.05  # complex reflectivity strength per metre of arc

# Speckle scene used in row 2.
# IMPORTANT — iso-range shell filling: a scatterer at lateral offset x
# and depth z appears at slant range ≈ z + x²/2z, so the range window
# [800, 820] m receives contributions from scatterers as shallow as
# D_LO − x_hw²/(2·z0) ≈ 80 m above the window.  A slab confined to the
# window depths truncates the off-axis beam and artificially slows the
# beam-sweep decorrelation (sigma_eff < sigma_theta; see
# experiments/forward2d_depth_consistency.py).  We therefore extend the
# slab upward to fill the full shell and scale the scatterer count to
# preserve areal density.
X_HW             = max(30.0, 2.0 * DEPTH_CENTRE * DEFAULT_SIGMA_THETA)
SHELL_M          = X_HW ** 2 / (2.0 * D_LO)          # ≈ 80 m at 810 m
SPECKLE_Z_MIN    = D_LO - 2.0 - SHELL_M
SPECKLE_Z_MAX    = D_HI + 2.0
SPECKLE_N        = int(2000 * (SPECKLE_Z_MAX - SPECKLE_Z_MIN) / 24.0)
SPECKLE_AMP_RMS  = 0.005


def add_colorbar(ax, im, label='', n_ticks=4):
    """Wrap panel_colorbar and return the colourbar axis so callers can
    attach external annotations relative to it."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes('right', size='4%', pad=0.04)
    cb      = plt.colorbar(im, cax=cax)
    cb.set_label(label, labelpad=2)
    cb.locator = plt.MaxNLocator(n_ticks)
    cb.update_ticks()
    cb.outline.set_linewidth(0.4)
    cax.tick_params(width=0.4, labelsize=6.0)
    return cax


def panel_letter(ax, letter, pad_x=-0.02, pad_y=1.02):
    ps.panel_label(ax, letter, x=pad_x, y=pad_y)


# ── Real EFZ slice ────────────────────────────────────────────────────────────

print(f'Loading real EFZ slice from {ZARR_PATH}...')
zf = zarr.open(str(ZARR_PATH))
R_full   = np.array(zf['Rcoarse'])
t_days   = np.array(zf['time_days'])
mask     = (R_full >= D_LO) & (R_full <= D_HI)
idx      = np.where(mask)[0]
depths_r = R_full[mask]
raw_cpx  = np.array(zf['raw_complex'].oindex[idx, :])
range_im = np.array(zf['range_img'][idx, :], dtype=np.float32)
amp_real_db = 20 * np.log10(np.abs(range_im) + 1e-15)
phase_real  = np.angle(raw_cpx)
raw_cpx_det = carrier_detrend(raw_cpx, depths_r)
phase_real_det = np.angle(raw_cpx_det)
print(f'  real {amp_real_db.shape}')


# ── Forward sims ──────────────────────────────────────────────────────────────

def make_speckle_scene(v_h):
    """Volume-scatter scene: speckle filling the full iso-range shell."""
    return [BackgroundSpeckle(
        z_min=SPECKLE_Z_MIN, z_max=SPECKLE_Z_MAX,
        x_halfwidth=X_HW, n_scatterers=SPECKLE_N,
        amplitude_rms=SPECKLE_AMP_RMS,
        vx=v_h, vz=V_Z, seed=11,
    )]


def make_layer_scene(v_h):
    """Multi-layer scene: three horizontal layers + faint background."""
    scene = []
    for z in LAYER_DEPTHS:
        scene.append(Layer(
            vertices=np.array([[-250.0, z], [250.0, z]]),
            vx=v_h, vz=V_Z, A_per_m=LAYER_A_PER_M,
            sample_spacing=0.2,
        ))
    # A faint diffuse background so the layer panels are not unphysically
    # clean (real data has *some* speckle in addition to any layers).
    # Same iso-range-shell extent as the speckle scene.
    scene.append(BackgroundSpeckle(
        z_min=SPECKLE_Z_MIN, z_max=SPECKLE_Z_MAX,
        x_halfwidth=X_HW,
        n_scatterers=int(600 * (SPECKLE_Z_MAX - SPECKLE_Z_MIN) / 24.0),
        amplitude_rms=0.0025,
        vx=v_h, vz=V_Z, seed=23,
    ))
    return scene


# Eight simulations: volume speckle and multi-layer, each at four
# horizontal shear velocities, all shown as carrier-detrended phase
# and compared to the single real-EFZ detrended-phase panel.
V_H_LIST = [0.0, 2.0, 4.0, 8.0]

t_days_sim = np.linspace(0.0, T_SPAN_DAYS, N_T_SIM)
depths_sim = np.linspace(D_LO, D_HI, N_Z_SIM)
CACHE = ROOT / 'data' / 'F_efz_volume_phase_sims.npz'

sim_results = {}   # {(kind, v_h): phase_det}
if CACHE.exists():
    print(f'Loading sim cache: {CACHE}')
    _c = np.load(CACHE)
    for kind in ('speckle', 'layer'):
        for v_h in V_H_LIST:
            sim_results[(kind, v_h)] = _c[f'{kind}_{v_h:.0f}']
else:
    print('Running forward simulations...')
    scene_fns = {'speckle': make_speckle_scene, 'layer': make_layer_scene}
    for kind in ('speckle', 'layer'):
        for v_h in V_H_LIST:
            scene = scene_fns[kind](v_h)
            rng = np.random.default_rng(42)
            S = simulate(scene, t_days_sim, depths_sim,
                           lambdac=LAMBDA_C, psf_sigma=PSF_SIGMA,
                           sigma_theta=SIGMA_THETA,
                           noise_sigma=0.0, rng=rng)
            sig_rms = float(np.sqrt(np.mean(np.abs(S) ** 2)))
            n_sigma = sig_rms / np.sqrt(10.0 ** (SNR_DB / 10.0))
            S = S + n_sigma / np.sqrt(2.0) * (
                rng.standard_normal(S.shape)
                + 1j * rng.standard_normal(S.shape))
            sim_results[(kind, v_h)] = np.angle(
                carrier_detrend(S, depths_sim))
            print(f'  {kind:8s} v_h={v_h:4.1f} done')
    np.savez(CACHE, **{f'{k}_{v:.0f}': sim_results[(k, v)]
                       for k in ('speckle', 'layer') for v in V_H_LIST})
    print(f'Cached sims -> {CACHE}')

# ── Plot ──────────────────────────────────────────────────────────────────────
# Fig.-4-style grid of carrier-detrended-phase panels: the real EFZ on
# top (centred), then two rows of forward-model simulations — volume
# speckle and multi-layer, each at v_h = 0, 2, 4, 8 m/yr.  Every panel
# carries its own small title; one shared phase colour bar on the right.

print('Plotting...')
extent_real = [t_days[0], t_days[-1], D_HI, D_LO]
extent_sim = [t_days_sim[0], t_days_sim[-1], D_HI, D_LO]
d_mid = DEPTH_CENTRE
zoom_lo, zoom_hi = d_mid - 5, d_mid + 5
NCOL = len(V_H_LIST)

fig = plt.figure(figsize=(FIG_W, 5.0))
# 8 fine columns so the single real panel can be centred (width of one
# sim panel) above the 4-wide simulation rows.
gs = gridspec.GridSpec(
    3, 2 * NCOL, figure=fig,
    hspace=0.42, wspace=0.18,
    left=0.075, right=0.88, top=0.94, bottom=0.085,
)
letters = iter('abcdefghijklmnop')
last_im = None


def phase_panel(ax, ph, extent, title, is_left, is_bottom, letter):
    global last_im
    last_im = ax.imshow(ph, aspect='auto', cmap=PHASE_CMAP,
                        vmin=-np.pi, vmax=np.pi, extent=extent,
                        origin='upper')
    ax.set_ylim(zoom_hi, zoom_lo)
    ax.set_title(title, fontsize=6.5, pad=8)
    if is_left:
        ax.set_ylabel('Depth (m)', labelpad=2)
    else:
        ax.set_yticklabels([])
    if is_bottom:
        ax.set_xlabel('Time (days)', labelpad=2)
    else:
        ax.set_xticklabels([])
    panel_letter(ax, letter, pad_x=-0.04, pad_y=1.13)


# ── Real EFZ — amplitude + detrended phase, centred on the top row ──
ax_ra = fig.add_subplot(gs[0, 1:3])
vmin_r = float(np.percentile(amp_real_db, 2))
vmax_r = float(np.percentile(amp_real_db, 99))
ax_ra.imshow(amp_real_db, aspect='auto', cmap=AMP_CMAP,
             vmin=vmin_r, vmax=vmax_r, extent=extent_real, origin='upper')
ax_ra.set_ylim(zoom_hi, zoom_lo)
ax_ra.set_ylabel('Depth (m)', labelpad=2)
ax_ra.set_title('Real EFZ amp.', fontsize=6.5, pad=8)
ax_ra.set_xticklabels([])
panel_letter(ax_ra, next(letters), pad_x=-0.04, pad_y=1.13)
add_colorbar(ax_ra, ax_ra.images[0], 'dB')

ax_rp = fig.add_subplot(gs[0, 5:7])
phase_panel(ax_rp, phase_real_det, extent_real, 'Real EFZ phase',
            is_left=False, is_bottom=False, letter=next(letters))

# ── Simulation rows: speckle (row 1), multi-layer (row 2) ──
row_kind = [('speckle', 'Speckle'), ('layer', 'Layer')]
for ri, (kind, kname) in enumerate(row_kind, start=1):
    for ci, v_h in enumerate(V_H_LIST):
        ax = fig.add_subplot(gs[ri, 2 * ci:2 * ci + 2])
        phase_panel(ax, sim_results[(kind, v_h)], extent_sim,
                    f'{kname}, $v_h$={v_h:.0f}',
                    is_left=(ci == 0), is_bottom=(ri == 2),
                    letter=next(letters))

# Shared phase colour bar on the far right
cax = fig.add_axes([0.895, 0.12, 0.014, 0.66])
cb = fig.colorbar(last_im, cax=cax)
cb.set_label('Phase (rad)', fontsize=7.0, labelpad=3)
cb.set_ticks([-np.pi, 0, np.pi])
cb.set_ticklabels(['$-\\pi$', '0', '$\\pi$'])
cb.ax.tick_params(labelsize=6.5, width=0.4)
cb.outline.set_linewidth(0.4)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print(f'Saved {OUT_PDF}')
print(f'Saved {OUT_PNG}')
