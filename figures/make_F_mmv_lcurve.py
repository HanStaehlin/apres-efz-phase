"""
make_F_mmv_lcurve.py

Re-render the F_mmv paper figure using a band-specific α picked by
the L-curve chord-distance corner (cs_mmv_lcurve_alphas.npz):

  F1 (shallow):   α = 0.39
  F4 (EFZ):       α = 0.39
  F3 (bed/lake):  α = 0.29
  F2 (below bed): α = 0.39

Layout identical to make_F_mmv.py / make_F_mmv_a07.py.

Output
------
  figs/F_mmv_lcurve.pdf
  figs/F_mmv_lcurve.png
"""

import sys
sys.path.insert(0, str((__import__('pathlib').Path(__file__).parent).resolve()))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, ConnectionPatch
from pathlib import Path
import zarr

import paper_style as ps

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

NPZ_F_BANDS = ROOT / 'data' / 'cs_mmv_F_bands_lcurve.npz'
ZARR_PATH   = ROOT / 'data' / 'ImageP2_python.zarr'

OUT_PDF = ROOT / 'figs' / 'F_mmv_lcurve.pdf'
OUT_PNG = ROOT / 'figs' / 'F_mmv_lcurve.png'

WINDOWS = [
    ('F1', 100,  120,  'Shallow layers'),
    ('F4', 800,  820,  'Echo-Free Zone'),
    ('F3', 1085, 1105, 'Bed / lake'),
    ('F2', 1790, 1810, 'Below bed (noise)'),
]

FIG_W    = ps.FIG_W_SINGLE
AMP_CMAP = ps.CMAP_AMP


def add_colorbar(ax, im, label='', n_ticks=4):
    ps.panel_colorbar(ax, im, label=label, n_ticks=n_ticks)


def panel_letter(ax, letter, pad_x=-0.02, pad_y=1.02):
    ps.panel_label(ax, letter, x=pad_x, y=pad_y)


def fade_alpha(values, vmin, vmax, min_alpha=0.12, gamma=1.0):
    """Per-point alpha that fades weak components toward transparent,
    matching the CLEAN overlay (fig:clean_overlay): colour still encodes
    atom amplitude via the normal c=/cmap=/vmin=/vmax= scatter kwargs, but
    weak atoms no longer stand out as strongly as strong ones."""
    norm = np.clip((np.asarray(values) - vmin) / (vmax - vmin), 0.0, 1.0)
    return min_alpha + (1.0 - min_alpha) * norm ** gamma


# ── Load atoms ────────────────────────────────────────────────────────────────

print(f'Loading {NPZ_F_BANDS}...')
d = np.load(NPZ_F_BANDS, allow_pickle=True)
window_starts = d['window_starts']
t_days_npz    = d['t_days']
n_bursts      = int(d['n_bursts_window'])
alpha_per_band = {tag: float(d[f'{tag}_alpha']) for tag, _, _, _ in WINDOWS}
print(f'  α per band: {alpha_per_band}, {len(window_starts)} windows')

band_data = {}
for tag, d_lo, d_hi, _ in WINDOWS:
    depths_per_w = d[f'{tag}_depths']
    amps_per_w   = d[f'{tag}_amps']
    raw_t, raw_d, raw_db = [], [], []
    for wi, s in enumerate(window_starts):
        dep = np.asarray(depths_per_w[wi], dtype=float)
        if dep.size == 0:
            continue
        amp = amps_per_w[wi]
        nrm = np.linalg.norm(amp, axis=1)
        t_c = t_days_npz[min(int(s) + n_bursts // 2, len(t_days_npz) - 1)]
        for k in range(len(dep)):
            raw_t.append(t_c)
            raw_d.append(float(dep[k]))
            raw_db.append(20 * np.log10(float(nrm[k]) + 1e-12))
    band_data[tag] = (np.asarray(raw_t), np.asarray(raw_d), np.asarray(raw_db))
    print(f'  {tag} ({d_lo}-{d_hi} m, α={alpha_per_band[tag]:.2f}): '
          f'{len(raw_t)} atoms')

all_db = np.concatenate([v[2] for v in band_data.values()])
amp_vmin = float(np.percentile(all_db, 5))
amp_vmax = float(np.percentile(all_db, 95))

# ── Load reference echogram ───────────────────────────────────────────────────

print('Loading echogram...')
zf = zarr.open(str(ZARR_PATH))
R_full = np.array(zf['Rcoarse'])
t_days = np.array(zf['time_days'])

D_MAX_ECHO = 2000.0
mask_echo = R_full <= D_MAX_ECHO
idx_echo  = np.where(mask_echo)[0]
t_step    = 4
amp_echo  = np.array(zf['range_img'][idx_echo[0]:idx_echo[-1] + 1, ::t_step],
                       dtype=np.float32)
amp_echo_db = 20.0 * np.log10(np.abs(amp_echo) + 1e-15)
depths_echo = R_full[idx_echo]
time_sub    = t_days[::t_step]
vmin_echo   = float(np.percentile(amp_echo_db, 5))
vmax_echo   = float(np.percentile(amp_echo_db, 99))

# ── Plot ─────────────────────────────────────────────────────────────
# Two layouts: the paper version (single column, 4 stacked atom panels)
# and the thesis version (full text width: an overview echogram panel
# plus the four windows, which the thesis page has room for).  Pass
# "thesis" on the command line for the latter.

THESIS = 'thesis' in sys.argv[1:]
SHORT = {'F1': 'Layered', 'F4': 'EFZ', 'F3': 'Bed / lake', 'F2': 'Noise'}


def _window_panel(ax, tag, d_min, d_max, letter, *, marker_s, clean=False,
                  box_col=None, header=False):
    """Faded echogram strip + MMV atoms for one depth window.

    clean=True uses the Fref-style layout: a small corner descriptor and a
    panel letter instead of a long per-panel title, with a shared column
    header on the first row.
    """
    mask_z = (depths_echo >= d_min - 0.5) & (depths_echo <= d_max + 0.5)
    echo_slice = amp_echo_db[mask_z, :]
    if echo_slice.size > 0:
        z_strip = depths_echo[mask_z]
        ax.imshow(echo_slice, aspect='auto', origin='upper',
                  extent=[time_sub[0], time_sub[-1], z_strip[-1], z_strip[0]],
                  cmap=AMP_CMAP, vmin=vmin_echo, vmax=vmax_echo,
                  alpha=0.45, interpolation='nearest', zorder=1)
    t_arr, d_arr, db_arr = band_data[tag]
    sc = None
    if len(t_arr) > 0:
        sc = ax.scatter(t_arr, d_arr, c=db_arr, cmap=AMP_CMAP,
                        vmin=amp_vmin, vmax=amp_vmax,
                        alpha=fade_alpha(db_arr, amp_vmin, amp_vmax),
                        s=marker_s, linewidths=0, zorder=3)
    ax.set_xlim(t_days[0], t_days[-1])
    ax.set_ylim(d_max, d_min)
    ax.set_ylabel('Depth (m)', labelpad=1, fontsize=6.5)
    if clean:
        # depth axis on the right so it does not collide with the echogram
        # column and the box-to-panel connectors on the left
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.text(0.975, 0.90,
                f'{SHORT[tag]}  $\\alpha$={alpha_per_band[tag]:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=5.8,
                color=box_col or '#1e3a8a', fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none',
                          alpha=0.75))
        if header:
            ax.set_title('MMV active atoms', pad=3, fontsize=7.5,
                         fontweight='bold')
        ps.panel_label(ax, letter)
    else:
        ax.set_title(
            f'({letter}) {SHORT[tag]}: {d_min}–{d_max} m, '
            f'$\\alpha$={alpha_per_band[tag]:.2f}',
            loc='left', fontsize=6.5, pad=3, color='#1e3a8a',
            fontweight='bold')
    return sc


print(f'Plotting ({"thesis full-page" if THESIS else "paper single-column"})...')

if THESIS:
    # Exact layout of Fig. (Fref): full echogram (left, all rows) + a 4x2
    # grid of zoom panels.  The amplitude/phase zoom panels of Fref are
    # reused verbatim, with the CS (MMV) active atoms overlaid on each.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    OUT_PDF = ROOT / 'figs' / 'F_mmv_lcurve.pdf'
    OUT_PNG = ROOT / 'figs' / 'F_mmv_lcurve.png'
    PHASE_CMAP = ps.CMAP_PHASE
    range_img_z   = zf['range_img']
    raw_complex_z = zf['raw_complex']
    ZONES = [(0, 600, ps.COL_LAYERS_FILL), (600, 1094, ps.COL_EFZ_FILL),
             (1094, 1200, ps.COL_BED_FILL)]
    # top-to-bottom: shallow, EFZ, bed, noise (same windows as the CS atoms)
    REF = [('F1', 100, 120), ('F4', 800, 820),
           ('F3', 1085, 1105), ('F2', 1790, 1810)]
    # Overlay the MMV atoms coloured by amplitude (dB) on the RdBu_r scale,
    # with weak components faded toward transparent -- the same encoding as
    # the CLEAN overlay (fig:clean_overlay) so the two read consistently.
    ATOM_KW = dict(cmap=AMP_CMAP, vmin=amp_vmin, vmax=amp_vmax, s=6,
                   linewidths=0, zorder=8)

    def _slice(d_min, d_max):
        idx = np.where((R_full >= d_min) & (R_full <= d_max))[0]
        amp = np.array(range_img_z[idx[0]:idx[-1] + 1, :], dtype=np.float32)
        cpx = np.array(raw_complex_z[idx[0]:idx[-1] + 1, :])
        return 20.0 * np.log10(np.abs(amp) + 1e-15), np.angle(cpx)

    # ── GridSpec: left column spans all rows (identical to Fref) ─────
    fig = plt.figure(figsize=(ps.FIG_W_DOUBLE, 7.0))
    gs = gridspec.GridSpec(
        4, 3, figure=fig, width_ratios=[1.0, 0.9, 0.9],
        hspace=0.10, wspace=0.30,
        left=0.07, right=0.97, top=0.96, bottom=0.06)

    # (a) full echogram (spans all 4 rows)
    ax_echo = fig.add_subplot(gs[:, 0])
    ax_echo.imshow(amp_echo_db, aspect='auto', cmap=AMP_CMAP,
                   vmin=vmin_echo, vmax=vmax_echo,
                   extent=[time_sub[0], time_sub[-1], D_MAX_ECHO, depths_echo[0]],
                   origin='upper')
    for z_top, z_bot, fc in ZONES:
        ax_echo.axhspan(z_top, min(z_bot, D_MAX_ECHO), color=fc, alpha=0.15,
                        zorder=2)
        if z_bot <= D_MAX_ECHO:
            ax_echo.axhline(z_bot, color='white', lw=0.7, ls='--', zorder=3)
    ax_echo.set_ylim(D_MAX_ECHO, 0)
    ax_echo.set_xlim(time_sub[0], time_sub[-1])
    ax_echo.set_xlabel('Time (days)', labelpad=2)
    ax_echo.set_ylabel('Depth (m)', labelpad=2)
    ax_echo.set_title('Full echogram', pad=3, fontsize=7.5, fontweight='bold')
    ps.panel_label(ax_echo, 'a')

    letters = iter('bcdefghi')
    for row, (tag, d_min, d_max) in enumerate(REF):
        amp_db, phase = _slice(d_min, d_max)
        vmin_amp = np.percentile(amp_db, 2)
        vmax_amp = np.percentile(amp_db, 98)
        extent = [t_days[0], t_days[-1], d_max, d_min]
        d_mid = 0.5 * (d_min + d_max)
        zoom_lo, zoom_hi = d_mid - 5, d_mid + 5
        t_arr, d_arr, db_arr = band_data[tag]

        # zoom box + margin letter on the full echogram
        ax_echo.add_patch(Rectangle(
            (time_sub[0], d_min), time_sub[-1] - time_sub[0], d_max - d_min,
            linewidth=1.8, edgecolor='white', facecolor='none', zorder=10))
        ax_echo.text(-0.02, d_mid, chr(ord('b') + row * 2),
                     transform=ax_echo.get_yaxis_transform(), ha='right',
                     va='center', fontsize=7, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.15', fc='black',
                               alpha=0.6, lw=0))

        # ── CS over amplitude ───────────────────────────────────────
        ax_amp = fig.add_subplot(gs[row, 1])
        im = ax_amp.imshow(amp_db, aspect='auto', cmap=AMP_CMAP,
                           vmin=vmin_amp, vmax=vmax_amp, extent=extent,
                           origin='upper')
        add_colorbar(ax_amp, im, 'dB')
        if len(t_arr):
            ax_amp.scatter(t_arr, d_arr, c=db_arr,
                          alpha=fade_alpha(db_arr, amp_vmin, amp_vmax),
                          **ATOM_KW)
        ax_amp.add_patch(Rectangle(
            (t_days[0], zoom_lo), t_days[-1] - t_days[0], zoom_hi - zoom_lo,
            linewidth=1.8, edgecolor='white', facecolor='none', zorder=10))
        ps.panel_label(ax_amp, next(letters))
        for ce, ca in [((time_sub[-1], d_min), (t_days[0], d_min)),
                       ((time_sub[-1], d_max), (t_days[0], d_max))]:
            fig.add_artist(ConnectionPatch(
                xyA=ce, coordsA=ax_echo.transData,
                xyB=ca, coordsB=ax_amp.transData, color='black', lw=0.5,
                alpha=0.6, linestyle=(0, (2, 1.5)), zorder=5))

        # ── CS over phase (central 10 m) ────────────────────────────
        ax_ph = fig.add_subplot(gs[row, 2])
        im_ph = ax_ph.imshow(phase, aspect='auto', cmap=PHASE_CMAP,
                             vmin=-np.pi, vmax=np.pi, extent=extent,
                             origin='upper')
        cax_ph = make_axes_locatable(ax_ph).append_axes('right', size='5%',
                                                        pad=0.03)
        if row == 0:
            cbp = plt.colorbar(im_ph, cax=cax_ph)
            cbp.set_label('rad', fontsize=6)
            cbp.ax.tick_params(labelsize=5.5)
            cbp.locator = plt.MaxNLocator(4)
            cbp.update_ticks()
        else:
            cax_ph.axis('off')
        ax_ph.set_ylim(zoom_hi, zoom_lo)
        if len(t_arr):
            m = (d_arr >= zoom_lo) & (d_arr <= zoom_hi)
            ax_ph.scatter(t_arr[m], d_arr[m], c=db_arr[m],
                         alpha=fade_alpha(db_arr[m], amp_vmin, amp_vmax),
                         **ATOM_KW)
        ps.panel_label(ax_ph, next(letters))

        for ax in (ax_amp, ax_ph):
            ax.set_xlim(t_days[0], t_days[-1])
            if row < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (days)', labelpad=2)
            ax.set_yticklabels([])
        if row == 0:
            ax_amp.set_title('CS over amplitude', pad=3, fontsize=7.5,
                             fontweight='bold')
            ax_ph.set_title('CS over phase', pad=3, fontsize=7.5,
                            fontweight='bold')

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    print(f'Saved {OUT_PDF}')
    print(f'Saved {OUT_PNG}')
    sys.exit(0)
else:
    OUT_PDF = ROOT / 'figs' / 'F_mmv_lcurve.pdf'
    OUT_PNG = ROOT / 'figs' / 'F_mmv_lcurve.png'

    fig = plt.figure(figsize=(FIG_W, 5.6))
    gs = gridspec.GridSpec(
        4, 2, figure=fig, width_ratios=[1.0, 0.045],
        hspace=0.34, wspace=0.06,
        left=0.185, right=0.85, top=0.95, bottom=0.075,
    )
    last_sc = None
    for row, ((tag, d_min, d_max, _), letter) in enumerate(
            zip(WINDOWS, 'abcd')):
        ax = fig.add_subplot(gs[row, 0])
        sc = _window_panel(ax, tag, d_min, d_max, letter, marker_s=3.0)
        last_sc = sc if sc is not None else last_sc
        if row < len(WINDOWS) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (days)', labelpad=2)
    cax = fig.add_subplot(gs[:, 1])

# Shared colorbar
cb = fig.colorbar(last_sc, cax=cax)
cb.set_label('atom amplitude (dB)', fontsize=6.5, labelpad=3)
cb.ax.tick_params(labelsize=6.0, width=0.4)
cb.outline.set_linewidth(0.4)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print(f'Saved {OUT_PDF}')
print(f'Saved {OUT_PNG}')
