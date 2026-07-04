"""
make_F_site — study-site figures.

  (a) Location map (Siegfried et al. 2023 panel, raster)
  (b) Deployment cross-section schematic (vector, depth-true axis)
  (c) Representative amplitude profile with zone shading and the
      specular lake reflection — the quantitative 20–30 dB EFZ drop.

Paper mode (default) writes one composite ``F_site.pdf`` (a over b|c).
Thesis mode (pass "thesis") splits it into two figures sharing the
study-site story:
  * ``F_site_map.pdf``      — the location map alone, and
  * ``F_site_section.pdf``  — the deployment cross-section (a) beside the
    amplitude profile (b), which share the depth axis.

Output:
  figs/F_site.pdf / .png
"""

from __future__ import annotations
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import paper_style as ps
ps.apply_style()

from make_setup_schematic import draw_schematic, Z_MAX, EFZ_TOP, BED

OUT_DIR = ROOT / 'figs'
THESIS_FIGS = ROOT / 'figs'

# Thesis mode (pass "thesis") swaps panel (a) to our own cartopy map
# (location_ours.png, from make_F_sitemap.py), writes to the thesis
# figure directory, and splits the composite into two figures; the paper
# default keeps the existing single location map and composite layout.
THESIS = 'thesis' in sys.argv[1:]
SAVE_DIR = THESIS_FIGS if THESIS else OUT_DIR
LOC_PNG = OUT_DIR / ('location_ours.png' if THESIS else 'location.png')

# ── Load assets ───────────────────────────────────────────────────────
loc_img = imread(str(LOC_PNG))
amp_cache = np.load(ROOT / 'data' / 'F0_amplitude_cache.npz')
Rc = amp_cache['Rc']
all_profs = amp_cache['all_profs']

# Single representative burst profile (the structure is stable across
# the 312-day record, so one burst is representative).
prof = all_profs[len(all_profs) // 2]

img_aspect = loc_img.shape[0] / loc_img.shape[1]          # h/w
fig_w = ps.FIG_W_SINGLE
map_h = fig_w * img_aspect


# ── panel drawers (shared by the composite and the split layouts) ──────
def draw_map(ax):
    ax.imshow(loc_img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_amplitude(ax):
    ax.plot(prof, Rc, color=ps.COL_LINE, lw=0.6, zorder=3)

    # specular lake reflection (marked on the representative profile)
    lake_mask = (Rc > 1060) & (Rc < 1130)
    i_lake = np.argmax(prof[lake_mask])
    ax.plot(prof[lake_mask][i_lake], Rc[lake_mask][i_lake], marker='*',
            color=ps.COL_ACCENT, markersize=8, markeredgecolor='white',
            markeredgewidth=0.5, zorder=6)

    # faint zone context (boundaries + labels, no filled bands)
    ax.axhline(EFZ_TOP, color=ps.COL_EFZ_TXT, lw=0.6, ls=':', alpha=0.6)
    ax.axhline(BED, color=ps.COL_EFZ_TXT, lw=0.6, ls=':', alpha=0.6)

    ax.set_ylim(Z_MAX, -190)        # match the schematic's axis exactly
    ax.set_yticks([0, 600, 1094])
    ax.set_yticklabels([])
    ax.set_xlabel('Amplitude (dB)', labelpad=2)
    ax.set_xlim(prof.min() - 2, 4)
    ps.light_grid(ax, axis='both')
    ax.set_axisbelow(True)


def save(fig, stem):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        out = SAVE_DIR / f'{stem}.{ext}'
        fig.savefig(out, dpi=300 if ext == 'png' else None)
        print(f'Saved {out.relative_to(ROOT)}')


if THESIS:
    # ── Figure 1: location map (full-bleed) ───────────────────────────
    fig_map = plt.figure(figsize=(fig_w, map_h))
    ax = fig_map.add_axes([0.0, 0.0, 1.0, 1.0])
    draw_map(ax)
    save(fig_map, 'F_site_map')

    # ── Figure 2: deployment cross-section (a) + amplitude profile (b),
    #    sharing the depth axis ─────────────────────────────────────────
    fig_sec = plt.figure(figsize=(fig_w, 3.05))
    gs = GridSpec(1, 2, figure=fig_sec, width_ratios=[1.55, 0.62],
                  wspace=0.06, left=0.125, right=0.975, top=0.95, bottom=0.155)
    axb = fig_sec.add_subplot(gs[0, 0])
    draw_schematic(axb, compact=True)
    ps.panel_label(axb, 'a', x=-0.135, y=1.02)
    axc = fig_sec.add_subplot(gs[0, 1])
    draw_amplitude(axc)
    ps.panel_label(axc, 'b', x=-0.16, y=1.02)
    save(fig_sec, 'F_site_section')

else:
    # ── Composite (paper): a over b|c — unchanged ─────────────────────
    fig = plt.figure(figsize=(fig_w, map_h + 3.30))
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[map_h, 3.05],
                  width_ratios=[1.55, 0.62],
                  hspace=0.10, wspace=0.06,
                  left=0.125, right=0.975, top=0.99, bottom=0.075)

    ax = fig.add_subplot(gs[0, :])
    draw_map(ax)
    ps.panel_label(ax, 'a', x=-0.085, y=1.0)

    ax = fig.add_subplot(gs[1, 0])
    draw_schematic(ax, compact=True)
    ps.panel_label(ax, 'b', x=-0.135, y=1.02)

    ax = fig.add_subplot(gs[1, 1])
    draw_amplitude(ax)
    ps.panel_label(ax, 'c', x=-0.16, y=1.02)

    save(fig, 'F_site')
