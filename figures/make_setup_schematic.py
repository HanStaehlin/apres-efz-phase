"""
make_setup_schematic — publication-quality deployment schematic.

`draw_schematic(ax)` renders a depth-true cross-section of the SiegVent
ApRES deployment:

  • ice surface with the ApRES instrument advecting with the ice flow
    (225.9 m/yr), radar beam cone
  • layered zone (0–600 m): coherent internal layers
  • Echo-Free Zone (600–1094 m): no resolvable layers, distributed
    volume scatterers (stipple)
  • subglacial lake at 1094 m + bedrock below
  • real depth axis on the left, zone labels in the paper palette

Run standalone to produce the single-panel figure; make_F_site.py
imports `draw_schematic` for the composite site figure.

Output (standalone):
  figs/F_setup.pdf / .png
"""

from __future__ import annotations
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrow

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import paper_style as ps
ps.apply_style()

OUT_DIR = ROOT / 'figs'

EFZ_TOP, BED = 600.0, 1094.0
LAKE_BOT = 1180.0          # bottom of the water layer (display)
Z_MAX = 1330.0             # bottom of the panel (bedrock)
X_MAX = 1000.0             # horizontal extent (arbitrary units)
X_R = 0.52 * X_MAX         # radar position


def draw_schematic(ax, compact=False):
    """Draw the deployment cross-section onto `ax`.

    compact=True uses shorter labels / smaller fonts for use as a
    sub-panel in a composite figure.
    """
    rng = np.random.default_rng(7)
    x = np.linspace(0, X_MAX, 400)
    fs = 5.4 if compact else 6.2

    # Volume-scatter stipple colour sampled from the paper's amplitude
    # colormap (RdBu_r) so the scatterers read as faint reflectors against
    # the blue ice, as in the echograms.
    col_scatter = plt.get_cmap(ps.CMAP_AMP)(0.70)

    def surf(amp, period, phase=0.0):
        return amp * np.sin(2 * np.pi * (x / period) + phase)

    # ── Zone fills ────────────────────────────────────────────────────
    z_surf = -8 + surf(5, 900, 0.4)                       # gentle surface
    z_bed = BED + surf(6, 520, 1.3)                       # lake roof
    z_lakebot = (LAKE_BOT + surf(18, 380, 0.2)
                 + 14 * np.sin(2 * np.pi * x / 950))

    # One ice fill for the whole column -- no separate EFZ shade -- so the
    # layered zone and EFZ share a background, matching the echograms.
    ax.fill_between(x, z_surf, z_bed, color=ps.COL_EFZ_FILL,
                    alpha=0.7, lw=0, zorder=1)
    ax.fill_between(x, z_bed, z_lakebot, color='#38bdf8', alpha=0.85,
                    lw=0, zorder=1)                        # lake water
    ax.fill_between(x, z_lakebot, Z_MAX, color='#a8a29e', alpha=0.85,
                    lw=0, zorder=1)                        # bedrock
    ax.fill_between(x, z_lakebot, Z_MAX, facecolor='none',
                    edgecolor='#78716c', hatch='///', lw=0, zorder=2,
                    alpha=0.6)

    # ── Internal layers (layered zone only) ───────────────────────────
    for i, zd in enumerate(np.linspace(60, 560, 9)):
        amp = 4 + 9 * (zd / 600.0)
        ax.plot(x, zd + surf(amp, 700 - 30 * i, 0.7 * i),
                color=ps.COL_LAYERS_TXT, lw=0.6, alpha=0.8, zorder=4)

    # ── Volume scatterers (stipple), throughout the ice column ─────────
    # Both the layered zone and the EFZ carry distributed scatterers; the
    # layered zone simply adds coherent layers on top.
    n_sc = 820
    xs = rng.uniform(0, X_MAX, n_sc)
    zs = rng.uniform(25.0, BED - 14, n_sc)
    ax.scatter(xs, zs, s=0.6, color=col_scatter, alpha=0.4,
               linewidths=0, zorder=3)

    # faint layered-zone / EFZ boundary (no filled band), matching panel (c)
    ax.axhline(EFZ_TOP, color=ps.COL_EFZ_TXT, lw=0.6, ls=':',
               alpha=0.5, zorder=3)

    # ── Interfaces ────────────────────────────────────────────────────
    ax.plot(x, z_surf, color='#0f172a', lw=1.1, zorder=6)
    ax.plot(x, z_bed, color='#0c4a6e', lw=1.0, zorder=4)
    ax.plot(x, z_lakebot, color='#57534e', lw=0.9, zorder=4)

    # ── Radar beam cone ──────────────────────────────────────────────
    half_w = 175.0      # half-width at the bed (clarity-exaggerated)
    ax.add_patch(Polygon([(X_R, -8), (X_R - half_w, BED),
                          (X_R + half_w, BED)], closed=True,
                         facecolor=ps.COL_ACCENT, alpha=0.10,
                         edgecolor='none', zorder=2))
    for sgn in (-1, 1):
        ax.plot([X_R, X_R + sgn * half_w], [-8, BED],
                color=ps.COL_ACCENT, lw=0.7, ls='--', alpha=0.7, zorder=4)
    ax.plot([X_R, X_R], [-8, BED], color=ps.COL_ACCENT, lw=0.6, ls=':',
            alpha=0.8, zorder=4)

    # ── ApRES instrument ─────────────────────────────────────────────
    inst_w, inst_h = 64, 52
    ax.add_patch(Rectangle((X_R - inst_w / 2, -14 - inst_h), inst_w,
                           inst_h, facecolor='#0f172a', edgecolor='none',
                           zorder=7))
    ax.plot([X_R, X_R], [-14 - inst_h, -14 - inst_h - 38],
            color='#0f172a', lw=1.0, zorder=7)
    ax.plot([X_R - 26, X_R + 26], [-14 - inst_h - 38] * 2,
            color='#0f172a', lw=1.0, zorder=7)
    ax.annotate('ApRES', (X_R + inst_w / 2 + 14, -14 - inst_h / 2),
                fontsize=fs + 0.3, color='#0f172a', va='center',
                fontweight='bold')

    # ice-flow arrow (instrument advects with the ice)
    ax.add_patch(FancyArrow(X_R - 215, -14 - inst_h - 68, 130, 0,
                            width=6, head_width=22, head_length=26,
                            facecolor='#0f172a', edgecolor='none',
                            zorder=7))
    flow_lbl = ('225.9 m yr$^{-1}$' if compact
                else 'ice flow  225.9 m yr$^{-1}$')
    ax.annotate(flow_lbl, (X_R - 150, -14 - inst_h - 104),
                fontsize=fs - 0.4, color='#0f172a', ha='center')

    # ── Zone labels ───────────────────────────────────────────────────
    if compact:
        ax.annotate('Echo-Free\nZone', (25, 840), fontsize=fs,
                    color=ps.COL_EFZ_TXT, va='center')
    else:
        ax.annotate('Echo-Free Zone', (25, 840),
                    fontsize=fs, color=ps.COL_EFZ_TXT, va='center')

    # ── Depth axis ────────────────────────────────────────────────────
    ax.set_ylim(Z_MAX, -190)
    ax.set_xlim(0, X_MAX)
    ax.set_yticks([0, 600, 1094])
    ax.set_yticklabels(['0', '600', '1094'])
    ax.set_ylabel('Depth (m)', labelpad=2)
    ax.set_xticks([])
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_bounds(0, Z_MAX)
    ax.tick_params(axis='y', length=2)


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(ps.FIG_W_SINGLE, 3.1))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.90, bottom=0.02)
    draw_schematic(ax)
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'F_setup.{ext}'
        fig.savefig(out, dpi=300 if ext == 'png' else None)
        print(f'Saved {out.relative_to(ROOT)}')
