"""
make_F_slope_concept — conceptual schematic for the Discussion: how the
three reflector populations in a resolution cell map to distinct
magnitude-coherence decorrelation regimes.  Pure vector drawing (no
data), paper palette.

  flat specular layer   -> no beam-sweep            -> v_h ~ 0 (coherent)
  sloped specular layer -> range-rate s*v_x         -> intermediate v_h
  volume scatterers     -> beam-sweep v_h*sigma_th  -> fast v_h

Output:
  figs/F_slope_concept.pdf / .png
"""

from __future__ import annotations
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.patches import ConnectionPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_style as ps
ps.apply_style()

OUT_DIR = ROOT / 'figs'

NAVY = ps.COL_LAYERS_TXT      # specular layers
BLUE = ps.COL_EFZ_TXT         # volume scatterers / curve
ACC = ps.COL_ACCENT           # motion arrows
INK = '#0f172a'

rng = np.random.default_rng(3)

fig = plt.figure(figsize=(ps.FIG_W_SINGLE, 3.1))
axT = fig.add_axes([0.04, 0.60, 0.92, 0.38])   # scene cartoons
axB = fig.add_axes([0.14, 0.13, 0.82, 0.42])   # P(v_h) regimes

cx = [0.5, 1.5, 2.5]
axT.set_xlim(0, 3); axT.set_ylim(0, 1); axT.axis('off')


def vx_arrow(ax, x, y, dx=0.18, color=ACC):
    ax.add_patch(FancyArrow(x - dx / 2, y, dx, 0, width=0.012,
                            head_width=0.06, head_length=0.05,
                            length_includes_head=True, facecolor=color,
                            edgecolor='none', zorder=5))


# ── (1) flat specular layer ───────────────────────────────────────────
x = cx[0]
axT.plot([x - 0.34, x + 0.34], [0.60, 0.60], color=NAVY, lw=2.2, zorder=3)
vx_arrow(axT, x, 0.74)
axT.text(x, 0.86, r'$v_x$', ha='center', va='center', fontsize=6, color=ACC)
axT.text(x, 0.40, 'flat\nspecular layer', ha='center', va='center',
         fontsize=6, color=NAVY)

# ── (2) sloped specular layer ─────────────────────────────────────────
x = cx[1]
sl = 0.22
axT.plot([x - 0.34, x + 0.34], [0.60 - sl, 0.60 + sl], color=NAVY, lw=2.2,
         zorder=3)
vx_arrow(axT, x, 0.80)
axT.text(x, 0.92, r'$v_x$', ha='center', va='center', fontsize=6, color=ACC)
axT.text(x, 0.36, 'sloped layer\n(slope $s$)', ha='center', va='center',
         fontsize=6, color=NAVY)

# ── (3) volume scatterers ─────────────────────────────────────────────
x = cx[2]
xs = x + rng.uniform(-0.30, 0.30, 26)
ys = 0.60 + rng.uniform(-0.16, 0.16, 26)
axT.scatter(xs, ys, s=5, color=BLUE, alpha=0.8, linewidths=0, zorder=3)
vx_arrow(axT, x, 0.84)
axT.add_patch(FancyArrow(x + 0.30, 0.84, 0.0, -0.10, width=0.012,
                         head_width=0.06, head_length=0.05,
                         length_includes_head=True, facecolor=ACC,
                         edgecolor='none', zorder=5))
axT.text(x - 0.04, 0.95, r'$v_x$', ha='center', va='center', fontsize=6, color=ACC)
axT.text(x + 0.42, 0.78, r'$v_z$', ha='center', va='center', fontsize=6, color=ACC)
axT.text(x, 0.34, 'volume\nscatterers', ha='center', va='center',
         fontsize=6, color=BLUE)

axT.text(0.0, 0.99, 'one resolution cell', ha='left', va='top',
         fontsize=6.2, color=INK, style='italic',
         transform=axT.transAxes)

# ── P(v_h) regimes ────────────────────────────────────────────────────
v = np.logspace(np.log10(0.4), np.log10(20), 400)
peaks = [(0.5, 0.10, 0.78), (2.0, 0.18, 0.46), (6.0, 0.28, 0.66)]  # (mu,width,amp) in log
P = np.zeros_like(v)
lv = np.log10(v)
for mu, w, amp in peaks:
    P += amp * np.exp(-0.5 * ((lv - np.log10(mu)) / w) ** 2)
axB.plot(v, P, color=BLUE, lw=1.3)
axB.fill_between(v, 0, P, color=BLUE, alpha=0.15, lw=0)
axB.set_xscale('log')
axB.set_xlim(0.4, 20)
axB.set_ylim(0, 1.45)
v_ticks = [0.5, 1, 2, 5, 10, 20]
axB.set_xticks(v_ticks); axB.set_xticklabels([str(t) for t in v_ticks])
axB.set_yticks([])
axB.set_xlabel(r'decorrelation rate $\rightarrow$ apparent $v_h$ (nominal)',
               labelpad=2, fontsize=6.5)
axB.set_ylabel(r'$P(v_h)$', labelpad=2, fontsize=6.5)
for sp in ('top', 'right', 'left'):
    axB.spines[sp].set_visible(False)

# regime labels above each peak
reg = [(0.5, 'coherent\n(flat layers)', NAVY),
       (2.0, 'sloped layers\n$\\propto s\\,v_x$', '#7c3aed'),
       (6.0, 'volume scatter\n$\\propto v_h\\,\\sigma_\\theta$', BLUE)]
peak_h = {vp: P[np.argmin(np.abs(v - vp))] for vp, _, _ in reg}
for vp, txt, col in reg:
    axB.text(vp, peak_h[vp] + 0.12, txt, ha='center', va='bottom',
             fontsize=5.4, color=col, fontweight='bold')

# connect each scene cartoon to its regime peak (short arrows)
for xc, (vp, _, _) in zip(cx, reg):
    cp = ConnectionPatch(
        xyA=(xc, 0.24), coordsA=axT.transData,
        xyB=(vp, peak_h[vp] + 0.30), coordsB=axB.transData,
        arrowstyle='-|>', mutation_scale=7, lw=0.7,
        color='#94a3b8', zorder=1)
    fig.add_artist(cp)

for ext in ('pdf', 'png'):
    out = OUT_DIR / f'F_slope_concept.{ext}'
    fig.savefig(out, dpi=300 if ext == 'png' else None)
    print(f'Saved {out.relative_to(ROOT)}')
