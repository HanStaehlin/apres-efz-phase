"""
Shared matplotlib style for all EFZ paper figures.

Import this module and call `apply_style()` at the top of every figure
script.  This enforces one consistent typography, colour palette, and
panel-labelling convention across the paper.

Usage
-----
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import paper_style as ps

    ps.apply_style()
    ...
    ps.panel_label(ax, "a")
    ps.zone_shade(ax, orientation="y")          # EFZ band on a depth axis
    ps.shared_colorbar(fig, axes[0, :], im, "dB")
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── IEEE column widths (inches) ──────────────────────────────────────
FIG_W_SINGLE = 3.39   # single column
FIG_W_DOUBLE = 7.16   # two-column span

# ── Depth zones (m) ──────────────────────────────────────────────────
EFZ_TOP = 600.0
EFZ_BOT = 1094.0

# ── Canonical palette ────────────────────────────────────────────────
# Zone fills (used for horizontal/vertical bands on depth axes)
COL_LAYERS_FILL = "#dbeafe"   # pale blue
COL_EFZ_FILL    = "#bfdbfe"   # light blue (EFZ highlight — consistent
                              # with kingslake/viterbi overview figures)
COL_BED_FILL    = "#e5e7eb"   # light grey

# Zone text colours (slightly darker than fills)
COL_LAYERS_TXT = "#1e3a8a"
COL_EFZ_TXT    = "#1d4ed8"
COL_BED_TXT    = "#374151"

# Accent lines
COL_LINE        = "#1d4ed8"   # primary (fit curves, axes)
COL_ACCENT      = "#ef4444"   # secondary (star points, tracks)
COL_TRACK       = "#fbbf24"   # layer tracks on dark echogram

# Window colours (Fig. zooms etc.)
COL_WIN = dict(shallow="#2563eb", efz="#0ea5e9",
               bed="#dc2626",     noise="#94a3b8")

# Violin / distribution colours (harmonised with blue palette)
COL_VIOLIN_LAYERS = COL_LAYERS_FILL
COL_VIOLIN_EFZ    = "#bae6fd"   # sky-200 — teal-ish, harmonises with blue
COL_VIOLIN_NOISE  = "#cbd5e1"   # slate-300 — warmer grey

# ── Colormaps ────────────────────────────────────────────────────────
CMAP_AMP   = "RdBu_r"       # perceptually uniform, greyscale-safe, print-friendly
CMAP_PHASE = "twilight"   # diverging, perceptually uniform — excellent for phase

# Single-hue blue sequential ramp — used for semblance, R² scatter,
# MPM pole density, and any other "value intensity" heatmap. Built from
# the paper palette so every blue heatmap in the paper is the same blue.
from matplotlib.colors import LinearSegmentedColormap as _LSC
CMAP_BLUE_SEQ = _LSC.from_list(
    "paper_blue_seq", ["white", COL_EFZ_FILL, COL_LINE], N=256)

# Variant that never goes to pure white — for plotting points on top of
# a coloured background (echogram overlays) where white would disappear.
CMAP_BLUE_DENSE = _LSC.from_list(
    "paper_blue_dense", [COL_LAYERS_FILL, COL_LINE, COL_LAYERS_TXT], N=256)


# ── rcParams preset ──────────────────────────────────────────────────
def apply_style() -> None:
    """Set matplotlib rcParams to the paper-wide house style."""
    plt.rcParams.update({
        # Typography — Helvetica is the IEEE / Nature / Science standard;
        # falls back to Arial on systems without Helvetica installed.
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         8.0,
        "axes.titlesize":    8.5,
        "axes.titleweight":  "bold",
        "axes.labelsize":    8.0,
        "xtick.labelsize":   7.0,
        "ytick.labelsize":   7.0,
        "legend.fontsize":   7.0,
        "legend.frameon":    False,       # frameless legends by default
        "legend.borderpad":  0.3,
        # Axes / ticks
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.formatter.use_mathtext": True,   # clean scientific notation
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        # Figure / save
        "figure.dpi":        200,
        "figure.facecolor":  "white",
        "figure.constrained_layout.use": False,  # we manage layout per figure
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
        # Font embedding (required for IEEE PDF submission)
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


# ── Panel labels ─────────────────────────────────────────────────────
def panel_label(ax: Axes, letter: str, *, x: float = -0.02, y: float = 1.02,
                fontsize: float = 9.5, weight: str = "bold") -> None:
    """Place a bold ``(a)`` style label at the top-left outside the axes.

    Uses axes-fraction coordinates so position is consistent regardless
    of subplot size.  Default placement is just above the axes box.
    """
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight=weight,
            ha="right", va="bottom", color="black", zorder=10)


# ── EFZ / zone shading ───────────────────────────────────────────────
def zone_shade(ax: Axes, *, orientation: str = "y", efz: bool = True,
               layers: bool = False, alpha_efz: float = 0.08,
               alpha_layers: float = 0.10,
               dashed_lines: bool = True) -> None:
    """Shade depth zones on a Matplotlib axis.

    House style (matches fig. 6.1 / F5, the reference figure): a light
    EFZ tint plus a thin, subtle boundary line rather than a heavy
    dashed rule, so the shading never competes with the data.

    Parameters
    ----------
    orientation : "y" if the depth axis is vertical (most depth plots),
                  "x" if horizontal.
    efz, layers : which zones to shade.
    dashed_lines: draw thin boundary lines at 600 / 1094 m.
    """
    span = ax.axhspan if orientation == "y" else ax.axvspan
    line = ax.axhline if orientation == "y" else ax.axvline

    if layers:
        span(0, EFZ_TOP, color=COL_LAYERS_FILL, alpha=alpha_layers, zorder=1)
    if efz:
        span(EFZ_TOP, EFZ_BOT, color=COL_EFZ_FILL, alpha=alpha_efz, zorder=1)
    if dashed_lines:
        line(EFZ_TOP, color="black", lw=0.65, ls=":", alpha=0.6, zorder=2)
        line(EFZ_BOT, color="black", lw=0.65, ls=":", alpha=0.6, zorder=2)


def zone_lines(ax: Axes, *, orientation: str = "y",
               color: str = "black", lw: float = 0.65,
               ls: str = ":") -> None:
    """Draw thin EFZ boundary lines (for velocity / coherence panels)."""
    line = ax.axhline if orientation == "y" else ax.axvline
    for d in (EFZ_TOP, EFZ_BOT):
        line(d, color=color, lw=lw, ls=ls, zorder=5)


def zone_label(ax: Axes, *, position: str = "right",
               fontsize: float = 7.5) -> None:
    """Place an italic 'EFZ' label in the shaded band."""
    if position == "right":
        x, ha = ax.get_xlim()[1] * 0.98, "right"
    else:
        x, ha = ax.get_xlim()[0] * 1.02, "left"
    ax.text(x, (EFZ_TOP + EFZ_BOT) / 2, "EFZ",
            ha=ha, va="center",
            fontsize=fontsize, color=COL_EFZ_TXT,
            fontstyle="italic", fontweight="bold")


# ── Echogram zone labels with contrast pill ──────────────────────────
def echogram_zone_label(ax: Axes, y: float, text: str, *,
                        fontsize: float = 5.5,
                        color: str = "white") -> None:
    """Place a zone label on an echogram with a semi-transparent background
    pill for readability on any colormap."""
    ax.text(0.04, y, text, transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=fontsize, color=color,
            fontweight="bold", clip_on=False, zorder=10,
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="black", alpha=0.45, lw=0))


# ── Light grid helper ────────────────────────────────────────────────
def light_grid(ax: Axes, *, axis: str = "both") -> None:
    """Add subtle grid lines to a line-plot panel."""
    ax.grid(True, axis=axis, color="#e5e7eb", lw=0.4, zorder=0)


# ── Colour bars ──────────────────────────────────────────────────────
def panel_colorbar(ax: Axes, im, label: str = "", n_ticks: int = 4,
                   pad: float = 0.04, size: str = "4%",
                   label_top: bool = False):
    """A small colour bar attached to a single axis (for individual panels).

    Set ``label_top=True`` to place the label horizontally above the bar
    instead of rotated alongside it — frees horizontal space when the
    axis is width-constrained.  Returns the colorbar.
    """
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size=size, pad=pad)
    cb      = plt.colorbar(im, cax=cax)
    if label_top:
        cax.set_title(label, fontsize=6.5, pad=3)
    else:
        cb.set_label(label, labelpad=2)
    cb.locator = mpl.ticker.MaxNLocator(n_ticks)
    cb.update_ticks()
    cb.outline.set_linewidth(0.4)
    cax.tick_params(width=0.4, labelsize=6.0)
    return cb


def shared_colorbar(fig, axes, im, label: str = "", *, orientation: str = "vertical",
                    pad: float = 0.02, fraction: float = 0.025,
                    n_ticks: int = 5):
    """One colour bar shared by several axes — replaces per-panel bars.

    `axes` may be a list or 1-D array of Axes along the shared dimension.
    """
    cb = fig.colorbar(im, ax=list(axes), orientation=orientation,
                      pad=pad, fraction=fraction)
    cb.set_label(label, labelpad=3)
    cb.locator = mpl.ticker.MaxNLocator(n_ticks)
    cb.update_ticks()
    cb.outline.set_linewidth(0.4)
    return cb
