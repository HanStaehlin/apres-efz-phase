#!/usr/bin/env python3
"""
Generate the 5 publication figures for the EFZ Phase Coherence paper.

F0  Introduction   — schematic | location schematic | single-measurement amplitude profile
F1  Shallow layers   100–110 m — amplitude | phase | tracked layers
F2  Below bed       1800–1810 m — amplitude | phase | (no layers: noise demo)
F3  Bed interface   1090–1100 m — amplitude | phase | (bed reflection)
F4  EFZ             800–810 m  — amplitude | phase | (no layers: the key question)

Output: figs/F{0-4}.pdf  (+ .png previews)

Usage:
    python figures/make_paper_figures.py
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
from mpl_toolkits.axes_grid1 import make_axes_locatable
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
from apres.io import fmcw_load, fmcw_range
from apres.superresolution import fmcw_matrix_pencil

OUT_DIR = ROOT / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared paper style ──────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_style as ps
ps.apply_style()

FIG_W   = ps.FIG_W_DOUBLE    # 7.16 in — two-column width
PANEL_H = 2.55               # inches (height of F1-F4 rows)

# Legacy alias kept so the rest of the script needs minimal changes
C = dict(
    layers_fill=ps.COL_LAYERS_FILL,
    efz_fill   =ps.COL_EFZ_FILL,
    bed_fill   =ps.COL_BED_FILL,
    lake_fill  ="#bfdbfe",
    layers_txt =ps.COL_LAYERS_TXT,
    efz_txt    =ps.COL_EFZ_TXT,
    bed_txt    =ps.COL_BED_TXT,
    track      =ps.COL_TRACK,
)

AMP_CMAP   = ps.CMAP_AMP    # now RdBu_r
PHASE_CMAP = ps.CMAP_PHASE


# ════════════════════════════════════════════════════════════════════
# Data loaders (load once, pass around)
# ════════════════════════════════════════════════════════════════════

def load_zarr():
    zpath = ROOT / "data" / "ImageP2_python.zarr"
    root  = zarr.open(str(zpath), mode="r")
    Rcoarse   = np.array(root["Rcoarse"]).flatten()
    time_days = np.array(root["time_days"]).flatten()
    return Rcoarse, root["range_img"], root["raw_complex"], time_days



def slice_window(Rcoarse, range_img_z, raw_complex_z, d_min, d_max):
    """Return (depths, amp_db, phase) for the depth window."""
    idx    = np.where((Rcoarse >= d_min) & (Rcoarse <= d_max))[0]
    depths = Rcoarse[idx]
    amp    = np.array(range_img_z[idx[0]:idx[-1] + 1, :]).astype(np.float32)
    cpx    = np.array(raw_complex_z[idx[0]:idx[-1] + 1, :])
    amp_db = 20.0 * np.log10(np.abs(amp) + 1e-15)
    phase  = np.angle(cpx)
    return depths, amp_db, phase


# ════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════

def add_colorbar(ax, im, label="", n_ticks=4):
    """Wrapper around ps.panel_colorbar for backward compatibility."""
    ps.panel_colorbar(ax, im, label=label, n_ticks=n_ticks)


def panel_letter(ax, letter, pad_x=-0.02, pad_y=1.02):
    """Wrapper around ps.panel_label for backward compatibility."""
    ps.panel_label(ax, letter, x=pad_x, y=pad_y)



# ════════════════════════════════════════════════════════════════════
# F0 — Introduction figure
# ════════════════════════════════════════════════════════════════════

def make_F0(Rcoarse, range_img_z, time_days):
    """
    Two panels (full-width figure):
      (a) Real full-column echogram with zone annotations
      (b) Amplitude profile envelope (all measurements)
    """
    dat_files = sorted((ROOT / "data" / "raw").glob("*.DAT"))
    data = fmcw_load(str(dat_files[len(dat_files) // 2]))
    Rc, _Rf, spec_cor, _spec = fmcw_range(data, pad_factor=8, max_range=1500)

    # Full echogram
    D_MAX = 1200.0
    mask  = Rcoarse <= D_MAX
    idx   = np.where(mask)[0]
    t_step = 4
    amp_full = np.array(range_img_z[idx[0]:idx[-1] + 1, ::t_step]).astype(np.float32)
    amp_full_db = 20.0 * np.log10(np.abs(amp_full) + 1e-15)
    depths_full = Rcoarse[idx]
    time_sub    = time_days[::t_step]

    # ── Amplitude Profile Caching ────────────────────────────────────
    cache_path = ROOT / "data" / "F0_amplitude_cache.npz"
    if cache_path.exists():
        print(f"  Loading amplitude envelope from cache: {cache_path}")
        cache_data = np.load(cache_path)
        Rc = cache_data["Rc"]
        all_profs = cache_data["all_profs"]
    else:
        print("  Generating amplitude envelope (no cache found)...")
        all_profs = []
        for fp in dat_files[::3]:
            try:
                d = fmcw_load(str(fp))
                Rc_i, _Rf_i, spec_i, _ = fmcw_range(d, pad_factor=8, max_range=1500)
                avg_i = np.mean(spec_i, axis=0)
                prof_i = 20 * np.log10(np.abs(avg_i) + 1e-30)
                prof_i -= prof_i.max()
                all_profs.append(prof_i)
                Rc = Rc_i  # keep the last one for depths
            except Exception:
                pass
        all_profs = np.array(all_profs)
        np.savez(cache_path, Rc=Rc, all_profs=all_profs)
        print(f"  Saved amplitude envelope to cache: {cache_path}")

    print(f"  Amplitude envelope: {all_profs.shape[0]} profiles")

    fig, axes = plt.subplots(1, 2, figsize=(3.39, 3.2),  # standard single-column width
                             gridspec_kw={"wspace": 0.15,
                                          "width_ratios": [1.4, 0.45],
                                          "left": 0.15, "right": 0.95,
                                          "top": 0.90, "bottom": 0.12})

    zones = [
        (0,    600,  C["layers_fill"], "Internal layers", C["layers_txt"]),
        (600,  1094, C["efz_fill"],    "Echo-Free Zone",  C["efz_txt"]),
        (1094, 1200, C["bed_fill"],    "Bed / Lake",      C["bed_txt"]),
    ]

    # ── (a) Real full-column echogram ────────────────────────────────
    ax = axes[0]
    vmin = np.percentile(amp_full_db, 5)
    vmax = np.percentile(amp_full_db, 99)
    extent_full = [time_sub[0], time_sub[-1], D_MAX, depths_full[0]]
    ax.imshow(amp_full_db, aspect="auto", cmap=AMP_CMAP,
              vmin=vmin, vmax=vmax, extent=extent_full, origin="upper")

    for z_top, z_bot, fc, label, tc in zones:
        ax.axhspan(z_top, min(z_bot, D_MAX), color=fc, alpha=0.18, zorder=2)
        if z_bot <= D_MAX:
            ax.axhline(z_bot, color="white", lw=0.7, ls="--", zorder=3)
        mid = (z_top + min(z_bot, D_MAX)) / 2
        ps.echogram_zone_label(ax, mid, label)

    ax.set_ylim(D_MAX, 0)
    ax.set_xlim(time_sub[0], time_sub[-1])
    ax.set_xlabel("Time (days)", labelpad=2)
    ax.set_ylabel("Depth (m)", labelpad=2)
    ax.set_title("Echogram", pad=3)
    panel_letter(ax, "a")

    # ── (b) Single amplitude profile ─────────────────────────────────
    ax2 = axes[1]

    prof = all_profs[len(all_profs) // 2]   # representative single sample

    ax2.plot(prof, Rc, color="#2563eb", lw=0.8)

    ax2.set_xlabel("Amplitude (dB)", labelpad=2)
    ax2.set_ylim(1200, 0)
    ax2.set_xlim(prof.min() - 2, 2)
    ax2.set_yticklabels([])
    ax2.set_title("Amplitude", pad=3)

    for z_top, z_bot, fc, label, tc in zones:
        if z_bot <= 1200:
            ax2.axhline(z_bot, color=tc, lw=0.6, ls=":", zorder=3, alpha=0.7)

    # Lake surface star
    lake_mask = (Rc > 1060) & (Rc < 1130)
    lake_idx  = np.argmax(prof[lake_mask])
    lake_d    = Rc[lake_mask][lake_idx]
    lake_db   = prof[lake_mask][lake_idx]
    ax2.plot(lake_db, lake_d, marker="*", color="#ef4444",
             markersize=8, markeredgecolor="white", markeredgewidth=0.5,
             zorder=6)

    ps.light_grid(ax2, axis="x")
    panel_letter(ax2, "b")

    _save(fig, "F0")


# ════════════════════════════════════════════════════════════════════
# F1–F4 — Depth-window triptychs
# ════════════════════════════════════════════════════════════════════

# (d_min, d_max, title, M)  — M=1 forces single pole; M=20 extracts top 20
WINDOWS = {
    "F1": (100,  120,  "Shallow internal layers (100–120 m)",    20),
    "F2": (1800, 1820, "Below bed – thermal noise (1800–1820 m)", 20),
    "F3": (1085, 1105, "Bed/lake interface (1085–1105 m)",         1),
    "F4": (800,  820,  "Echo-Free Zone (800–820 m)",              20),
}


def compute_mpm_results(step=5):
    """
    Run MPM on every WINDOWS entry for every step-th DAT file.
    Returns {tag: {"days": [...], "depths": [...]}} dict.
    """
    data_dir  = ROOT / "data"
    dat_files = sorted(data_dir.glob("**/*.DAT"))[::step]
    print(f"  Running MPM over {len(dat_files)} files (step={step})…")

    results = {tag: {"days": [], "depths": [], "powers": []} for tag in WINDOWS}
    t0 = None

    for i, fp in enumerate(dat_files):
        if i % 50 == 0:
            print(f"    {i}/{len(dat_files)}")
        try:
            data = fmcw_load(str(fp))
        except Exception:
            continue

        if t0 is None and data.time_stamp is not None:
            t0 = data.time_stamp
        day_i = ((data.time_stamp - t0).total_seconds() / 86400.0
                 if t0 and data.time_stamp else i * step / 24.0)

        for tag, (d_min, d_max, _, M) in WINDOWS.items():
            try:
                res = fmcw_matrix_pencil(data, d_min, d_max, M=M)
                for d, p in zip(res["depths"], res["powers"]):
                    if d_min <= d <= d_max:
                        results[tag]["days"].append(day_i)
                        results[tag]["depths"].append(d)
                        results[tag]["powers"].append(p)
            except Exception:
                pass

    for tag, v in results.items():
        print(f"    {tag}: {len(v['days'])} MPM poles collected")
    return results


def _plot_mpm_poles(ax, tag, mpm_data, pole_cmap, label_color="black",
                    dot_size=4):
    """Scatter MPM poles coloured by amplitude onto `ax` (no background drawn)."""
    mpm_days   = mpm_data[tag]["days"]
    mpm_depths = mpm_data[tag]["depths"]
    mpm_powers = mpm_data[tag]["powers"]
    if len(mpm_days) > 0:
        pows = np.array(mpm_powers)
        p_lo = np.percentile(pows, 5)
        p_hi = np.percentile(pows, 95)
        sc = ax.scatter(mpm_days, mpm_depths,
                        c=pows, cmap=pole_cmap,
                        vmin=p_lo, vmax=p_hi,
                        s=dot_size, alpha=0.75, linewidths=0, zorder=3)
        add_colorbar(ax, sc, "dB")
        ax.text(0.02, 0.97, f"{len(mpm_days)} poles",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=5.5, color=label_color)
    else:
        ax.text(0.5, 0.5, "No poles",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color=label_color, fontweight="bold")


def make_echo_figure(tag, d_min, d_max, suptitle,
                     Rcoarse, range_img_z, raw_complex_z, time_days,
                     mpm_data, two_panel=False):
    depths, amp_db, phase = slice_window(
        Rcoarse, range_img_z, raw_complex_z, d_min, d_max)

    n_cols = 2 if two_panel else 3
    # 2-panel: panels are wider so give a bit more height to keep aspect
    fig_h  = PANEL_H if not two_panel else PANEL_H + 0.1
    fig, axes = plt.subplots(
        1, n_cols, figsize=(FIG_W, fig_h),
        gridspec_kw={"wspace": 0.32 if two_panel else 0.38,
                     "left": 0.09, "right": 0.97,
                     "top": 0.84, "bottom": 0.17})

    vmin_amp = np.percentile(amp_db, 2)
    vmax_amp = np.percentile(amp_db, 98)
    extent   = [time_days[0], time_days[-1], d_max, d_min]

    # ── (a) Amplitude ────────────────────────────────────────────────
    ax = axes[0]
    im_a = ax.imshow(amp_db, aspect="auto", cmap=AMP_CMAP,
                     vmin=vmin_amp, vmax=vmax_amp,
                     extent=extent, origin="upper")
    ax.set_xlabel("Time (days)", labelpad=2)
    ax.set_ylabel("Depth (m)", labelpad=2)
    ax.set_title("Amplitude")
    add_colorbar(ax, im_a, "dB")
    panel_letter(ax, "a")

    # ── (b) Phase ────────────────────────────────────────────────────
    ax = axes[1]
    im_b = ax.imshow(phase, aspect="auto", cmap=PHASE_CMAP,
                     vmin=-np.pi, vmax=np.pi,
                     extent=extent, origin="upper")
    ax.set_xlabel("Time (days)", labelpad=2)
    ax.set_yticklabels([])
    ax.set_title("Phase")
    add_colorbar(ax, im_b, "rad")
    panel_letter(ax, "b")

    if not two_panel:
        # ── (c) MPM poles – clean white background ───────────────────
        from matplotlib.colors import LinearSegmentedColormap
        POLE_CMAP_LIGHT = LinearSegmentedColormap.from_list(
            "poles_light", ["#dbeafe", "#3b82f6", "#1e3a8a"], N=256)
        ax = axes[2]
        ax.set_facecolor("white")
        _plot_mpm_poles(ax, tag, mpm_data, POLE_CMAP_LIGHT,
                        label_color="#1e3a8a", dot_size=4)
        ax.set_xlabel("Time (days)", labelpad=2)
        ax.set_yticklabels([])
        ax.set_title("MPM poles")
        panel_letter(ax, "c")

    for a in axes:
        a.set_ylim(d_max, d_min)
        a.set_xlim(time_days[0], time_days[-1])

    fig.suptitle(suptitle, fontsize=8.5, fontweight="bold", y=0.97)
    _save(fig, tag)


# ════════════════════════════════════════════════════════════════════
# Fref — Combined 3×3 reference figure (F1 / F2 / F3)
# ════════════════════════════════════════════════════════════════════

def make_F_references(Rcoarse, range_img_z, raw_complex_z, time_days, mpm_data=None):
    """
    New layout for TGARS:
      Left column (full height): full echogram 0–1200 m with zone annotations
                                  and dashed rectangles marking zoom regions
      Middle column (4 rows):     amplitude zoom-ins for each window
      Right column (4 rows):      phase zoom-ins (central 10 m) for each window
    """
    from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch

    ref_tags = ["F1", "F4", "F3", "F2"]   # top-to-bottom: shallow, EFZ, bed, noise
    row_titles = [
        "Shallow layers",
        "Echo-Free Zone",
        "Bed / lake",
        "Below bed (noise)",
    ]

    # ── Full echogram data ──────────────────────────────────────────
    D_MAX_ECHO = 2000.0
    mask_echo = Rcoarse <= D_MAX_ECHO
    idx_echo  = np.where(mask_echo)[0]
    t_step = 4
    amp_echo = np.array(range_img_z[idx_echo[0]:idx_echo[-1]+1, ::t_step]).astype(np.float32)
    amp_echo_db = 20.0 * np.log10(np.abs(amp_echo) + 1e-15)
    depths_echo = Rcoarse[idx_echo]
    time_sub = time_days[::t_step]

    zones = [
        (0,    600,  C["layers_fill"], "Internal layers", C["layers_txt"]),
        (600,  1094, C["efz_fill"],    "Echo-Free Zone",  C["efz_txt"]),
        (1094, 1200, C["bed_fill"],    "Bed / Lake",      C["bed_txt"]),
    ]

    # ── Figure layout: GridSpec with left column spanning all rows ───
    fig = plt.figure(figsize=(FIG_W, 7.0))
    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        width_ratios=[1.0, 0.9, 0.9],
        hspace=0.10, wspace=0.30,
        left=0.07, right=0.97, top=0.96, bottom=0.06,
    )

    # ── (a) Full echogram (spans all 4 rows) ────────────────────────
    ax_echo = fig.add_subplot(gs[:, 0])
    vmin_echo = np.percentile(amp_echo_db, 5)
    vmax_echo = np.percentile(amp_echo_db, 99)
    extent_echo = [time_sub[0], time_sub[-1], D_MAX_ECHO, depths_echo[0]]
    ax_echo.imshow(amp_echo_db, aspect="auto", cmap=AMP_CMAP,
                   vmin=vmin_echo, vmax=vmax_echo,
                   extent=extent_echo, origin="upper")

    # Zone shading and labels
    for z_top, z_bot, fc, label, tc in zones:
        ax_echo.axhspan(z_top, min(z_bot, D_MAX_ECHO), color=fc, alpha=0.15, zorder=2)
        if z_bot <= D_MAX_ECHO:
            ax_echo.axhline(z_bot, color="white", lw=0.7, ls="--", zorder=3)

    ax_echo.set_ylim(D_MAX_ECHO, 0)
    ax_echo.set_xlim(time_sub[0], time_sub[-1])
    ax_echo.set_xlabel("Time (days)", labelpad=2)
    ax_echo.set_ylabel("Depth (m)", labelpad=2)
    ax_echo.set_title("Full echogram", pad=3, fontsize=7.5, fontweight="bold")
    panel_letter(ax_echo, "a")

    # ── Zoom-in panels (4 rows × 2 cols: amplitude + phase) ────────
    zoom_colors = [ps.COL_WIN["shallow"], ps.COL_WIN["efz"],
                   ps.COL_WIN["bed"],     ps.COL_WIN["noise"]]
    letters = iter("bcdefghi")

    for row, (tag, title, zc) in enumerate(zip(ref_tags, row_titles, zoom_colors)):
        d_min, d_max, _, M = WINDOWS[tag]
        depths, amp_db, phase = slice_window(
            Rcoarse, range_img_z, raw_complex_z, d_min, d_max)
        vmin_amp = np.percentile(amp_db, 2)
        vmax_amp = np.percentile(amp_db, 98)
        extent = [time_days[0], time_days[-1], d_max, d_min]

        d_mid = (d_min + d_max) / 2.0
        zoom_lo, zoom_hi = d_mid - 5, d_mid + 5

        # Mark zoom region on the full echogram — white rectangle, high contrast
        if d_max <= D_MAX_ECHO:
            rect = Rectangle(
                (time_sub[0], d_min),
                time_sub[-1] - time_sub[0], d_max - d_min,
                linewidth=1.8, edgecolor="white", facecolor="none",
                linestyle="-", zorder=10)
            ax_echo.add_patch(rect)
            # Label on the left margin
            mid_d = (d_min + d_max) / 2
            ax_echo.text(-0.02, mid_d, f"{chr(ord('b') + row*2)}",
                         transform=ax_echo.get_yaxis_transform(),
                         ha="right", va="center", fontsize=7,
                         fontweight="bold", color="white",
                         bbox=dict(boxstyle="round,pad=0.15",
                                   fc="black", alpha=0.6, lw=0))

        # ── Amplitude zoom-in ───────────────────────────────────────
        ax_amp = fig.add_subplot(gs[row, 1])
        im = ax_amp.imshow(amp_db, aspect="auto", cmap=AMP_CMAP,
                           vmin=vmin_amp, vmax=vmax_amp,
                           extent=extent, origin="upper")
        add_colorbar(ax_amp, im, "dB")
        # Mark phase zoom window — same solid style as full-echogram rectangles
        rect2 = Rectangle(
            (time_days[0], zoom_lo),
            time_days[-1] - time_days[0], zoom_hi - zoom_lo,
            linewidth=1.8, edgecolor="white", facecolor="none",
            linestyle="-", zorder=10)
        ax_amp.add_patch(rect2)
        panel_letter(ax_amp, next(letters))

        # ── Connection lines from full echogram zoom box to amp panel ─
        # Only draw for windows that fit within the full echogram range;
        # noise window at 1800–1820 m lies below D_MAX_ECHO=2000 m but
        # below the lake so it still fits.
        if d_max <= D_MAX_ECHO:
            for (corner_echo, corner_amp) in [
                ((time_sub[-1], d_min), (time_days[0], d_min)),  # top edge
                ((time_sub[-1], d_max), (time_days[0], d_max)),  # bottom edge
            ]:
                cp = ConnectionPatch(
                    xyA=corner_echo, coordsA=ax_echo.transData,
                    xyB=corner_amp,  coordsB=ax_amp.transData,
                    color="black", lw=0.5, alpha=0.6,
                    linestyle=(0, (2, 1.5)), zorder=5,
                )
                fig.add_artist(cp)

        # ── Phase zoom-in (central 10 m) ────────────────────────────
        # Reserve colorbar space on every row for alignment, but only
        # draw the bar on the top row — all four panels share the same
        # -π to +π scale, so four identical colorbars were redundant.
        ax_ph = fig.add_subplot(gs[row, 2])
        im_ph = ax_ph.imshow(phase, aspect="auto", cmap=PHASE_CMAP,
                             vmin=-np.pi, vmax=np.pi,
                             extent=extent, origin="upper")
        divider_ph = make_axes_locatable(ax_ph)
        cax_ph = divider_ph.append_axes("right", size="5%", pad=0.03)
        if row == 0:
            cb = plt.colorbar(im_ph, cax=cax_ph)
            cb.set_label("rad", fontsize=6)
            cb.ax.tick_params(labelsize=5.5)
            cb.locator = plt.MaxNLocator(4)
            cb.update_ticks()
        else:
            cax_ph.axis("off")
        ax_ph.set_ylim(zoom_hi, zoom_lo)
        panel_letter(ax_ph, next(letters))

        # Formatting
        for col_idx, ax in enumerate([ax_amp, ax_ph]):
            ax.set_xlim(time_days[0], time_days[-1])
            if row < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (days)", labelpad=2)
            ax.set_yticklabels([])

        # Store first-row axes for column titles
        if row == 0:
            ax_amp.set_title("Amplitude (zoom)", pad=3,
                             fontsize=7.5, fontweight="bold")
            ax_ph.set_title("Phase (zoom)", pad=3,
                            fontsize=7.5, fontweight="bold")

    _save(fig, "Fref")

    # ── Save each subplot as an individual PNG ───────────────────────
    row_tags = ["shallow", "efz", "bed", "noise"]

    for row, (tag, title, zc) in enumerate(zip(ref_tags, row_titles, zoom_colors)):
        d_min, d_max, _, M = WINDOWS[tag]
        depths, amp_db_w, phase_w = slice_window(
            Rcoarse, range_img_z, raw_complex_z, d_min, d_max)
        vmin_amp = np.percentile(amp_db_w, 2)
        vmax_amp = np.percentile(amp_db_w, 98)
        extent_w = [time_days[0], time_days[-1], d_max, d_min]
        d_mid = (d_min + d_max) / 2.0
        zoom_lo_w, zoom_hi_w = d_mid - 5, d_mid + 5

        for panel, cmap, vlo, vhi, ylim, cb_label in [
            ("amp",   AMP_CMAP,   vmin_amp, vmax_amp, (d_max, d_min),        "Amplitude (dB)"),
            ("phase", PHASE_CMAP, -np.pi,   np.pi,    (zoom_hi_w, zoom_lo_w), "Phase (rad)"),
        ]:
            fig_s, ax_s = plt.subplots(figsize=(2.4, 2.0),
                                       gridspec_kw={"left": 0.15, "right": 0.78,
                                                    "top": 0.85, "bottom": 0.16})
            im_s = ax_s.imshow(amp_db_w if panel == "amp" else phase_w,
                               aspect="auto", cmap=cmap, vmin=vlo, vmax=vhi,
                               extent=extent_w, origin="upper")
            ax_s.set_ylim(*ylim)
            ax_s.set_xlim(time_days[0], time_days[-1])
            ax_s.set_xlabel("Time (days)", labelpad=2, fontsize=6)
            ax_s.set_ylabel("Depth (m)", labelpad=2, fontsize=6)
            ax_s.tick_params(labelsize=5.5)
            cb = fig_s.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
            cb.set_label(cb_label, fontsize=5.5)
            cb.ax.tick_params(labelsize=5)
            out = OUT_DIR / f"Fref_{row_tags[row]}_{panel}.png"
            fig_s.savefig(str(out), dpi=200, bbox_inches="tight")
            plt.close(fig_s)
            print(f"  saved {out.relative_to(ROOT)}")

    # Full echogram panel
    fig_s, ax_s = plt.subplots(figsize=(2.4, 4.8),
                               gridspec_kw={"left": 0.17, "right": 0.80,
                                            "top": 0.93, "bottom": 0.08})
    im_echo = ax_s.imshow(amp_echo_db, aspect="auto", cmap=AMP_CMAP,
                          vmin=vmin_echo, vmax=vmax_echo,
                          extent=extent_echo, origin="upper")
    for z_top, z_bot, fc, label, tc in zones:
        ax_s.axhspan(z_top, min(z_bot, D_MAX_ECHO), color=fc, alpha=0.15, zorder=2)
        if z_bot <= D_MAX_ECHO:
            ax_s.axhline(z_bot, color="white", lw=0.7, ls="--", zorder=3)
    ax_s.set_ylim(D_MAX_ECHO, 0)
    ax_s.set_xlim(time_sub[0], time_sub[-1])
    ax_s.set_xlabel("Time (days)", labelpad=2, fontsize=6)
    ax_s.set_ylabel("Depth (m)", labelpad=2, fontsize=6)
    ax_s.tick_params(labelsize=5.5)
    ax_s.set_title("Full echogram\n0–2000 m", fontsize=6.5, fontweight="bold", pad=3)
    cb = fig_s.colorbar(im_echo, ax=ax_s, fraction=0.046, pad=0.04)
    cb.set_label("Amplitude (dB)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    out = OUT_DIR / "Fref_echogram.png"
    fig_s.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig_s)
    print(f"  saved {out.relative_to(ROOT)}")


# ════════════════════════════════════════════════════════════════════
# F_mpm — MPM pole structure at 4 reference windows (for Section VI)
# ════════════════════════════════════════════════════════════════════

def make_F_mpm(Rcoarse, range_img_z, time_days, mpm_data):
    """
    MPM pole figure matching Fref layout:
      Left column (full height): full echogram 0–2000 m
      Right column (4 rows):     MPM pole scatter at each window
    """
    from matplotlib.patches import Rectangle

    # Palette-consistent blue ramp for pole density (never pure white —
    # poles are plotted on a coloured echogram-style background).
    POLE_CMAP = ps.CMAP_BLUE_DENSE

    ref_tags = ["F1", "F4", "F3", "F2"]
    row_titles = ["Shallow layers", "Echo-Free Zone",
                  "Bed / lake", "Below bed (noise)"]
    zoom_colors = [ps.COL_WIN["shallow"], ps.COL_WIN["efz"],
                   ps.COL_WIN["bed"],     ps.COL_WIN["noise"]]

    # Full echogram
    D_MAX_ECHO = 2000.0
    mask_echo = Rcoarse <= D_MAX_ECHO
    idx_echo = np.where(mask_echo)[0]
    t_step = 4
    amp_echo = np.array(range_img_z[idx_echo[0]:idx_echo[-1]+1, ::t_step]).astype(np.float32)
    amp_echo_db = 20.0 * np.log10(np.abs(amp_echo) + 1e-15)
    depths_echo = Rcoarse[idx_echo]
    time_sub = time_days[::t_step]

    zones = [
        (0,    600,  C["layers_fill"], "Internal layers", C["layers_txt"]),
        (600,  1094, C["efz_fill"],    "Echo-Free Zone",  C["efz_txt"]),
        (1094, 2000, C["bed_fill"],    "Bed / Lake",      C["bed_txt"]),
    ]

    fig = plt.figure(figsize=(FIG_W, 7.0))
    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        width_ratios=[1.0, 1.2],
        hspace=0.10, wspace=0.25,
        left=0.07, right=0.97, top=0.96, bottom=0.06,
    )

    # ── (a) Full echogram ───────────────────────────────────────────
    ax_echo = fig.add_subplot(gs[:, 0])
    vmin_echo = np.percentile(amp_echo_db, 5)
    vmax_echo = np.percentile(amp_echo_db, 99)
    extent_echo = [time_sub[0], time_sub[-1], D_MAX_ECHO, depths_echo[0]]
    ax_echo.imshow(amp_echo_db, aspect="auto", cmap=AMP_CMAP,
                   vmin=vmin_echo, vmax=vmax_echo,
                   extent=extent_echo, origin="upper")

    for z_top, z_bot, fc, label, tc in zones:
        ax_echo.axhspan(z_top, min(z_bot, D_MAX_ECHO),
                        color=fc, alpha=0.15, zorder=2)
        if z_bot <= D_MAX_ECHO:
            ax_echo.axhline(z_bot, color="white", lw=0.7, ls="--", zorder=3)

    ax_echo.set_ylim(D_MAX_ECHO, 0)
    ax_echo.set_xlim(time_sub[0], time_sub[-1])
    ax_echo.set_xlabel("Time (days)", labelpad=2)
    ax_echo.set_ylabel("Depth (m)", labelpad=2)
    ax_echo.set_title("Full echogram", pad=3, fontsize=7.5, fontweight="bold")
    panel_letter(ax_echo, "a")

    # ── MPM pole panels (4 rows) ────────────────────────────────────
    letters = iter("bcde")

    for row, (tag, title, zc) in enumerate(zip(ref_tags, row_titles, zoom_colors)):
        d_min, d_max, _, M = WINDOWS[tag]

        # Mark on echogram
        if d_max <= D_MAX_ECHO:
            rect = Rectangle(
                (time_sub[0], d_min),
                time_sub[-1] - time_sub[0], d_max - d_min,
                linewidth=1.8, edgecolor="white", facecolor="none",
                linestyle="-", zorder=10)
            ax_echo.add_patch(rect)
            mid_d = (d_min + d_max) / 2
            ax_echo.text(-0.02, mid_d, f"{chr(ord('b') + row)}",
                         transform=ax_echo.get_yaxis_transform(),
                         ha="right", va="center", fontsize=7,
                         fontweight="bold", color="white",
                         bbox=dict(boxstyle="round,pad=0.15",
                                   fc="black", alpha=0.6, lw=0))

        ax = fig.add_subplot(gs[row, 1])
        ax.set_facecolor("white")

        _plot_mpm_poles(ax, tag, mpm_data, POLE_CMAP,
                        label_color="#1e3a8a", dot_size=4)

        ax.set_xlim(time_days[0], time_days[-1])
        ax.set_ylim(d_max, d_min)

        if row < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (days)", labelpad=2)
        ax.set_ylabel("Depth (m)", labelpad=2)

        # Row title — black, regular weight, placed outside the colorbar.
        # The colorbar strip + its tick labels + label ("dB") need ~55pt
        # of clearance to keep the row title from overlapping them.
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(title, rotation=270, labelpad=55,
                      fontsize=7.0, color="black")

        if row == 0:
            ax.set_title("MPM poles", pad=3, fontsize=7.5, fontweight="bold")
        panel_letter(ax, next(letters))

    _save(fig, "F_mpm")


# ════════════════════════════════════════════════════════════════════
# F5 — CW-MLPR (M6 covariance) velocity validation figure
# ════════════════════════════════════════════════════════════════════

def make_F_coherence(Rcoarse, raw_complex_z):
    """
    Standalone coherence-evidence figure (single column):
      (a) Lag-1 temporal coherence vs depth (mean-subtracted)
      (b) Zone coherence distributions
    Belongs to the "evidence for EFZ signal" section, BEFORE methods.
    """
    EFZ_TOP, EFZ_BOT = 600, 1094

    print("  Computing temporal coherence (mean-subtracted)…")
    coh_depths, coh_raw, coh_smooth = compute_temporal_coherence(
        raw_complex_z, Rcoarse, d_min=0.0, d_max=1800.0, smooth_bins=100,
        mean_subtract=True)
    m_layer = (coh_depths >= 50)    & (coh_depths < EFZ_TOP)
    m_efz   = (coh_depths >= EFZ_TOP) & (coh_depths < EFZ_BOT)
    m_noise = (coh_depths >= EFZ_BOT + 100) & (coh_depths <= 1800)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(ps.FIG_W_SINGLE, 3.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.20,
                  width_ratios=[0.75, 0.48],
                  left=0.145, right=0.88, top=0.93, bottom=0.13)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

    # ── (a) Coherence profile ────────────────────────────────────────
    ax = axes[0]
    ax.plot(coh_raw, coh_depths, color="#cbd5e1", lw=0.25, alpha=0.5, zorder=1)
    ax.plot(coh_smooth, coh_depths, color=ps.COL_LAYERS_TXT, lw=1.1, zorder=3)
    ps.zone_shade(ax, efz=True, alpha_efz=0.10, dashed_lines=False)
    for mask, col in [(m_layer, C["layers_txt"]),
                      (m_efz,   C["efz_txt"]),
                      (m_noise, C["bed_txt"])]:
        med = np.median(coh_raw[mask])
        ax.axvline(med, color=col, lw=0.8, ls="--", zorder=4, alpha=0.9)
        mid_d = (coh_depths[mask][0] + coh_depths[mask][-1]) / 2
        ax.text(0.03, mid_d, f"{med:.3f}",
                transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=5.5, color=col,
                fontweight="bold")
    ps.zone_lines(ax, color="black")
    ax.set_xlabel("$|\\gamma|$", labelpad=2)
    ax.set_ylabel("Depth (m)", labelpad=2)
    ax.set_title("Coherence", pad=3)
    ax.set_ylim(1800, 0)
    ax.set_xlim(-0.02, 1.02)
    panel_letter(ax, "a")

    # ── (b) Zone distributions (compact violin) ──────────────────────
    ax = axes[1]
    zone_data   = [coh_raw[m_layer], coh_raw[m_efz], coh_raw[m_noise]]
    zone_colors = [ps.COL_VIOLIN_LAYERS, ps.COL_VIOLIN_EFZ, ps.COL_VIOLIN_NOISE]
    zone_edges  = [C["layers_txt"],  C["efz_txt"],  C["bed_txt"]]
    zone_labels = ["Lyr", "EFZ", "Nse"]
    vp = ax.violinplot(zone_data, positions=[1, 2, 3],
                       showmedians=True, showextrema=False, widths=0.7)
    for body, fc, ec in zip(vp["bodies"], zone_colors, zone_edges):
        body.set_facecolor(fc); body.set_edgecolor(ec); body.set_alpha(0.85)
    vp["cmedians"].set_colors(zone_edges)
    vp["cmedians"].set_linewidth(1.4)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(zone_labels, fontsize=6)
    ax.set_ylabel("$|\\gamma|$", labelpad=2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Dist.", pad=3)
    ps.light_grid(ax, axis="y")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    panel_letter(ax, "b", pad_x=-0.22)

    _save(fig, "F_coherence")


def make_F5(Rcoarse, raw_complex_z, range_img_z, time_days):
    """
    Two compact panels (single column):
      (a) CW-MLPR (M6 multi-lag covariance) velocity scatter coloured by
          SNR, with the plug-flow fit (layered zone only, extrapolated)
      (b) Viterbi layer tracking (independent cross-check)
    """
    from apres.velocity import covariance_velocity_profile

    EFZ_TOP, EFZ_BOT = 600, 1094
    LAKE_DEPTH, LAKE_VEL = 1094.0, 1.84
    D_MIN_PLOT, D_MAX_PLOT = 0, 1800
    LAMBDAC = 0.5608
    WINDOW_M, STEP_M = 20.0, 5.0
    SNR_GATE_DB = 2.0
    N_LAGS = 8

    # ── CW-MLPR (M6) velocity profile ─────────────────────────────────
    # Same configuration as reports/figures/make_vz_m6_profile.py: the
    # validated production estimator (no R² gate, no global mean
    # subtraction), SNR reference from range_img**2 (codebase standard).
    print("  Running M6 covariance_velocity_profile…")
    S = np.asarray(raw_complex_z[:], dtype=np.complex64)
    pow_prof = np.mean(np.array(range_img_z).astype(np.float64) ** 2, axis=1)
    prof = covariance_velocity_profile(
        S, Rcoarse, time_days, lambdac=LAMBDAC,
        depth_min=0.0, depth_max=1800.0,
        window_m=WINDOW_M, step_m=STEP_M, n_lags=N_LAGS,
        snr_gate_db=SNR_GATE_DB,
        mean_subtract_start=600.0, mean_subtract_stop=800.0,
        power_profile=pow_prof)
    v_depths = np.array(prof["depths"])
    m6_v     = np.array(prof["velocities"])
    snr_db   = np.array(prof["snr_db"])
    ok       = np.isfinite(m6_v)
    reliable = ok & (v_depths <= LAKE_DEPTH)

    # ── plug-flow fit: layered zone ONLY, extrapolated into the EFZ ──
    fit_mask = reliable & (v_depths < EFZ_TOP)
    (eps_zz, w_s), pf_cov = np.polyfit(
        v_depths[fit_mask], m6_v[fit_mask], 1, cov=True)
    pf_v = eps_zz * v_depths + w_s
    pf_sig = np.sqrt(pf_cov[0, 0] * v_depths**2 + pf_cov[1, 1]
                     + 2.0 * pf_cov[0, 1] * v_depths)
    pf_upper, pf_lower = pf_v + pf_sig, pf_v - pf_sig
    ice_mask = v_depths <= LAKE_DEPTH

    # residual statistics for the paper text
    efz_mask = reliable & (v_depths >= EFZ_TOP) & (v_depths <= EFZ_BOT)
    rms_layer = float(np.sqrt(np.mean(
        (m6_v[fit_mask] - (eps_zz * v_depths[fit_mask] + w_s))**2)))
    rms_efz = float(np.sqrt(np.mean(
        (m6_v[efz_mask] - (eps_zz * v_depths[efz_mask] + w_s))**2)))
    wH = eps_zz * LAKE_DEPTH + w_s
    wH_sig = float(np.sqrt(pf_cov[0, 0] * LAKE_DEPTH**2 + pf_cov[1, 1]
                           + 2.0 * pf_cov[0, 1] * LAKE_DEPTH))
    print(f"  plug-flow (layered zone, n={fit_mask.sum()}): "
          f"eps_zz = ({eps_zz*1e3:.2f} ± {np.sqrt(pf_cov[0,0])*1e3:.2f})"
          f"e-3 /yr,  w_s = {w_s:.3f} ± {np.sqrt(pf_cov[1,1]):.3f} m/yr")
    print(f"  w(H={LAKE_DEPTH:.0f} m) = {wH:.2f} ± {wH_sig:.2f} m/yr")
    print(f"  RMS residual: layered {rms_layer:.3f} m/yr | "
          f"EFZ (extrapolated, n={efz_mask.sum()}) {rms_efz:.3f} m/yr")
    print(f"  lake excess: {LAKE_VEL - wH:.2f} ± {wH_sig:.2f} m/yr")

    # ── Viterbi optimal-tracking velocities (overlay-only) ──────────────
    # No RMS / R² shown — just the extracted scalar v_z per tracked layer.
    try:
        from scipy.io import loadmat as _loadmat
        _vit_path = ROOT / "output" / "optimal_tracking.mat"
        if _vit_path.exists():
            _vit = _loadmat(str(_vit_path))
            _vit_depths = np.array(_vit["layer_depths"]).flatten()
            _vit_vels   = np.array(_vit["velocities_m_yr"]).flatten()
        else:
            _vit_depths = _vit_vels = np.array([])
    except Exception:
        _vit_depths = _vit_vels = np.array([])

    # ── figure layout: 2 panels (single column) ────────────────────────
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(ps.FIG_W_SINGLE, 3.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.42,
                  width_ratios=[1.05, 1.00],
                  left=0.135, right=0.985, top=0.93, bottom=0.13)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

    def _efz_lines(ax, color="black"):
        ps.zone_lines(ax, color=color)

    # ── (a) CW-MLPR (M6) velocity scatter ────────────────────────────
    ax = axes[0]
    ps.zone_shade(ax, efz=True, alpha_efz=0.08, dashed_lines=False)
    # Filled circles: every window above the lake bed (no SNR gating);
    # empty circles: windows below the bed.
    above = ok & (v_depths <= LAKE_DEPTH)
    below = ok & (v_depths > LAKE_DEPTH)
    sc = ax.scatter(m6_v[above], v_depths[above], c=snr_db[above],
                    cmap="RdBu_r", s=8,
                    vmin=SNR_GATE_DB,
                    vmax=float(np.nanpercentile(snr_db[above], 99)),
                    linewidths=0, zorder=3)
    ax.scatter(m6_v[below], v_depths[below],
               facecolors="none", edgecolors=ps.COL_WIN["noise"], s=8,
               linewidths=0.5, zorder=2)
    # Plug-flow line and confidence band — only above lake.
    # Red accent contrasts the blue SNR-coloured scatter.
    ax.fill_betweenx(v_depths[ice_mask], pf_lower[ice_mask], pf_upper[ice_mask],
                     color=ps.COL_ACCENT, alpha=0.15, linewidths=0, zorder=2)
    ax.plot(pf_v[ice_mask], v_depths[ice_mask], color=ps.COL_ACCENT, lw=1.1, ls="--",
            label="Plug-flow", zorder=4)
    # Lake surface velocity (specular reflection measurement)
    ax.plot(LAKE_VEL, LAKE_DEPTH, marker="*", color=ps.COL_ACCENT,
            markersize=10, markeredgecolor="white", markeredgewidth=0.6,
            zorder=6, label=f"Lake ({LAKE_VEL:.2f} m/yr)")
    _efz_lines(ax)
    add_colorbar(ax, sc, "SNR (dB)")
    ax.set_xlabel("Velocity (m yr$^{-1}$)", labelpad=2)
    ax.set_xlim(-3, 3)
    ax.set_ylabel("Depth (m)", labelpad=2)
    ax.set_title("CW-MLPR", pad=3)
    ax.set_ylim(D_MAX_PLOT, D_MIN_PLOT)
    ax.legend(fontsize=5.5, loc="lower right")
    panel_letter(ax, "a")

    # ── (b) Viterbi layer tracking (independent cross-check) ─────────
    ax = axes[1]
    ps.zone_shade(ax, efz=True, alpha_efz=0.08, dashed_lines=False)
    if _vit_depths.size > 0:
        ax.scatter(_vit_vels, _vit_depths, marker="D", s=8,
                   facecolors="none", edgecolors="#111111", linewidths=0.6,
                   zorder=3, label="Viterbi layers")
    # Same plug-flow fit (from the CW-MLPR layered zone) for comparison
    ax.fill_betweenx(v_depths[ice_mask], pf_lower[ice_mask], pf_upper[ice_mask],
                     color=ps.COL_ACCENT, alpha=0.15, linewidths=0, zorder=2)
    ax.plot(pf_v[ice_mask], v_depths[ice_mask], color=ps.COL_ACCENT, lw=1.1,
            ls="--", label="Plug-flow", zorder=4)
    ax.plot(LAKE_VEL, LAKE_DEPTH, marker="*", color=ps.COL_ACCENT,
            markersize=10, markeredgecolor="white", markeredgewidth=0.6,
            zorder=6, label=f"Lake ({LAKE_VEL:.2f} m/yr)")
    _efz_lines(ax)
    ax.set_xlabel("Velocity (m yr$^{-1}$)", labelpad=2)
    ax.set_xlim(-3, 3)
    ax.set_yticklabels([])
    ax.set_title("Viterbi tracking", pad=3)
    ax.set_ylim(D_MAX_PLOT, D_MIN_PLOT)
    ax.legend(fontsize=5.5, loc="lower right")
    panel_letter(ax, "b")

    # No suptitle — panel titles are self-explanatory
    _save(fig, "F5")


# ════════════════════════════════════════════════════════════════════
# Temporal coherence helper (used by make_F5)
# ════════════════════════════════════════════════════════════════════

def compute_temporal_coherence(raw_complex_z, Rcoarse,
                               d_min=0.0, d_max=1800.0,
                               smooth_bins=100, mean_subtract=False):
    """
    Per-bin lag-1 interferometric coherence magnitude:
        |γ(z)| = |Σ_t S(z,t)·S*(z,t+1)| / sqrt(Σ|S(z,t)|² · Σ|S(z,t+1)|²)
    With mean_subtract=True the complex temporal mean is removed per bin
    first (S' = S − ⟨S⟩_t), suppressing time-invariant artifacts
    (cable reflections, antenna coupling) — matches Table I.
    Returns (depths, coh_raw, coh_smooth).
    """
    idx = np.where((Rcoarse >= d_min) & (Rcoarse <= d_max))[0]
    depths = Rcoarse[idx]
    S  = np.array(raw_complex_z[idx[0]:idx[-1] + 1, :]).astype(np.complex64)
    if mean_subtract:
        S = S - S.mean(axis=1, keepdims=True)
    S1 = S[:, :-1]
    S2 = S[:, 1:]
    num   = np.abs(np.sum(S1 * np.conj(S2), axis=1))
    denom = np.sqrt(np.sum(np.abs(S1) ** 2, axis=1) *
                    np.sum(np.abs(S2) ** 2, axis=1))
    coh = np.where(denom > 0, num / denom, 0.0)
    half = smooth_bins // 2
    coh_smooth = np.array([
        np.median(coh[max(0, i - half):i + half + 1])
        for i in range(len(coh))
    ])
    return depths, coh, coh_smooth


# ════════════════════════════════════════════════════════════════════
# Save helper
# ════════════════════════════════════════════════════════════════════

def _save(fig, tag):
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{tag}.{ext}"
        fig.savefig(str(out))
        print(f"  saved {out.relative_to(ROOT)}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    print("Loading zarr...")
    Rcoarse, range_img_z, raw_complex_z, time_days = load_zarr()

    print("\nComputing MPM results (this may take a few minutes)...")
    mpm_data = compute_mpm_results(step=5)

    print("\nF0: intro figure")
    make_F0(Rcoarse, range_img_z, time_days)

    print("\nFref: combined reference figure (F1/F2/F3)")
    make_F_references(Rcoarse, range_img_z, raw_complex_z, time_days, mpm_data)

    print("\nF_mpm: MPM pole structure figure")
    make_F_mpm(Rcoarse, range_img_z, time_days, mpm_data)

    print("\nF5: coherence + CW-MLPR (M6) velocity validation")
    make_F5(Rcoarse, raw_complex_z, range_img_z, time_days)

    print("\nAll figures saved to", OUT_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
