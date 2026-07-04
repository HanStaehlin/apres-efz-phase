#!/usr/bin/env python3
"""
Phase detrending figure for TGARS paper.

Shows why EFZ phase coherence is real and not an artifact:
  Left:   Phase echograms at 3 depths × 3 processing stages
          (raw → carrier-detrended → mean-subtracted)
  Right:  Full-depth coherence profiles (raw vs mean-subtracted)
          + zone violin distributions
"""

import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_style as ps
ps.apply_style()

FIG_W      = ps.FIG_W_DOUBLE
PHASE_CMAP = ps.CMAP_PHASE
AMP_CMAP   = ps.CMAP_AMP
EFZ_TOP, EFZ_BOT = ps.EFZ_TOP, ps.EFZ_BOT
SMOOTH_BINS = 100

# Depth windows — same order and depths as Fref
WINDOWS = [
    ("Shallow layers",    100, 120, ps.COL_WIN["shallow"]),
    ("Echo-Free Zone",    800, 820, ps.COL_WIN["efz"]),
    ("Bed / lake",        1085, 1105, ps.COL_WIN["bed"]),
    ("Below bed (noise)", 1800, 1820, ps.COL_WIN["noise"]),
]


def lag1_coherence(S):
    S1, S2 = S[:, :-1], S[:, 1:]
    num   = np.abs(np.sum(S1 * np.conj(S2), axis=1))
    denom = np.sqrt(np.sum(np.abs(S1)**2, axis=1) *
                    np.sum(np.abs(S2)**2, axis=1))
    return np.where(denom > 0, num / denom, 0.0)


def smooth(arr, half=SMOOTH_BINS // 2):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = np.median(arr[max(0, i-half):i+half+1])
    return out


def main():
    (ROOT / "figs").mkdir(parents=True, exist_ok=True)
    zpath = ROOT / "data" / "ImageP2_python.zarr"
    zroot = zarr.open(str(zpath), mode="r")
    Rcoarse   = np.array(zroot["Rcoarse"]).flatten()
    time_days = np.array(zroot["time_days"]).flatten()
    raw_cpx_z = zroot["raw_complex"]

    # ── Estimate carrier gradient from data (same as visualization_app) ──
    print("Estimating spatial phase gradient from data...")
    idx_fit = np.where((Rcoarse >= 50) & (Rcoarse <= 1800))[0]
    depths_fit = Rcoarse[idx_fit]
    cpx_fit = np.array(raw_cpx_z[idx_fit[0]:idx_fit[-1]+1, :])
    n_t = cpx_fit.shape[1]
    sample_idx = np.linspace(0, n_t - 1, min(50, n_t), dtype=int)

    gradients = []
    for ti in sample_idx:
        col = cpx_fit[:, ti]
        phase_col = np.unwrap(np.angle(col))
        valid = np.isfinite(phase_col) & (np.abs(col) > 1e-20)
        if valid.sum() > 10:
            coeffs = np.polyfit(depths_fit[valid], phase_col[valid], 1)
            gradients.append(coeffs[0])

    grad_mean = np.median(gradients)
    lam_eff = 2 * np.pi / abs(grad_mean)
    print(f"  Estimated gradient: {grad_mean:.4f} rad/m")
    print(f"  Effective spatial period: {lam_eff*100:.1f} cm "
          f"(theoretical λ_c/2 = {168e6/300e6/2*100:.1f} cm)")
    del cpx_fit  # free memory

    # ── Figure layout ───────────────────────────────────────────────
    # Style already applied at module level via ps.apply_style()

    fig = plt.figure(figsize=(FIG_W, 7.0))
    # 4 rows × 3 cols of phase echograms + a thin colorbar strip on the right
    gs_phase = GridSpec(4, 4, figure=fig,
                        width_ratios=[1.0, 1.0, 1.0, 0.035],
                        hspace=0.20, wspace=0.08,
                        left=0.06, right=0.94,
                        top=0.94, bottom=0.06)

    stage_titles = ["Raw phase", "Carrier detrended", "Mean subtracted"]
    letters = iter("abcdefghijkl")
    last_im = None

    for row, (wname, d_min, d_max, _wcol) in enumerate(WINDOWS):
        idx = np.where((Rcoarse >= d_min) & (Rcoarse <= d_max))[0]
        depths = Rcoarse[idx]
        cpx = np.array(raw_cpx_z[idx[0]:idx[-1]+1, :])

        d_mid = (d_min + d_max) / 2
        zoom_lo, zoom_hi = d_mid - 5, d_mid + 5
        extent = [time_days[0], time_days[-1], d_max, d_min]

        # Processing stages
        carrier = grad_mean * depths
        cpx_detrend = cpx * np.exp(-1j * carrier[:, None])
        cpx_meansub = cpx_detrend - np.mean(cpx_detrend, axis=1, keepdims=True)

        stages = [cpx, cpx_detrend, cpx_meansub]

        for col, (stage_cpx, stitle) in enumerate(zip(stages, stage_titles)):
            ax = fig.add_subplot(gs_phase[row, col])
            ph = np.angle(stage_cpx)
            last_im = ax.imshow(ph, aspect="auto", cmap=PHASE_CMAP,
                                vmin=-np.pi, vmax=np.pi,
                                extent=extent, origin="upper")
            ax.set_ylim(zoom_hi, zoom_lo)

            if row == 0:
                ax.set_title(stitle, fontsize=7.5, fontweight="bold", pad=3)
            if row == len(WINDOWS) - 1:
                ax.set_xlabel("Time (days)", labelpad=2)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel("Depth (m)", labelpad=2)
            else:
                ax.set_yticklabels([])

            ltr = next(letters)
            ps.panel_label(ax, ltr)

        # Row label on the right, just inside the plot column — black, regular
        # weight so it doesn't compete with the phase imagery.
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(wname, rotation=270, labelpad=14,
                      fontsize=7.0, color="black")

    # Shared colorbar for all 12 phase panels (all share −π .. +π scale)
    cax = fig.add_subplot(gs_phase[:, 3])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Phase (rad)", fontsize=7.0, labelpad=3)
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels(["$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])
    cb.ax.tick_params(labelsize=6.5, width=0.4)
    cb.outline.set_linewidth(0.4)

    out = ROOT / "figs" / "F_phase_detrend.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"\n  Saved: {out}")
    plt.close(fig)

    # ── Amplitude detrend figure ─────────────────────────────────────
    print("Generating amplitude detrend figure...")
    fig2 = plt.figure(figsize=(FIG_W, 7.0))
    gs_amp = GridSpec(4, 3, figure=fig2,
                      hspace=0.20, wspace=0.08,
                      left=0.06, right=0.97,
                      top=0.94, bottom=0.06)

    stage_titles_amp = ["Raw amplitude", "Carrier detrended", "Mean subtracted"]
    letters2 = iter("abcdefghijkl")

    for row, (wname, d_min, d_max, wcol) in enumerate(WINDOWS):
        idx = np.where((Rcoarse >= d_min) & (Rcoarse <= d_max))[0]
        depths = Rcoarse[idx]
        cpx = np.array(raw_cpx_z[idx[0]:idx[-1]+1, :])

        d_mid = (d_min + d_max) / 2
        zoom_lo, zoom_hi = d_mid - 5, d_mid + 5
        extent = [time_days[0], time_days[-1], d_max, d_min]

        carrier = grad_mean * depths
        cpx_detrend = cpx * np.exp(-1j * carrier[:, None])
        cpx_meansub = cpx_detrend - np.mean(cpx_detrend, axis=1, keepdims=True)

        stages = [cpx, cpx_detrend, cpx_meansub]

        # Shared colour scale across all 3 stages (raw sets the range)
        amp_raw_db = 20.0 * np.log10(np.abs(cpx) + 1e-15)
        vmin = np.percentile(amp_raw_db, 2)
        vmax = np.percentile(amp_raw_db, 98)

        for col, (stage_cpx, stitle) in enumerate(zip(stages, stage_titles_amp)):
            ax = fig2.add_subplot(gs_amp[row, col])
            amp_db = 20.0 * np.log10(np.abs(stage_cpx) + 1e-15)
            ax.imshow(amp_db, aspect="auto", cmap=AMP_CMAP,
                      vmin=vmin, vmax=vmax,
                      extent=extent, origin="upper")
            ax.set_ylim(zoom_hi, zoom_lo)

            if row == 0:
                ax.set_title(stitle, fontsize=7.5, fontweight="bold", pad=3)
            if row == len(WINDOWS) - 1:
                ax.set_xlabel("Time (days)", labelpad=2)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel("Depth (m)", labelpad=2)
            else:
                ax.set_yticklabels([])

            ltr = next(letters2)
            ps.panel_label(ax, ltr)

        # Row label on right
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(wname, rotation=270, labelpad=20,
                      fontsize=6.5, color=wcol, fontweight="bold")

    out2 = ROOT / "figs" / "F_amp_detrend.pdf"
    fig2.savefig(out2, dpi=300)
    fig2.savefig(out2.with_suffix(".png"), dpi=200)
    print(f"  Saved: {out2}")
    plt.close(fig2)

    # ── Coherence comparison across processing stages ────────────────
    print("Generating coherence stage comparison figure...")

    D_MIN, D_MAX = 0.0, 1800.0
    idx_all = np.where((Rcoarse >= D_MIN) & (Rcoarse <= D_MAX))[0]
    depths_all = Rcoarse[idx_all]
    cpx_all = np.array(raw_cpx_z[idx_all[0]:idx_all[-1]+1, :]).astype(np.complex64)

    carrier_all = grad_mean * depths_all
    cpx_detrend_all = cpx_all * np.exp(-1j * carrier_all[:, None])
    cpx_meansub_all = cpx_detrend_all - np.mean(cpx_detrend_all, axis=1, keepdims=True)

    stages_coh = [
        (cpx_all,         "Raw"),
        (cpx_detrend_all, "Carrier detrended"),
        (cpx_meansub_all, "Mean subtracted"),
    ]

    # Zone masks
    m_layer = (depths_all >= 50)      & (depths_all < EFZ_TOP)
    m_efz   = (depths_all >= EFZ_TOP) & (depths_all < EFZ_BOT)
    m_noise = (depths_all >= EFZ_BOT + 100) & (depths_all <= D_MAX)

    zone_colors = [ps.COL_LAYERS_TXT, ps.COL_EFZ_TXT, ps.COL_BED_TXT]
    zone_fill_colors = [ps.COL_VIOLIN_LAYERS, ps.COL_VIOLIN_EFZ, ps.COL_VIOLIN_NOISE]
    zone_labels = ["Layered", "EFZ", "Noise"]

    fig3, axes = plt.subplots(
        2, 3, figsize=(FIG_W, 5.0),
        gridspec_kw={"hspace": 0.35, "wspace": 0.12,
                     "left": 0.08, "right": 0.97,
                     "top": 0.93, "bottom": 0.09}
    )

    for col, (S, stitle) in enumerate(stages_coh):
        coh = lag1_coherence(S)
        coh_sm = smooth(coh)

        # ── (top row) Coherence profile ──────────────────────────────
        ax = axes[0, col]
        ax.plot(coh, depths_all, color="#e2e8f0", lw=0.2, alpha=0.4, zorder=1)
        ax.plot(coh_sm, depths_all, color="#1e3a8a", lw=1.1, zorder=3)

        for mask, zc, zlabel in zip([m_layer, m_efz, m_noise], zone_colors, zone_labels):
            med = np.median(coh[mask])
            mid_d = (depths_all[mask][0] + depths_all[mask][-1]) / 2
            ax.axvline(med, color=zc, lw=0.8, ls="--", alpha=0.9, zorder=4)
            ax.text(0.03, mid_d, f"{med:.3f}",
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=5.5,
                    color=zc, fontweight="bold")

        ps.zone_lines(ax)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(D_MAX, 0)
        ax.set_title(stitle, fontsize=7.5, fontweight="bold", pad=3)
        if col == 0:
            ax.set_ylabel("Depth (m)", labelpad=2)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("$|\\gamma|$", labelpad=2)
        ps.light_grid(ax, axis="x")

        ltr = chr(ord("a") + col)
        ps.panel_label(ax, ltr)

        # ── (bottom row) Violin distributions ────────────────────────
        ax2 = axes[1, col]
        zone_data = [coh[m_layer], coh[m_efz], coh[m_noise]]
        vp = ax2.violinplot(zone_data, positions=[1, 2, 3],
                            showmedians=True, showextrema=False, widths=0.7)
        for body, zfill, zc in zip(vp["bodies"], zone_fill_colors, zone_colors):
            body.set_facecolor(zfill)
            body.set_edgecolor(zc)
            body.set_alpha(0.6)
        vp["cmedians"].set_colors(zone_colors)
        vp["cmedians"].set_linewidth(1.4)

        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels(zone_labels, fontsize=6)
        ax2.set_ylim(-0.02, 1.02)
        ps.light_grid(ax2, axis="y")
        if col == 0:
            ax2.set_ylabel("$|\\gamma|$", labelpad=2)
        else:
            ax2.set_yticklabels([])

        ltr2 = chr(ord("d") + col)
        ps.panel_label(ax2, ltr2)

    out3 = ROOT / "figs" / "F_coherence_stages.pdf"
    fig3.savefig(out3, dpi=300)
    fig3.savefig(out3.with_suffix(".png"), dpi=200)
    print(f"  Saved: {out3}")
    plt.close(fig3)


if __name__ == "__main__":
    main()
