"""
cs_mmv_F_bands_lcurve.py

Run sliding MMV for the four F_mmv reference bands at the per-band α
picked by the L-curve chord-distance corner (cs_mmv_lcurve_alphas.npz).

Output
------
  data/cs_mmv_F_bands_lcurve.npz
"""

import time
import numpy as np
from pathlib import Path
import zarr

from apres.compressed_sensing import (
    build_dictionary, estimate_lipschitz, estimate_lambda_mmv,
    fista_mmv_bpdn_gpu, _HAS_MPS,
)

# ── Config ────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent.parent
ZARR_PATH    = ROOT / 'data' / 'ImageP2_python.zarr'
ALPHAS_NPZ   = ROOT / 'data' / 'cs_mmv_lcurve_alphas.npz'
OUT_NPZ      = ROOT / 'data' / 'cs_mmv_F_bands_lcurve.npz'

OVERSAMPLE       = 4
MAX_ITER         = 200
TOL              = 1e-5
MIN_AMP_DB       = -30.0
F_LO, F_HI       = 200e6, 400e6
HALO             = 2.0
N_BURSTS_WINDOW  = 30
STRIDE           = 5
MERGE_DIST       = 0.21

BANDS = [
    ('F1', 100.0,  120.0),
    ('F4', 800.0,  820.0),
    ('F3', 1085.0, 1105.0),
    ('F2', 1790.0, 1810.0),
]

# Load picked α per band (chord-distance method — more robust corner)
print(f'Loading L-curve alphas from {ALPHAS_NPZ}...')
ad = np.load(ALPHAS_NPZ)
ALPHA_PER_BAND = {tag: float(ad[f'{tag}_alpha_chord']) for tag, _, _ in BANDS}
for tag in ('F1', 'F4', 'F3', 'F2'):
    print(f'  {tag}: α = {ALPHA_PER_BAND[tag]:.3f}')


print(f'MPS: {_HAS_MPS}', flush=True)
zf = zarr.open(str(ZARR_PATH))
R_full = np.array(zf['Rcoarse'])
t_days = np.array(zf['time_days'])
n_times = len(t_days)


def extract_atoms(X, fine_depths, d_lo, d_hi):
    row_norm = np.linalg.norm(X, axis=1)
    if row_norm.max() < 1e-30:
        return np.array([]), np.zeros((0, X.shape[1]), dtype=complex)
    thresh = row_norm.max() * 10 ** (MIN_AMP_DB / 20)
    active = np.where(row_norm > thresh)[0]
    if len(active) == 0:
        return np.array([]), np.zeros((0, X.shape[1]), dtype=complex)
    active = active[np.argsort(fine_depths[active])]
    clusters = [[active[0]]]
    for i in range(1, len(active)):
        if fine_depths[active[i]] - fine_depths[active[i - 1]] <= MERGE_DIST:
            clusters[-1].append(active[i])
        else:
            clusters.append([active[i]])
    depths, amps = [], []
    for cl in clusters:
        w = row_norm[cl]
        ad_ = float(np.sum(w * fine_depths[cl]) / np.sum(w))
        if d_lo <= ad_ < d_hi:
            depths.append(ad_)
            amps.append(X[cl, :].sum(axis=0))
    if not depths:
        return np.array([]), np.zeros((0, X.shape[1]), dtype=complex)
    return np.array(depths), np.array(amps)


window_starts = np.arange(0, n_times, STRIDE)
results = {}
for tag, d_lo, d_hi in BANDS:
    alpha = ALPHA_PER_BAND[tag]
    proc_lo = d_lo - HALO
    proc_hi = d_hi + HALO
    print(f'\n=== {tag}: {d_lo}-{d_hi} m, α={alpha:.3f} ===', flush=True)
    mask = (R_full >= proc_lo) & (R_full <= proc_hi)
    idx  = np.where(mask)[0]
    d_ax = R_full[mask]
    Y    = np.array(zf['raw_complex'].oindex[idx, :], dtype=np.complex128)

    A, fine_depths = build_dictionary(d_ax, F_LO, F_HI, OVERSAMPLE)
    L = estimate_lipschitz(A)

    band_d, band_a = [], []
    t0 = time.time()
    for wi, s in enumerate(window_starts):
        e = min(s + N_BURSTS_WINDOW, n_times)
        if e - s < 2:
            band_d.append(np.array([]))
            band_a.append(np.zeros((0, e - s), dtype=complex))
            continue
        Y_w = Y[:, s:e]
        lam = estimate_lambda_mmv(A, Y_w, alpha=alpha)
        X_w, _ = fista_mmv_bpdn_gpu(A, Y_w, lam, L=L,
                                      max_iter=MAX_ITER, tol=TOL)
        dep, amp = extract_atoms(X_w, fine_depths, d_lo, d_hi)
        band_d.append(dep)
        band_a.append(amp)
        if (wi + 1) % 100 == 0 or wi + 1 == len(window_starts):
            print(f'  {wi+1}/{len(window_starts)}  ({time.time()-t0:.1f}s)',
                  flush=True)
    print(f'  {tag} done in {time.time()-t0:.1f}s', flush=True)

    results[tag] = dict(
        depths=np.array(band_d, dtype=object),
        amps=np.array(band_a, dtype=object),
        d_lo=d_lo, d_hi=d_hi,
        alpha=alpha,
    )

save = dict(window_starts=window_starts, t_days=t_days,
            n_bursts_window=N_BURSTS_WINDOW, stride=STRIDE)
for tag in ('F1', 'F2', 'F3', 'F4'):
    r = results[tag]
    save[f'{tag}_depths'] = r['depths']
    save[f'{tag}_amps']   = r['amps']
    save[f'{tag}_d_lo']   = r['d_lo']
    save[f'{tag}_d_hi']   = r['d_hi']
    save[f'{tag}_alpha']  = r['alpha']

np.savez_compressed(OUT_NPZ, **save)
print(f'\nSaved {OUT_NPZ}', flush=True)
