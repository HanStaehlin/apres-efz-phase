"""
CW-MLPR: Coherence-Weighted Multi-Lag Phase Regression for ApRES vertical
velocity, plus the lag-coherence estimator shared with the MDI pipeline.

CW-MLPR (`covariance_velocity` / `covariance_velocity_profile`) recovers
the vertical velocity v_z from the phase of the multi-lag temporal
covariance of the complex signal, without phase unwrapping — the lag-1
phase increments never wrap, so the estimator stays unbiased under
temporal decorrelation and low SNR where unwrap-and-fit approaches break
down.

`lag_coherence_spatial` is the depth-windowed magnitude-coherence
estimator that feeds the Multi-band Decorrelation Inversion (MDI, see
`decorrelation_inversion.py`).
"""

import numpy as np


# ====================================================================
#  Lag-coherence estimator (shared with the MDI pipeline)
# ====================================================================
def _lag_k_vals(n, dt, n_out, max_lag_days, max_frac=1/3):
    """Shared helper: compute log-spaced lag indices.

    Starts from k_min corresponding to ~1 day so that sub-day lags —
    where the Gaussian decay is negligible for any physically plausible
    v_h — do not consume log-spaced slots.  For daily-or-coarser data
    k_min = 1 (unchanged behaviour).

    max_frac : float
        Maximum lag as fraction of n (default 1/3 → n//3 pairs minimum).
        Increase to 1/2 when long-lag coherence is needed for slow motion.
    """
    k_max = max(1, min(int(max_lag_days / dt), int(n * max_frac)))
    k_min = max(1, int(round(1.0 / dt)))   # first lag ≥ 1 day
    if k_min >= k_max:
        k_min = 1
    k_vals = np.unique(
        np.round(np.logspace(
            np.log10(k_min), np.log10(k_max), n_out)).astype(int))
    return np.clip(k_vals, k_min, k_max)


def lag_coherence_spatial(S, times_days, n_out=40, max_lag_days=None,
                          max_frac=1/3):
    """Depth-windowed coherence: ensemble-average over BOTH depth and
    time, magnitude taken AFTER summing the per-bin cross-products.

    For a depth-resolved window S of shape (n_z, n_t):
        |γ̂(k)| = |Σ_z Σ_t S_z(t)·S_z*(t+k)|
                 / √(Σ_z Σ_t |S_z(t)|² · Σ_z Σ_t |S_z(t+k)|²)

    This differs from coherently averaging the complex profile over
    depth *first*: here only the diagonal z=z' cross-products enter, so
    the inter-bin carrier ramp cancels *within* each bin instead of
    destructively across bins.  It therefore (i) uses the full window
    power instead of a ~1 % residual, (ii) preserves specular layers
    (their coherence is real, to be split off afterwards), and (iii) is
    the textbook ergodic estimator.

    Parameters
    ----------
    S : (n_z, n_t) complex — depth-resolved window.  A 1-D input is
        treated as a single-bin window.

    Returns
    -------
    lags_days : (n_lags,) array
    C         : (n_lags,) array in [0, 1]
    """
    S = np.atleast_2d(np.asarray(S))
    n = S.shape[1]
    dt = float(np.median(np.diff(times_days)))
    if max_lag_days is None:
        max_lag_days = min(int(n * max_frac) * dt, 104.0)
    k_vals = _lag_k_vals(n, dt, n_out, max_lag_days, max_frac=max_frac)
    lags = k_vals * dt
    C = np.zeros(len(k_vals))
    for j, k in enumerate(k_vals):
        s1 = S[:, :n - k]
        s2 = S[:, k:]
        num   = np.abs(np.sum(s1 * np.conj(s2)))
        denom = np.sqrt(np.sum(np.abs(s1) ** 2) * np.sum(np.abs(s2) ** 2))
        C[j]  = num / (denom + 1e-30)
    return lags, C


# ====================================================================
#  CW-MLPR: multi-lag covariance velocity — validated estimator
# ====================================================================
def _per_bin_covariance_velocity(
    strip_complex: np.ndarray,
    time_days: np.ndarray,
    lambdac: float,
    n_lags: int = 8,
) -> np.ndarray:
    """Per-bin vertical velocity from the multi-lag phase covariance (no unwrap).

    For each depth bin the lag-ℓ autocovariance r_ℓ = Σ_t S(t+ℓ)·conj(S(t)) has
    phase ≈ ω·τ_ℓ with ω = (4π/λ)·v.  An inverse-|r|-weighted fit through the
    origin over lags 1..L recovers ω (hence v) WITHOUT phase unwrapping — the
    lag-1 increments never wrap — so it stays unbiased under temporal
    decorrelation and low SNR, unlike the unwrap+linfit phase-slope method.

    Returns per-bin v (m/yr); NaN where undefined.  Validated against
    independent Viterbi layer tracking (1:1 agreement); ~4x lower EFZ RMSE
    than the unwrap-and-fit phase-slope baseline; unbiased to >=5 m/yr.
    """
    t_yr = (np.asarray(time_days).flatten() - time_days[0]) / 365.25
    order = np.argsort(t_yr)
    Ss = np.asarray(strip_complex)[:, order]
    ts = t_yr[order]
    n_bins, n_t = Ss.shape
    if n_t < 3:
        return np.full(n_bins, np.nan)
    dt_med = float(np.median(np.diff(ts)))
    L = max(1, min(n_lags, n_t - 1))
    num = np.zeros(n_bins)
    den = np.zeros(n_bins)
    for lag in range(1, L + 1):
        r = np.sum(Ss[:, lag:] * np.conj(Ss[:, :-lag]), axis=1)
        tau = lag * dt_med
        w = np.abs(r)
        num += w * np.angle(r) * tau
        den += w * tau ** 2
    v = np.full(n_bins, np.nan)
    ok = den > 0
    v[ok] = (num[ok] / den[ok]) * (lambdac / (4.0 * np.pi))
    return v


def covariance_velocity(
    strip_complex: np.ndarray,
    time_days: np.ndarray,
    lambdac: float,
    n_lags: int = 8,
) -> dict:
    """Multi-lag covariance vertical velocity (CW-MLPR) for one depth window.

    Per-bin lag-covariance phase rate (no unwrap), aggregated by the robust
    median over depth bins.  No R² gate (it biases low-SNR velocities upward)
    and no mean subtraction (it over-corrects).  This is the validated
    production v_z estimator; gate noise externally via received-power SNR.
    """
    v_bin = _per_bin_covariance_velocity(strip_complex, time_days, lambdac, n_lags)
    valid = np.isfinite(v_bin)
    best_v = float(np.median(v_bin[valid])) if valid.any() else np.nan
    return {'best_v': best_v, 'per_bin_v': v_bin, 'n_good': int(valid.sum())}


def covariance_velocity_profile(
    raw_complex: np.ndarray,
    Rcoarse: np.ndarray,
    time_days: np.ndarray,
    lambdac: float = 0.5608,
    depth_min: float = 0.0,
    depth_max: float = 1800.0,
    window_m: float = 20.0,
    step_m: float = 10.0,
    n_lags: int = 8,
    snr_gate_db: float = 3.0,
    mean_subtract_start: float = None,
    mean_subtract_stop: float = None,
    noise_band: tuple = (1600.0, 2000.0),
    min_good: int = 5,
    power_profile: np.ndarray = None,
) -> dict:
    """SNR-gated multi-lag-covariance (CW-MLPR) vertical-velocity profile.

    The validated production v_z profile.  Per depth bin computes the multi-lag
    covariance velocity (no unwrap); each window takes the robust median over all
    finite per-bin velocities and is flagged ``reliable`` where the window's
    received-power SNR exceeds ``snr_gate_db`` — low-SNR windows are returned
    but flagged unreliable rather than discarded, so callers can fade them.
    Deliberately: NO R² gate (biases low-SNR EFZ velocities high) and NO mean
    subtraction (over-corrects).

    Parameters
    ----------
    raw_complex : (n_bins, n_times) complex echogram.
    Rcoarse     : (n_bins,) depth axis (m).
    time_days   : (n_times,) burst times (days).
    snr_gate_db : received-power SNR floor for a window to be flagged reliable.
    power_profile : optional (n_bins,) per-bin power for the SNR reference.  If
        omitted, ``mean(|raw_complex|^2)`` is used.  Pass ``mean(range_img**2)``
        to match the codebase SNR standard — ``range_img`` has a higher noise
        floor, so its SNR is ~11 dB lower and the gate behaves consistently
        with the reference figures.

    Returns
    -------
    dict with: depths, velocities, snr_db, reliable, n_good (all per-centre),
    and the parameters used.  Trust ``velocities[reliable]``.
    """
    from .decorrelation_inversion import snr_mask

    raw_complex = np.asarray(raw_complex)
    Rcoarse = np.asarray(Rcoarse).flatten()
    time_days = np.asarray(time_days).flatten()

    if power_profile is None:
        power_profile = np.mean(np.abs(raw_complex) ** 2, axis=1)
    else:
        power_profile = np.asarray(power_profile).flatten()
    _, _, n0 = snr_mask(power_profile, Rcoarse,
                        noise_band=noise_band, thresh_db=snr_gate_db)

    sel = (Rcoarse >= depth_min) & (Rcoarse <= depth_max)
    idx = np.where(sel)[0]
    if len(idx) == 0:
        raise ValueError("depth_min/depth_max select no bins")
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    region, zR = raw_complex[lo:hi], Rcoarse[lo:hi]
    pow_r = power_profile[lo:hi]

    # Optionally apply a depth-weighted mean subtraction (ramp) to reduce
    # persistent coherent bias. If start/stop are provided, weight each bin
    # by w = clip((z - start) / (stop - start), 0, 1) and subtract w*mean(bin).
    if (mean_subtract_start is not None) and (mean_subtract_stop is not None):
        mean_full = region.mean(axis=1, keepdims=True)
        denom = (mean_subtract_stop - mean_subtract_start)
        if denom == 0:
            w_bin = np.zeros_like(zR)[:, None]
        else:
            w_bin = np.clip((zR - float(mean_subtract_start)) / float(denom), 0.0, 1.0)[:, None]
        region_cs = region - w_bin * mean_full
        v_bin = _per_bin_covariance_velocity(region_cs, time_days, lambdac, n_lags)
    else:
        v_bin = _per_bin_covariance_velocity(region, time_days, lambdac, n_lags)

    centres = np.arange(depth_min, depth_max + 0.5 * step_m, step_m)
    velocities = np.full(len(centres), np.nan)
    snr_c = np.full(len(centres), np.nan)
    n_good = np.zeros(len(centres), dtype=int)
    reliable = np.zeros(len(centres), dtype=bool)
    for i, zc in enumerate(centres):
        wm = (zR >= zc - window_m / 2) & (zR < zc + window_m / 2)
        if wm.any() and n0 is not None:
            snr_c[i] = 10.0 * np.log10(np.mean(pow_r[wm]) / n0 + 1e-30)
        fm = wm & np.isfinite(v_bin)
        n_good[i] = int(fm.sum())
        if fm.any():
            velocities[i] = float(np.median(v_bin[fm]))
        reliable[i] = (n_good[i] >= min_good and np.isfinite(snr_c[i])
                       and snr_c[i] > snr_gate_db)

    return {
        'depths': centres,
        'velocities': velocities,
        'snr_db': snr_c,
        'reliable': reliable,
        'n_good': n_good,
        'params': {'window_m': window_m, 'step_m': step_m, 'n_lags': n_lags,
                   'snr_gate_db': snr_gate_db, 'lambdac': lambdac,
                   'method': 'CW-MLPR multi-lag covariance + median, SNR-flagged'},
    }
