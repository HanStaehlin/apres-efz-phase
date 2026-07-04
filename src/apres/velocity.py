#!/usr/bin/env python3
"""
Phase-based Velocity Estimation for ApRES Deep Layers

Instead of tracking individual layers (which is noisy at depth), this
approach finds the dominant phase-slope in the SVD-denoised echogram
at each depth window.  The slope in the (depth x time) plane directly
yields the vertical velocity.

Two methods are implemented:

1. **Phase-slope** -- For each depth bin, fit a line to the unwrapped
   temporal phase of the SVD-denoised complex signal.  The slope gives
   the velocity via  v = (dphi/dt) * lambda_c / (4*pi).  Robust median
   over the depth window.

2. **Phase-R² Radon** -- For each depth bin, compute the optimal
   velocity from linear regression on unwrapped phase (mathematically
   identical to phase-slope).  Filter by R² > 0.5 and take the
   amplitude-weighted median.  Also produces a semblance map showing
   per-bin R² across candidate velocities for visualisation.

Usage
-----
    python radon_velocity.py \\
        --data data/apres/ImageP2_python.mat \\
        --svd-components 3 \\
        --depth-min 200 --depth-max 1094 \\
        --window 10 \\
        --output output/apres/radon_velocity.json

Author: SiegVent2023 project
"""

import numpy as np
from scipy.io import loadmat
import json
import argparse
import time as time_mod
from typing import Optional


# ====================================================================
#  MDL rank selection
# ====================================================================
def mdl_rank(singular_values: np.ndarray, n_samples: int) -> int:
    """
    Estimate the signal subspace dimension via the Minimum Description
    Length (MDL) criterion (Wax & Kailath, 1985).

    For a (p x N) matrix with singular values s_1 >= s_2 >= ... >= s_p,
    MDL selects the rank k that minimises

        MDL(k) = -N·(p-k)·log(geometric_mean / arithmetic_mean)
                 + ½·k·(2p-k)·log(N)

    where the means are taken over s_{k+1}^2 ... s_p^2  (the "noise"
    eigenvalues).  The first term rewards fit; the second penalises
    model complexity.

    Returns at least 1 to avoid a zero-rank reconstruction.
    """
    s2 = singular_values ** 2
    p = len(s2)
    N = n_samples

    best_k, best_mdl = 1, np.inf
    for k in range(0, p):
        noise = s2[k:]          # eigenvalues attributed to noise
        m = len(noise)
        if m == 0:
            break
        arith = noise.mean()
        # geometric mean via log-sum (numerically stable)
        geom = np.exp(np.mean(np.log(noise + 1e-30)))
        # likelihood term
        ll = -N * m * np.log(geom / (arith + 1e-30) + 1e-30)
        # penalty term (number of free parameters)
        penalty = 0.5 * k * (2 * p - k) * np.log(N)
        val = ll + penalty
        if val < best_mdl:
            best_mdl = val
            best_k = k

    return max(1, best_k)


# ====================================================================
#  Method 1: Phase-slope velocity
# ====================================================================
def phase_slope_velocity(
    strip_complex: np.ndarray,
    time_days: np.ndarray,
    lambdac: float,
) -> dict:
    """
    Estimate velocity from the temporal phase slope at each depth bin,
    then take the robust median over the window.

    For a reflector moving at velocity v, the radar phase evolves as
        phi(t) = phi_0 + (4*pi / lambda_c) * v * t
    so  v = dphi/dt * lambda_c / (4*pi).
    """
    n_bins, n_times = strip_complex.shape
    t_yr = (time_days - time_days[0]) / 365.25

    velocities = np.full(n_bins, np.nan)
    r2_values = np.full(n_bins, np.nan)

    for i in range(n_bins):
        z = strip_complex[i, :]
        amp = np.abs(z)

        # Skip very low amplitude bins
        if np.median(amp) < 1e-6:
            continue

        phase = np.unwrap(np.angle(z))

        try:
            coeffs = np.polyfit(t_yr, phase, 1)
            slope = coeffs[0]  # rad/yr

            phase_pred = np.polyval(coeffs, t_yr)
            ss_res = np.sum((phase - phase_pred) ** 2)
            ss_tot = np.sum((phase - phase.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-30)

            v = slope * lambdac / (4 * np.pi)
            velocities[i] = v
            r2_values[i] = r2
        except (np.linalg.LinAlgError, ValueError):
            continue

    valid = np.isfinite(velocities)
    if valid.sum() == 0:
        return {'best_v': np.nan, 'median_r2': 0.0, 'n_good': 0}

    good = valid & (r2_values > 0.5)
    if good.sum() >= 3:
        best_v = float(np.median(velocities[good]))
        med_r2 = float(np.median(r2_values[good]))
        n_good = int(good.sum())
    else:
        best_v = float(np.median(velocities[valid]))
        med_r2 = float(np.median(r2_values[valid]))
        n_good = int(valid.sum())

    return {
        'best_v': best_v,
        'median_r2': med_r2,
        'n_good': n_good,
    }


def phase_slope_velocity_err(
    strip_complex: np.ndarray,
    time_days: np.ndarray,
    lambdac: float,
) -> dict:
    """
    Phase-slope velocity with a per-bin Cramér–Rao error bar.

    This is the principled replacement for the R²>0.5 gate.  R² measures
    *linearity* and so conflates thermal SNR with constancy of motion: a
    high-SNR bin whose v_z genuinely varies in time has large, REAL residuals
    and a low R², so it is wrongly rejected.  Nothing forces v_z to be
    constant, so an R² gate penalises signal.

    Instead we report, per depth bin:

      v            slope velocity  (m/yr), unchanged from phase_slope_velocity
      sigma_v_meas the CRLB precision (m/yr) — the irreducible thermal floor
      sigma_v_fit  the full-fit residual SE (m/yr) — includes any non-linearity
      eta          sigma_v_fit / sigma_v_meas = sqrt(reduced chi^2)
      r2           the old metric, kept for comparison only

    Math
    ----
    For a tone z_k = A·exp(i φ_k)+n_k at times t_k with per-burst power SNR
    ρ = A²/σ_n², the phase variance floor is σ_φ² = 1/(2ρ) (the per-sample
    phase CRLB).  Because OLS is the MLE for a line in white Gaussian phase
    noise, the slope estimator ATTAINS the bound:

        var(ω̂) = σ_φ² / S_tt,   S_tt = Σ_k (t_k - t̄)²,   ω = (4π/λ) v
        σ_v,meas = (λ/4π) · σ_φ / sqrt(S_tt).

    σ_φ is recovered DATA-DRIVEN (no SNR needed) from successive phase
    differences — the Allan variance at lag 1.  With φ_k = φ_0 + ω t_k +
    m_k + ε_k (m = smooth real motion, ε = thermal):

        Δφ_k = ω·Δt + Δm_k + Δε_k,   Var(Δφ) ≈ 2σ_φ²   (Δm_k ≪ Δε at lag 1)

    so σ̂_φ² = ½·Var(Δφ - mean).  We use a MAD form so cycle slips (rare
    ±2π jumps after unwrap) do not corrupt the floor:

        σ̂_φ = 1.4826 · median|Δφ - median(Δφ)| / sqrt(2).

    The full-fit residual SE σ_v,fit uses ss_res/(N-2) instead of σ̂_φ²; it
    equals σ_v,meas when the motion is linear and inflates when it is not.
    η = σ_v,fit/σ_v,meas is the physical, SNR-normalised model-adequacy flag.

    Caveat: σ_v,meas treats thermal noise as white across bursts (correct).
    For the *average* rate of time-varying motion this is the right sampling
    uncertainty, with η flagging when one rate is a lossy summary.  Treating
    the motion itself as a nuisance random process would call for a
    Newey–West/HAC inflation; the white-thermal floor is the conservative,
    interpretable error bar reported here.

    Validity threshold: the CRLB is attained only ABOVE the phase-estimation
    threshold SNR (≈10 dB, ρ≈10).  Below it the unwrap suffers cycle slips, the
    estimator leaves the linear regime and the true scatter explodes while the
    robust σ_φ stays at the floor — so σ_v,meas UNDERSTATES the error for
    low-SNR bins.  Gate on thermal SNR (not R²) before trusting the bar.

    The aggregate ``best_v`` is the inverse-variance (1/σ_v,meas²) weighted
    mean over finite bins; ``best_v_err`` is its propagated error scaled by the
    Birge ratio sqrt(reduced χ²) so it reflects real bin-to-bin scatter rather
    than the over-tight independent-thermal-bins formal error.
    """
    n_bins, n_times = strip_complex.shape
    t_yr = (time_days - time_days[0]) / 365.25
    k_conv = lambdac / (4.0 * np.pi)                  # rad/yr -> m/yr
    S_tt = float(np.sum((t_yr - t_yr.mean()) ** 2))

    v = np.full(n_bins, np.nan)
    sigma_v_meas = np.full(n_bins, np.nan)
    sigma_v_fit = np.full(n_bins, np.nan)
    eta = np.full(n_bins, np.nan)
    r2 = np.full(n_bins, np.nan)

    if n_times < 3 or S_tt <= 0:
        return {
            'v': v, 'sigma_v_meas': sigma_v_meas, 'sigma_v_fit': sigma_v_fit,
            'eta': eta, 'r2': r2, 'best_v': np.nan, 'best_v_err': np.nan,
            'n_good': 0,
        }

    for i in range(n_bins):
        zc = strip_complex[i, :]
        if np.median(np.abs(zc)) < 1e-6:
            continue
        phase = np.unwrap(np.angle(zc))
        try:
            coeffs = np.polyfit(t_yr, phase, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue
        slope = coeffs[0]                              # rad/yr
        v[i] = slope * k_conv

        resid = phase - np.polyval(coeffs, t_yr)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((phase - phase.mean()) ** 2))
        r2[i] = 1.0 - ss_res / (ss_tot + 1e-30)

        # data-driven thermal phase noise: robust lag-1 Allan estimate
        dphi = np.diff(phase)
        mad = np.median(np.abs(dphi - np.median(dphi)))
        sigma_phi = 1.4826 * mad / np.sqrt(2.0)
        if sigma_phi <= 0:                             # degenerate; fall back to OLS
            sigma_phi = np.sqrt(ss_res / max(n_times - 2, 1))
        sigma_v_meas[i] = k_conv * sigma_phi / np.sqrt(S_tt)

        sigma_phi_fit = np.sqrt(ss_res / (n_times - 2))
        sigma_v_fit[i] = k_conv * sigma_phi_fit / np.sqrt(S_tt)
        eta[i] = sigma_v_fit[i] / (sigma_v_meas[i] + 1e-30)

    good = np.isfinite(v) & np.isfinite(sigma_v_meas) & (sigma_v_meas > 0)
    if good.sum() >= 1:
        w = 1.0 / sigma_v_meas[good] ** 2
        best_v = float(np.sum(w * v[good]) / np.sum(w))
        formal_err = np.sqrt(1.0 / np.sum(w))
        # Birge-ratio (reduced-χ²) inflation: pure inverse-variance weighting
        # assumes every bin is independent and thermal-limited, which yields an
        # absurdly tight error over many bins.  Scale by sqrt(reduced χ²) so the
        # aggregate honestly reflects bin-to-bin scatter (η>1, spatial
        # correlation, residual non-linearity).
        if good.sum() >= 2:
            chi2_red = np.sum(w * (v[good] - best_v) ** 2) / (good.sum() - 1)
            birge = np.sqrt(max(chi2_red, 1.0))
        else:
            birge = 1.0
        best_v_err = float(formal_err * birge)
        n_good = int(good.sum())
    else:
        best_v, best_v_err, n_good = np.nan, np.nan, 0

    return {
        'v': v,
        'sigma_v_meas': sigma_v_meas,
        'sigma_v_fit': sigma_v_fit,
        'eta': eta,
        'r2': r2,
        'best_v': best_v,
        'best_v_err': best_v_err,
        'n_good': n_good,
    }


# ====================================================================
#  Lag-coherence estimators (real-data / MDI pipeline)
# ====================================================================
LAMBDA_ICE = 0.56    # carrier wavelength in ice (m)


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


def lag_coherence_complex(S, times_days, n_out=40, max_lag_days=None, max_frac=1/3):
    """Complex per-lag cross-coherence γ̂(τ), preserving the vz phase ramp.

    The phase of γ̂(τ) rotates linearly with lag at rate ω_vz = 4π·vz/λ,
    so the slope of angle(γ̂) vs τ directly gives v_z.

    Returns
    -------
    lags_days : (n_lags,) array
    C_cx      : (n_lags,) complex array
    """
    n  = len(S)
    dt = float(np.median(np.diff(times_days)))
    if max_lag_days is None:
        max_lag_days = min(int(n * max_frac) * dt, 104.0)

    k_vals = _lag_k_vals(n, dt, n_out, max_lag_days, max_frac=max_frac)
    lags   = k_vals * dt
    C_cx   = np.zeros(len(k_vals), dtype=complex)
    for j, k in enumerate(k_vals):
        s1 = S[:n - k];  s2 = S[k:]
        num   = np.mean(s1 * np.conj(s2))
        denom = np.sqrt(np.mean(np.abs(s1) ** 2) * np.mean(np.abs(s2) ** 2))
        C_cx[j] = num / (denom + 1e-30)
    return lags, C_cx


def lag_coherence_spatial(S, times_days, n_out=40, max_lag_days=None,
                          max_frac=1/3):
    """Depth-windowed coherence: ensemble-average over BOTH depth and
    time, magnitude taken AFTER summing the per-bin cross-products.

    For a depth-resolved window S of shape (n_z, n_t):
        |γ̂(k)| = |Σ_z Σ_t S_z(t)·S_z*(t+k)|
                 / √(Σ_z Σ_t |S_z(t)|² · Σ_z Σ_t |S_z(t+k)|²)

    This differs from `lag_coherence(S.mean(axis=0), …)` (which coherently
    averages the complex profile over depth *first*): here only the
    diagonal z=z' cross-products enter, so the inter-bin carrier ramp
    cancels *within* each bin instead of destructively across bins.  It
    therefore (i) uses the full window power instead of a ~1 % residual,
    (ii) preserves specular layers (their coherence is real, to be split
    off afterwards), and (iii) is the textbook ergodic estimator.

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


def vz_from_phase_slope(lags_days, C_cx, lambda_ice=LAMBDA_ICE):
    """Estimate v_z (m/yr) from the linear phase slope of complex coherence.

    The cross-correlation carries a phase ramp exp(i·ω_vz·τ) from vertical
    motion, where ω_vz = 4π·vz/λ (rad/day).  A weighted linear fit to
    unwrap(angle(γ̂(τ))) vs τ recovers ω_vz, hence v_z.

    Returns
    -------
    vz_myr : float, estimated v_z in m/yr
    phi    : (n_lags,) unwrapped phase array used for the fit
    """
    phi    = np.unwrap(np.angle(C_cx))
    w      = np.abs(C_cx)
    coeffs = np.polyfit(lags_days, phi, 1, w=w)
    omega  = coeffs[0]                          # rad / day
    vz_m_per_day = omega * lambda_ice / (4.0 * np.pi)
    return float(vz_m_per_day * 365.25), phi


# ====================================================================
#  Method 2: Phase-R² Radon
# ====================================================================
def radon_velocity(
    strip_complex: np.ndarray,
    time_days: np.ndarray,
    lambdac: float,
    v_candidates: np.ndarray,
    dz: float = None,  # kept for API compat, unused
) -> dict:
    """
    Velocity estimation via phase-R² Radon transform.

    Works in the *unwrapped* phase domain — the same domain as the
    phase-slope method.  For each depth bin, the optimal velocity is

        v_i = (Σ φ·t / Σ t²) · λ / (4π)

    which is identical to the phase-slope linear-regression answer.
    A per-bin R² measures fit quality, and the final velocity is the
    R²-filtered, amplitude-weighted median — matching the phase-slope
    aggregation.

    For visualisation, a per-bin-normalised R² semblance map is also
    returned (each bin votes equally for its best velocity), so the
    Radon heatmap remains informative.

    Bins below the 10th-percentile amplitude are excluded (phase
    is undefined for noise).
    """
    n_bins, n_times = strip_complex.shape
    t_yr = (time_days - time_days[0]) / 365.25

    # --- Amplitude gating ---
    amp = np.abs(strip_complex)
    median_amp = np.median(amp, axis=1)
    noise_floor = np.percentile(median_amp, 10)
    good = median_amp > noise_floor
    if good.sum() < 3:
        good = np.ones(n_bins, dtype=bool)
    n_good = int(good.sum())
    weights = median_amp[good]

    # --- Unwrapped phase (same as phase-slope) ---
    phases = np.unwrap(np.angle(strip_complex[good]), axis=1)  # (n_good, n_times)

    # Centre time and phase so the intercept drops out
    t_c = t_yr - t_yr.mean()
    ph_c = phases - phases.mean(axis=1, keepdims=True)

    # --- Per-bin optimal velocity (= phase-slope answer) ---
    pt = ph_c @ t_c                        # Σ(φ·t) per bin, (n_good,)
    tt = np.dot(t_c, t_c)                  # Σ(t²),  scalar
    ss_tot = np.sum(ph_c ** 2, axis=1)     # Σ(φ²),  (n_good,)

    per_bin_slope = pt / tt                                   # rad/yr
    per_bin_v = per_bin_slope * lambdac / (4.0 * np.pi)       # m/yr
    per_bin_r2 = pt ** 2 / (tt * ss_tot + 1e-30)              # R² ∈ [0,1]
    np.clip(per_bin_r2, 0.0, 1.0, out=per_bin_r2)

    # --- Velocity pick: R²-filtered weighted median ---
    high_r2 = per_bin_r2 > 0.5
    if high_r2.sum() >= 3:
        sel_v = per_bin_v[high_r2]
        sel_w = weights[high_r2]
    else:
        sel_v = per_bin_v
        sel_w = weights

    # Weighted median
    order = np.argsort(sel_v)
    cumw = np.cumsum(sel_w[order])
    best_v = float(sel_v[order][np.searchsorted(cumw, cumw[-1] / 2.0)])

    # --- Semblance map (for visualisation) ---
    # Per-bin normalised R²: each bin peaks at 1.0 at its optimal v
    slopes = 4.0 * np.pi * v_candidates / lambdac
    numerator = 2.0 * np.outer(pt, slopes) - (slopes ** 2)[None, :] * tt
    r2_map = numerator / (ss_tot[:, None] + 1e-30)
    np.clip(r2_map, 0.0, 1.0, out=r2_map)
    # Normalise each row so every bin contributes equally to the map
    row_max = r2_map.max(axis=1, keepdims=True)
    r2_norm = r2_map / (row_max + 1e-30)
    semblance = (r2_norm.T @ weights)
    semblance_norm = semblance / (semblance.max() + 1e-30)

    med = np.median(semblance)

    return {
        'best_v': best_v,
        'semblance': semblance_norm,
        'peak_snr': float(semblance.max() / (med + 1e-30)),
    }


# ====================================================================
#  Method 3: Multi-lag covariance velocity (M6) — validated estimator
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

    Returns per-bin v (m/yr); NaN where undefined.  Validated (see
    experiments/vz_estimator_comparison.py and vz_timesub_validate.py): agrees
    1:1 with independent Viterbi layer tracking; ~4× lower EFZ RMSE than the
    phase-slope baseline; unbiased to ≥5 m/yr.
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
    """Multi-lag covariance vertical velocity (M6) for one depth window.

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
    """SNR-gated multi-lag-covariance (M6) vertical-velocity profile.

    The validated production v_z profile.  Per depth bin computes the multi-lag
    covariance velocity (no unwrap); each window takes the robust median over all
    finite per-bin velocities and is flagged ``reliable`` where the window's
    received-power SNR exceeds ``snr_gate_db`` (matches experiments/
    vz_profile_to1800.py — low-SNR windows are returned but flagged unreliable
    rather than discarded, so callers can fade them).  Deliberately: NO R² gate
    (biases low-SNR EFZ velocities high) and NO mean subtraction (over-corrects —
    see experiments/vz_timesub_validate.py).

    Parameters
    ----------
    raw_complex : (n_bins, n_times) complex echogram.
    Rcoarse     : (n_bins,) depth axis (m).
    time_days   : (n_times,) burst times (days).
    snr_gate_db : received-power SNR floor for a window to be flagged reliable.
    power_profile : optional (n_bins,) per-bin power for the SNR reference.  If
        omitted, ``mean(|raw_complex|^2)`` is used.  Pass ``mean(range_img**2)``
        to match the codebase SNR standard (experiments/vz_profile_to1800.py) —
        ``range_img`` has a higher noise floor, so its SNR is ~11 dB lower and
        the gate behaves consistently with the reference figures.

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
                   'method': 'M6 multi-lag covariance + median, SNR-flagged'},
    }


# ====================================================================
#  Main pipeline
# ====================================================================
def radon_velocity_profile(
    data_path: str,
    depth_min: float = 200.0,
    depth_max: float = 1094.0,
    window_m: float = 10.0,
    step_m: float = 5.0,
    svd_components: int = 3,
    svd_mode: str = 'none',  # 'none', 'local', 'global'
    v_min: float = -0.5,
    v_max: float = 1.5,
    n_velocities: int = 200,
    verbose: bool = True,
) -> dict:
    """Estimate vertical velocity profile using phase-slope and Radon.

    svd_mode controls denoising:
      - 'none':   use raw complex data (no SVD)
      - 'local':  apply SVD independently within each sliding window
      - 'global': apply SVD to the entire depth region first (original)
    """
    t0 = time_mod.time()

    if verbose:
        print("=" * 70)
        print(f"SLOPE-BASED VELOCITY ESTIMATION -- svd_mode={svd_mode}")
        print("=" * 70)

    # -- Load --
    if verbose:
        print(f"\nLoading {data_path} ...")
    
    if data_path.endswith('.mat'):
        mat = loadmat(data_path)
        raw_complex = np.array(mat['RawImageComplex'])
        Rcoarse = mat['Rcoarse'].flatten()
        time_days = mat['TimeInDays'].flatten()
        lambdac = float(mat.get('lambdac', np.array([0.5608])).flatten()[0])
        del mat
    elif data_path.endswith('.zarr'):
        import zarr
        r = zarr.open(data_path, 'r')
        raw_complex = np.array(r['raw_complex'])
        Rcoarse = np.array(r['Rcoarse']).flatten()
        time_days = np.array(r['time_days']).flatten()
        lambdac = 0.5608
        del r
    else:
        raise ValueError(f"Unknown data format: {data_path}")

    n_bins, n_times = raw_complex.shape
    dz = float(Rcoarse[1] - Rcoarse[0])

    if verbose:
        print(f"  Data: {n_bins} bins x {n_times} times")
        print(f"  dz = {dz:.4f} m, lambda_c = {lambdac:.4f} m")
        print(f"  Time: {time_days[-1] - time_days[0]:.1f} days "
              f"({(time_days[-1] - time_days[0])/365.25:.2f} yr)")

    # -- Extract region --
    idx_start = np.searchsorted(Rcoarse, depth_min)
    idx_end = np.searchsorted(Rcoarse, depth_max)
    region = raw_complex[idx_start:idx_end, :]
    depths_region = Rcoarse[idx_start:idx_end]
    n_region = len(depths_region)

    if verbose:
        print(f"  Region: {n_region} bins, "
              f"{depths_region[0]:.1f} -- {depths_region[-1]:.1f} m")

    # -- SVD denoising --
    if svd_mode == 'global':
        if verbose:
            print(f"\nGlobal SVD denoising (k={svd_components}) ...")
        U, S, Vh = np.linalg.svd(region, full_matrices=False)
        total_energy = np.sum(S ** 2)
        kept_energy = np.sum(S[:svd_components] ** 2)
        if verbose:
            print(f"  Kept energy: {kept_energy/total_energy*100:.1f}%")
        S_trunc = np.zeros_like(S)
        S_trunc[:svd_components] = S[:svd_components]
        denoised_global = U @ np.diag(S_trunc) @ Vh
    elif svd_mode == 'none':
        if verbose:
            print("\nNo SVD denoising -- using raw complex data")
        denoised_global = None
    elif svd_mode == 'local':
        if verbose:
            print(f"\nLocal SVD denoising (k={svd_components} per window)")
        denoised_global = None
    elif svd_mode == 'mdl':
        if verbose:
            print("\nLocal SVD denoising with MDL rank selection")
        denoised_global = None
    else:
        raise ValueError(f"Unknown svd_mode: {svd_mode}")

    # -- Windows --
    window_bins = max(1, int(round(window_m / dz)))
    step_bins = max(1, int(round(step_m / dz)))
    v_candidates = np.linspace(v_min, v_max, n_velocities)
    window_starts = list(range(0, n_region - window_bins + 1, step_bins))
    n_windows = len(window_starts)

    if verbose:
        print(f"\nWindow: {window_m:.1f} m = {window_bins} bins, "
              f"step: {step_m:.1f} m = {step_bins} bins")
        print(f"Windows: {n_windows}")
        print(f"Velocity search: [{v_min:.2f}, {v_max:.2f}] m/yr, "
              f"{n_velocities} candidates\n")

    # -- Results --
    centers = []
    ps_vel, ps_r2, ps_ng = [], [], []
    rd_vel, rd_snr, rd_sem = [], [], []
    svd_ranks = []

    for wi, i in enumerate(window_starts):
        cd = float(depths_region[i + window_bins // 2])
        centers.append(cd)

        if svd_mode == 'global':
            strip_c = denoised_global[i : i + window_bins, :]
            svd_ranks.append(svd_components)
        elif svd_mode in ('local', 'mdl'):
            # SVD within this window only
            win_raw = region[i : i + window_bins, :]
            U_w, S_w, Vh_w = np.linalg.svd(win_raw, full_matrices=False)
            if svd_mode == 'mdl':
                k = mdl_rank(S_w, n_times)
            else:
                k = svd_components
            svd_ranks.append(k)
            S_t = np.zeros_like(S_w)
            S_t[:k] = S_w[:k]
            strip_c = U_w @ np.diag(S_t) @ Vh_w
        else:  # 'none'
            strip_c = region[i : i + window_bins, :]
            svd_ranks.append(0)

        # 1. Phase-slope
        ps = phase_slope_velocity(strip_c, time_days, lambdac)
        ps_vel.append(ps['best_v'])
        ps_r2.append(ps['median_r2'])
        ps_ng.append(ps['n_good'])

        # 2. Phase-R² Radon
        rd = radon_velocity(strip_c, time_days, lambdac, v_candidates, dz)
        rd_vel.append(rd['best_v'])
        rd_snr.append(rd['peak_snr'])
        rd_sem.append(rd['semblance'].tolist())

        if verbose and ((wi + 1) % 20 == 0 or wi + 1 == n_windows):
            rank_str = f"  k={k}" if svd_mode in ('local', 'mdl') else ""
            print(f"  [{wi+1:4d}/{n_windows}] d={cd:7.1f}m  "
                  f"phase={ps['best_v']:+.4f}  "
                  f"radon={rd['best_v']:+.4f}{rank_str}")

    elapsed = time_mod.time() - t0

    # -- Nye reference (Auto-fit to confident shallow data) --
    centers_arr = np.array(centers)
    ps_arr = np.array(ps_vel)
    r2_arr = np.array(ps_r2)
    
    # Fit depth range < 1080m, confident R2 > 0.5
    fit_mask = np.isfinite(ps_arr) & (centers_arr < 1080.0) & (r2_arr > 0.5)
    
    if fit_mask.sum() >= 2:
        nye_sl, nye_int = np.polyfit(centers_arr[fit_mask], ps_arr[fit_mask], 1)
        if verbose:
            print(f"\n  Nye fit: intercept={nye_int:.4f}, slope={nye_sl:.6f} "
                  f"(based on {fit_mask.sum()} points < 1080m)")
    else:
        # Fallback if no valid points
        nye_int, nye_sl = 0.0453, 0.000595
        if verbose:
            print("\n  Nye fit: Using fallback parameters (insufficient points)")
            
    nye_v = [nye_int + nye_sl * d for d in centers]

    # -- Summary --
    if verbose:
        print(f"Done in {elapsed:.1f} s")
        arr_ps = np.array(ps_vel)
        arr_rd = np.array(rd_vel)
        arr_nye = np.array(nye_v)
        arr_k = np.array(svd_ranks)

        vp = np.isfinite(arr_ps)
        if vp.any():
            resid = arr_ps[vp] - arr_nye[vp]
            print(f"\n  Phase-slope: {vp.sum()}/{len(arr_ps)} valid, "
                  f"v=[{arr_ps[vp].min():.4f}, {arr_ps[vp].max():.4f}], "
                  f"RMS vs Nye={np.sqrt((resid**2).mean()):.4f}")

        print(f"  Radon: v=[{arr_rd.min():.4f}, {arr_rd.max():.4f}]")

        if svd_mode == 'mdl':
            print(f"  MDL ranks: min={arr_k.min()}, max={arr_k.max()}, "
                  f"median={np.median(arr_k):.0f}")

    return {
        'svd_mode': svd_mode,
        'svd_components': svd_components,
        'window_m': window_m, 'step_m': step_m,
        'depth_spacing_m': dz, 'lambdac': lambdac,
        'depths': centers,
        'phase_slope_velocities': ps_vel,
        'phase_slope_r2': ps_r2,
        'phase_slope_n_good': ps_ng,
        'radon_velocities': rd_vel,
        'radon_snrs': rd_snr,
        'radon_semblances': rd_sem,
        'svd_ranks': svd_ranks,
        'nye_velocities': nye_v,
        'v_candidates': v_candidates.tolist(),
        'elapsed_s': elapsed,
    }


# ====================================================================
#  Plotting
# ====================================================================
def plot_results(results: dict, save_path: Optional[str] = None):
    """Plot velocity profiles from phase-slope and Radon methods."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping plot")
        return

    depths = np.array(results['depths'])
    nye = np.array(results['nye_velocities'])
    ps  = np.array(results['phase_slope_velocities'])
    r2  = np.array(results['phase_slope_r2'])
    rd  = np.array(results['radon_velocities'])
    vc  = np.array(results['v_candidates'])
    sem = np.array(results['radon_semblances'])
    ranks = np.array(results.get('svd_ranks', []))

    if results['svd_mode'] == 'mdl' and len(ranks) > 0:
        fig, axes = plt.subplots(1, 4, figsize=(24, 8), sharey=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)

    # 1: Phase-slope
    ax = axes[0]
    vp = np.isfinite(ps)
    if vp.any():
        sc = ax.scatter(ps[vp], depths[vp], c=r2[vp], cmap='viridis',
                        s=20, vmin=0, vmax=1, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='R²', shrink=0.6)
    ax.plot(nye, depths, 'r--', lw=1.5, label='Nye')
    ax.set_xlabel('Velocity (m/yr)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Phase-slope')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # 2: MDL Rank Profile (Optional)
    if results['svd_mode'] == 'mdl' and len(ranks) > 0:
        ax = axes[1]
        ax.plot(ranks, depths, 'k-', lw=1.5, alpha=0.7)
        ax.scatter(ranks, depths, c='k', s=10)
        ax.set_xlabel('Selected SVD Rank (k)')
        ax.set_title('MDL Optimal Subspace Rank')
        ax.grid(True, alpha=0.3)
        axes_idx_offset = 1
    else:
        axes_idx_offset = 0

    # 3: Radon semblance
    ax = axes[1 + axes_idx_offset]
    if sem.size:
        ax.imshow(sem, aspect='auto',
                  extent=[vc[0], vc[-1], depths[-1], depths[0]],
                  cmap='inferno', interpolation='bilinear')
        ax.plot(nye, depths, 'c--', lw=1.5, label='Nye')
        ax.plot(rd, depths, 'w.', ms=3, alpha=0.6, label='pick')
    ax.set_xlabel('Velocity (m/yr)')
    ax.set_title('Radon semblance')
    ax.legend(fontsize=8)

    # 4: Comparison
    ax = axes[2 + axes_idx_offset]
    ax.plot(nye, depths, 'r--', lw=2, label='Nye', zorder=3)
    gp = vp & (r2 > 0.7)
    if gp.any():
        ax.scatter(ps[gp], depths[gp], c='steelblue', s=15, alpha=0.7,
                   label=f'Phase R²>0.7 (n={gp.sum()})', edgecolors='none')
    ax.scatter(rd, depths, c='darkorange', s=10, alpha=0.5,
              label='Radon', edgecolors='none', marker='s')
    ax.set_xlabel('Velocity (m/yr)')
    ax.set_title('Comparison')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = save_path or '/tmp/radon_velocity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {out}")
    plt.close()


# ====================================================================
#  CLI
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Slope-based velocity estimation for ApRES data')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--plot', default=None)
    parser.add_argument('--svd-components', type=int, default=3)
    parser.add_argument('--svd-mode', choices=['none', 'local', 'global', 'mdl'],
                        default='none',
                        help='SVD denoising mode: none, local (per window fixed k), '
                             'global (entire region fixed k), or mdl (auto-k via MDL)')
    parser.add_argument('--depth-min', type=float, default=200.0)
    parser.add_argument('--depth-max', type=float, default=1094.0)
    parser.add_argument('--window', type=float, default=10.0)
    parser.add_argument('--step', type=float, default=5.0)
    parser.add_argument('--v-min', type=float, default=-0.5)
    parser.add_argument('--v-max', type=float, default=1.5)
    parser.add_argument('--n-velocities', type=int, default=200)

    args = parser.parse_args()

    results = radon_velocity_profile(
        data_path=args.data,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        window_m=args.window,
        step_m=args.step,
        svd_components=args.svd_components,
        svd_mode=args.svd_mode,
        v_min=args.v_min,
        v_max=args.v_max,
        n_velocities=args.n_velocities,
    )

    # Save JSON (skip semblances — too large)
    import os
    out_path = args.output or 'output/apres/radon_velocity.json'
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    save = {k: v for k, v in results.items() if k != 'radon_semblances'}
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved results to {out_path}")

    plot_results(results, save_path=args.plot)


if __name__ == '__main__':
    main()
