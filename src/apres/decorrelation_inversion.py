"""
Multi-band Decorrelation Inversion (MDI) for horizontal-velocity
distributions in glacial ice.

Motivation
----------
Coherence-based horizontal-velocity estimators (v1, Doppler width, …)
collapse the rich information in the decorrelation curve |γ(τ; λ)|
to a single scalar v_h.  When the depth window contains a *mixture*
of scatterer populations (e.g. fast englacial drift + a static
specular layer + a small fraction of slow basal material) any scalar
estimator picks one effective value and discards the rest, often
with a strong bias toward the brightest contributor.

MDI recovers the full *velocity distribution* P(v_h) per depth window
by inverting the multi-band coherence curves.  This:

  • exposes multi-modal scatterer populations (bias-free median + IQR);
  • is robust to bright outliers (they appear as a peak in P(v_h)
    instead of contaminating the median);
  • uses the wavelength diversity of the FMCW chirp (low/full/high
    sub-bands of the 200–400 MHz pulse) to expand the observable
    "spread × time" range and improve conditioning of the inverse.

Mathematical structure (Inverse Laplace on v²)
----------------------------------------------
For one velocity v and one wavelength λ, beam-sweep theory predicts a
Gaussian magnitude coherence:

    |γ_v(τ; λ)| = exp(−2π²·σ_fd(v)²·τ²),
                with σ_fd = √2 · v · σ_θ / λ.

For independent scatterer populations with normalised distribution
P(v), magnitude coherence is *linear* in P(v):

    |γ(τ; λ)| = ∫ P(v) · K(τ, λ; v) dv
              = ∫ P(v) · exp(-4π²·σ_θ²·v²·τ² / λ²) dv .

Under the change of variable u = v²,
    K(τ, λ; v) = exp(-s · u),    s = 4π²·σ_θ²·τ² / λ² ≥ 0,

i.e. the forward operator is a *Laplace transform* of P̃(u) at sample
points s(τ, λ).  Different (τ, λ) pairs sample different s.  Stacking
N_τ lags × N_band wavelengths gives a forward operator

    A ∈ ℝ^(N_τ · N_band) × N_v ,   A[i, j] = K(τ_i, λ_b(i); v_j)

and the inverse problem is

    minimise_{P ≥ 0}  ‖A P − g_obs‖²  +  α · ‖L P‖²

with L = second-difference matrix (Tikhonov smoothness) and the
non-negativity constraint reflecting that P is a probability
distribution.  We solve it with SciPy's NNLS on the augmented system,
which is the standard regularised-NNLS formulation for ill-posed
inverse Laplace problems.

Why "multi-band" matters
------------------------
For fixed σ_θ and a single band, s ∝ τ²/λ² explores only one
direction in (s, τ) space.  Three bands give three "looks" at the
same P(v): two short λ (low/high band) give larger s for the same
τ, sampling slower components, while the long λ (full band) gives
finer s sampling for fast components.  The resulting forward
operator A is better-conditioned and the inversion can resolve
narrower P(v) features than any single band.

API
---
    forward_matrix(v_grid, lags_days, lambdas, sigma_theta) -> A
    invert_pv(gamma_obs, A, v_grid, alpha) -> (P_normalised, residual, smooth)
    invert_window(gamma_by_band, lags_days, v_grid, lambdas_by_band,
                  alpha, sigma_theta) -> dict with P + summary stats

The output `P` is normalised so that ∫ P(v) dv = 1 on the supplied
`v_grid`.  Use `pv_stats(v_grid, P)` for the standard moments and
percentile pairs.

History
-------
The technique was prototyped under the working name "q-space" (after
diffusion-MRI propagator imaging).  It is now packaged here as a
production module with the canonical name *Multi-band Decorrelation
Inversion*.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import nnls
from typing import Mapping, Optional, Sequence, Tuple


# ── Default ApRES beam parameter ─────────────────────────────────────
# Two-way Gaussian beam angular standard deviation. 30° 3-dB beam-width
# → σ_θ ≈ 0.22 rad (matches the rest of the apres package).
DEFAULT_SIGMA_THETA = 0.22


# ── Forward operator ─────────────────────────────────────────────────
def forward_matrix(v_grid: np.ndarray,
                   lags_days: np.ndarray,
                   lambdas: Sequence[float],
                   sigma_theta: float = DEFAULT_SIGMA_THETA) -> np.ndarray:
    """Build the multi-band forward operator A for MDI.

    A[i, j] = exp(-4π²·σ_θ²·v_j²·(τ_i / λ_b(i))²)

    Row order: all τ for band 0, then all τ for band 1, …
    `gamma_obs` rows must be supplied in the same order.

    Parameters
    ----------
    v_grid : (N_v,) — velocity grid (m/yr).  Log-spaced is fine.
    lags_days : (N_τ,) — temporal lag samples (days).
    lambdas : sequence of wavelengths (m), one per band.
    sigma_theta : antenna beam angular std (rad).

    Returns
    -------
    A : (N_band · N_τ, N_v) real ndarray.
    """
    v      = np.asarray(v_grid, dtype=np.float64)
    tau_yr = np.asarray(lags_days, dtype=np.float64) / 365.25
    pieces = []
    for lam in lambdas:
        coef = (2.0 * np.pi * sigma_theta / lam) ** 2     # 4π²σ_θ²/λ²
        K = np.exp(-coef * (v[None, :] ** 2) * (tau_yr[:, None] ** 2))
        pieces.append(K)
    return np.vstack(pieces)


def stack_gamma(gamma_by_band: Mapping[str, np.ndarray],
                band_order: Sequence[str]) -> np.ndarray:
    """Stack |γ(τ; λ)| in the order forward_matrix expects."""
    return np.concatenate([np.asarray(gamma_by_band[b]).ravel()
                           for b in band_order])


# ── Tikhonov smoothness operator ─────────────────────────────────────
def second_diff_matrix(n: int) -> np.ndarray:
    """Second-difference matrix L (shape (n-2, n)).

    (L P)_k = P_{k+2} − 2 P_{k+1} + P_k.
    Penalising ‖L P‖² promotes smooth P(v) without forcing it to zero.
    """
    L = np.zeros((n - 2, n))
    idx = np.arange(n - 2)
    L[idx, idx]     = 1.0
    L[idx, idx + 1] = -2.0
    L[idx, idx + 2] = 1.0
    return L


# ── Noise gate ───────────────────────────────────────────────────────
# Pure noise has a flat magnitude-coherence floor |γ| ≈ 1/√N_t (Rician
# bias of |⟨S S*⟩|).  The only flat column in the forward operator is
# v_min, so NNLS sends that floor to v ≈ 0 and the unit-mass
# normalisation inflates it into a spurious delta at zero velocity.
#
# Noise and a *genuine* static reflector are BOTH flat coherence curves
# — they differ only in AMPLITUDE (≈0.02 vs ≈0.9).  The only reliable
# discriminator is therefore the coherence LEVEL at short lags, not the
# solver mass or the curve shape.  `gate_is_noise` implements that test.
#
# (Empirically validated: a short-lag level gate at ~3/√N_t rejects
#  60/60 pure-noise realisations with zero false v_min spikes, while
#  keeping genuine static reflectors, moving populations, and real
#  shallow-layer / EFZ windows.)

def short_lag_level(gamma_obs: np.ndarray, n_bands: int = 3,
                    n_short: int = 5) -> float:
    """Mean |γ| over the `n_short` shortest lags, averaged across bands.

    `gamma_obs` is the stacked vector (n_bands · n_lag,) in the band
    order used to build the forward operator.  Used as a signal-presence
    metric: a value near the noise floor (1/√N_t) means no resolvable
    coherent scatterers are present.
    """
    g = np.asarray(gamma_obs).ravel()
    n_lag = len(g) // n_bands
    vals = [np.mean(g[b * n_lag: b * n_lag + n_short]) for b in range(n_bands)]
    return float(np.mean(vals))


def gate_is_noise(gamma_obs: np.ndarray, n_eff: int, *,
                  n_bands: int = 3, n_short: int = 5,
                  gate_mult: float = 3.0) -> bool:
    """True if the short-lag coherence level is at the noise floor.

    n_eff : number of independent temporal looks (≈ N_t).  The white-
            noise coherence floor is ≈ 1/√n_eff; we reject below
            `gate_mult` × that floor.

    NOTE: this coherence-level gate is the *fallback* noise test for when
    no clean noise-reference band is available.  Prefer the power-based
    `snr_mask` (computed from absolute received power) when the profile
    extends into a thermal-noise region — it is the standard, more
    interpretable radar SNR criterion.
    """
    level = short_lag_level(gamma_obs, n_bands=n_bands, n_short=n_short)
    return level < gate_mult / np.sqrt(max(n_eff, 1))


# ── Power-based SNR mask (preferred pre-filter) ─────────────────────
def estimate_noise_floor(power_profile: np.ndarray,
                         depths: np.ndarray,
                         noise_band: Tuple[float, float] = (1600.0, 2000.0)
                         ) -> Optional[float]:
    """Median received power in a deep thermal-noise band.

    Returns None if `depths` does not cover the band (so the caller can
    fall back to the coherence gate with a warning).

    power_profile : (N,) received power per depth bin, e.g. ⟨|S(z,t)|²⟩_t.
    depths        : (N,) depth (m) for each bin.
    noise_band    : (lo, hi) depth window assumed to contain only thermal
                    noise (default 1600–2000 m, below the Mercer bed).
    """
    depths = np.asarray(depths)
    power_profile = np.asarray(power_profile)
    sel = (depths >= noise_band[0]) & (depths <= noise_band[1])
    if sel.sum() < 5:
        return None
    return float(np.median(power_profile[sel]))


def snr_mask(power_profile: np.ndarray,
             depths: np.ndarray,
             noise_band: Tuple[float, float] = (1600.0, 2000.0),
             thresh_db: float = 3.0,
             noise_floor: Optional[float] = None,
             ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Boolean keep-mask from received-power SNR.

    The "gold standard" pre-filter: kill noise *before* computing
    coherence.  SNR(z) = 10·log10(P(z) / N0); keep bins with SNR ≥
    `thresh_db`.

    Returns
    -------
    keep   : (N,) bool — True where SNR ≥ thresh_db.
    snr_db : (N,) float — the per-bin SNR in dB (NaN if N0 unavailable).
    n0     : float or None — the noise floor used (None if unavailable).

    If the noise floor cannot be estimated (profile does not reach the
    noise band and `noise_floor` not supplied) a warning is issued and
    an all-True mask is returned (no masking) so the caller can proceed,
    ideally with the coherence-gate fallback enabled instead.
    """
    power_profile = np.asarray(power_profile, dtype=float)
    depths = np.asarray(depths, dtype=float)
    n0 = noise_floor
    if n0 is None:
        n0 = estimate_noise_floor(power_profile, depths, noise_band)
    if n0 is None or n0 <= 0:
        warnings.warn(
            "snr_mask: could not estimate a power noise floor "
            f"(no ≥5 bins in {noise_band} m or N0≤0). "
            "Returning an all-keep mask — no power masking applied. "
            "Enable the coherence-level gate (invert_pv(..., n_eff=N_t)) "
            "as a fallback noise test.",
            RuntimeWarning, stacklevel=2)
        return (np.ones(len(power_profile), dtype=bool),
                np.full(len(power_profile), np.nan), None)
    snr_db = 10.0 * np.log10(power_profile / n0 + 1e-30)
    return snr_db >= thresh_db, snr_db, n0


# ── Inversion (regularised non-negative least squares) ───────────────
def invert_pv(gamma_obs: np.ndarray,
              A: np.ndarray,
              v_grid: np.ndarray,
              alpha: float = 1e-2,
              *,
              n_eff: int | None = None,
              n_bands: int = 3,
              n_short: int = 5,
              gate_mult: float = 3.0) -> Tuple[np.ndarray, float, float]:
    """Solve min_{P ≥ 0} ‖A P − γ_obs‖² + α·‖L P‖² via NNLS.

    Returns (P_normalised_to_unit_mass, residual_norm, smoothness_norm).

    Noise gate
    ----------
    If `n_eff` is given (the number of independent temporal looks, ≈ N_t),
    the window is first tested with `gate_is_noise`.  Windows whose
    short-lag coherence sits at the noise floor return an all-zero P
    (residual/smoothness still computed against the all-zero solution),
    preventing the spurious v ≈ 0 delta that pure noise otherwise
    produces.  With `n_eff=None` (default) the gate is disabled and the
    original behaviour is preserved.
    """
    n_v   = len(v_grid)

    if n_eff is not None and gate_is_noise(
            gamma_obs, n_eff, n_bands=n_bands, n_short=n_short,
            gate_mult=gate_mult):
        L = second_diff_matrix(n_v)
        P0 = np.zeros(n_v)
        res_norm    = float(np.linalg.norm(A @ P0 - np.asarray(gamma_obs).ravel()))
        smooth_norm = 0.0
        return P0, res_norm, smooth_norm

    L     = second_diff_matrix(n_v)
    A_aug = np.vstack([A, np.sqrt(alpha) * L])
    b_aug = np.concatenate([gamma_obs.ravel(), np.zeros(L.shape[0])])

    P, _ = nnls(A_aug, b_aug, maxiter=10_000)
    res_norm    = float(np.linalg.norm(A @ P - gamma_obs.ravel()))
    smooth_norm = float(np.linalg.norm(L @ P))

    # Normalise to ∫ P(v) dv = 1 on the grid (trapezoidal rule).
    if P.sum() > 0:
        v  = np.asarray(v_grid, dtype=np.float64)
        dv = np.diff(v)
        mass = float(np.sum(0.5 * (P[:-1] + P[1:]) * dv))
        if mass > 0:
            P = P / mass
    return P, res_norm, smooth_norm


def l_curve(gamma_obs: np.ndarray, A: np.ndarray, v_grid: np.ndarray,
            alphas: np.ndarray) -> dict:
    """Sweep α and return L-curve diagnostics (Hansen 1992).

    Returns a dict with keys
        'alpha', 'res', 'smooth', 'corner_alpha'.
    """
    res, sm = [], []
    for a in alphas:
        _, r, s = invert_pv(gamma_obs, A, v_grid, alpha=float(a))
        res.append(r); sm.append(s)
    res, sm = np.asarray(res), np.asarray(sm)
    lr, ls = np.log(res + 1e-30), np.log(sm + 1e-30)
    if len(alphas) >= 5:
        dlr  = np.gradient(lr); dls = np.gradient(ls)
        d2lr = np.gradient(dlr); d2ls = np.gradient(dls)
        k = (dlr * d2ls - dls * d2lr) / ((dlr ** 2 + dls ** 2) ** 1.5 + 1e-30)
        ix = int(np.argmax(k))
        corner = float(alphas[ix])
    else:
        corner = float(np.median(alphas))
    return dict(alpha=alphas, res=res, smooth=sm, corner_alpha=corner)


# ── Summary statistics of an inverted P(v) ───────────────────────────
def pv_stats(v_grid: np.ndarray, P: np.ndarray) -> dict:
    """Return mean / median / mode / FWHM and a set of percentile pairs.

    Pairs returned: p005/p995 (99 %), p025/p975 (95 %), p05/p95 (90 %),
                    p16/p84 (68 %).  All in m/yr.
    """
    v   = np.asarray(v_grid, dtype=np.float64)
    keys = ('mean', 'median', 'mode', 'fwhm',
            'p005', 'p025', 'p05', 'p16',
            'p84', 'p95', 'p975', 'p995')
    if P.sum() <= 0:
        return {k: float('nan') for k in keys}
    dv  = np.gradient(v)
    pdf = P / np.sum(P * dv)
    cdf = np.cumsum(pdf * dv)
    cdf /= cdf[-1] + 1e-30
    def _pct(q):
        return float(np.interp(q, cdf, v))
    mode  = float(v[int(np.argmax(pdf))])
    mean  = float(np.sum(v * pdf * dv))
    half  = pdf.max() * 0.5
    above = pdf >= half
    fwhm  = float(v[above].max() - v[above].min()) if above.any() else 0.0
    return dict(
        mean=mean, median=_pct(0.5), mode=mode, fwhm=fwhm,
        p005=_pct(0.005), p025=_pct(0.025),
        p05=_pct(0.05),  p16=_pct(0.16),
        p84=_pct(0.84),  p95=_pct(0.95),
        p975=_pct(0.975), p995=_pct(0.995),
    )


# ── Coherent / decorrelating split ───────────────────────────────────
def split_coherent(v_grid: np.ndarray, P: np.ndarray,
                   v_coh: float = 1.0) -> dict:
    """Separate the unresolved "coherent" pile from the velocity signal.

    The lowest-velocity bins of an MDI solution do not represent a
    physical 0 m/yr population.  They collect (i) specular reflectors
    that stay coherent under lateral motion (|γ|≈1, no beam-sweep
    decorrelation → their v_h is *unobservable*) and (ii) any residual
    noise floor.  Reporting that pile as part of P(v_h) biases the
    velocity statistics toward zero.

    This splits P at `v_coh`:

      f_coherent : mass fraction at v < v_coh — the coherent/unresolved
                   fraction, reported as its own diagnostic (NOT a
                   velocity).
      P_dec      : P restricted to v ≥ v_coh and renormalised to unit
                   mass — the velocity distribution of the *decorrelating*
                   (volume-scatter) component.
      stats      : pv_stats(v_grid, P_dec) — median/percentiles computed
                   on the decorrelating support only, so the coherent
                   pile no longer drags the median down.

    A bin that is entirely coherent/noise (f_coherent ≈ 1) returns an
    all-zero P_dec and NaN stats.

    Returns a dict: {f_coherent, P_dec, **stats}.
    """
    v = np.asarray(v_grid, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    dv = np.gradient(v)
    total = float(np.sum(P * dv))
    if total <= 0:
        out = dict(f_coherent=float('nan'), P_dec=np.zeros_like(P))
        out.update(pv_stats(v, np.zeros_like(P)))
        return out

    lo = v < v_coh
    f_coherent = float(np.sum(P[lo] * dv[lo]) / total)
    P_dec = P.copy()
    P_dec[lo] = 0.0
    dec_mass = float(np.sum(P_dec * dv))
    if dec_mass > 0:
        P_dec = P_dec / dec_mass
    out = dict(f_coherent=f_coherent, P_dec=P_dec)
    out.update(pv_stats(v, P_dec))
    return out


# ── End-to-end: single window ────────────────────────────────────────
def invert_window(gamma_by_band: Mapping[str, np.ndarray],
                  lags_days: np.ndarray,
                  v_grid: np.ndarray,
                  lambdas_by_band: Mapping[str, float],
                  alpha: float = 1e-2,
                  sigma_theta: float = DEFAULT_SIGMA_THETA,
                  n_eff: int | None = None,
                  gate_mult: float = 3.0) -> dict:
    """End-to-end pipeline for a single depth window.

    Stacks γ across bands, builds A, runs invert_pv, returns the
    distribution + statistics.  Pass `n_eff` (≈ number of temporal
    looks) to enable the noise gate — windows at the coherence noise
    floor return an all-zero P and `noise_rejected=True`.
    """
    band_order = list(gamma_by_band.keys())
    A = forward_matrix(v_grid, lags_days,
                       [lambdas_by_band[b] for b in band_order],
                       sigma_theta=sigma_theta)
    g = stack_gamma(gamma_by_band, band_order)
    level = short_lag_level(g, n_bands=len(band_order))
    P, res, sm = invert_pv(g, A, v_grid, alpha=alpha,
                           n_eff=n_eff, n_bands=len(band_order),
                           gate_mult=gate_mult)
    rejected = bool(P.sum() == 0)
    out = dict(P=P, residual=res, smoothness=sm,
               short_lag_level=level, noise_rejected=rejected)
    out.update(pv_stats(v_grid, P))
    return out


# ── Wavelength convention used by the apres package ─────────────────
# Centre frequency of the three sub-bands of the 200–400 MHz pulse,
# converted to wavelength in ice (c_ice ≈ 168 m/μs).  Override if your
# system uses a different chirp.
LAMBDAS_DEFAULT = {
    'low':  168e6 / 250e6,   # 0.672 m (200–300 MHz band → fc = 250 MHz)
    'full': 168e6 / 300e6,   # 0.560 m (200–400 MHz band → fc = 300 MHz)
    'high': 168e6 / 350e6,   # 0.480 m (300–400 MHz band → fc = 350 MHz)
}
