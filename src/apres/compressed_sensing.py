#!/usr/bin/env python3
"""
Compressed Sensing Range Estimation for ApRES Echograms (FISTA / BPDN).

Recovers a sparse set of reflector positions and complex amplitudes from
the range-compressed ApRES echo by solving the L1-regularised least-squares
problem (Basis Pursuit Denoising):

    min_x  (1/2)||y - Ax||_2^2  +  λ ||x||_1

where:
    y  =  observed complex range profile (N depth bins)
    A  =  oversampled DFT dictionary  (N × M,  M = oversample * N)
    x  =  sparse reflector amplitudes on a fine depth grid
    λ  =  regularisation parameter controlling sparsity

The solver is FISTA (Fast Iterative Shrinkage-Thresholding Algorithm),
which achieves O(1/k²) convergence via Nesterov momentum — considerably
faster than vanilla ISTA's O(1/k).

Key advantage over FFT + CLEAN:
    The FFT imposes a Rayleigh resolution limit  ΔR = c_ice / (2·BW) ≈ 0.42 m.
    CLEAN partially overcomes this (greedy L0 heuristic), but compressed
    sensing solves the *convex relaxation* of the exact L0 problem with
    provable recovery guarantees under the Restricted Isometry Property.

Usage
-----
    # Synthetic validation (super-resolution test):
    python compressed_sensing.py --test

    # Real data, single depth window in the EFZ:
    python compressed_sensing.py \\
        --data data/apres/ImageP2_python.zarr \\
        --depth-start 900 --depth-end 910 \\
        --compare-clean

    # Full processing for a depth range:
    python compressed_sensing.py \\
        --data data/apres/ImageP2_python.zarr \\
        --depth-start 800 --depth-end 1000 \\
        --oversample 4 --lambda-alpha 0.1

References
----------
    Beck, A. & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
        algorithm for linear inverse problems. SIAM J. Imaging Sci., 2(1).
    Candès, E.J., Romberg, J. & Tao, T. (2006). Robust uncertainty
        principles: exact signal reconstruction from highly incomplete
        frequency information. IEEE Trans. Inf. Theory, 52(2).
    Donoho, D.L. (2006). Compressed sensing. IEEE Trans. Inf. Theory, 52(4).

Author: SiegVent2023 project
"""

import argparse
import numpy as np
from scipy.signal.windows import blackman
from pathlib import Path
import time as timer
from typing import Optional, Tuple, List, Dict, Union

try:
    from apres.io import fmcw_load, ApRESData
except ImportError:
    fmcw_load = None   # type: ignore
    ApRESData = None   # type: ignore

# ── ApRES radar parameters (shared with clean_deconvolution.py) ─────
c_ice = 168e6        # speed of light in ice (m/s)
f_start = 200e6      # chirp start frequency (Hz)
f_end = 400e6        # chirp end frequency (Hz)
fc = (f_start + f_end) / 2
BW = f_end - f_start
RAYLEIGH_LIMIT = c_ice / (2 * BW)  # ≈ 0.42 m

SUBBANDS = {
    'full': (f_start, f_end),
    'low':  (f_start, fc),
    'high': (fc, f_end),
}


# ════════════════════════════════════════════════════════════════════
#  Step 1: Oversampled DFT Dictionary
# ════════════════════════════════════════════════════════════════════

def build_dictionary(
    depth_axis: np.ndarray,
    f_lo: float,
    f_hi: float,
    oversample: int = 4,
    n_freq: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the oversampled DFT dictionary matrix for FMCW de-ramp sensing.

    The FMCW radar measures a beat signal whose frequency is proportional
    to the two-way travel time (and hence depth) of each reflector. The
    range-compressed echo at depth bin z_i for a reflector at depth d_j is:

        A[i, j] = Σ_f  w(f) · exp(-j·2π·f · 2(z_i - d_j)/c_ice)

    where w(f) is the Blackman window, summed over the chirp bandwidth.

    An oversampled dictionary places candidate reflector depths on a grid
    that is `oversample` times finer than the standard FFT depth spacing,
    enabling super-resolution recovery.

    Parameters
    ----------
    depth_axis : array, shape (N,)
        Depth values (m) of the observed range bins.
    f_lo, f_hi : float
        Frequency band edges (Hz).
    oversample : int
        Oversampling factor. The fine grid has M = oversample × N points.
        Higher = finer grid = better resolution, but larger matrix.
    n_freq : int
        Number of frequency samples for the chirp integration.

    Returns
    -------
    A : complex array, shape (N, M)
        Dictionary matrix. Each column is the expected range profile
        for a unit-amplitude reflector at the corresponding fine-grid depth.
    fine_depths : array, shape (M,)
        The fine depth grid (m).
    """
    N = len(depth_axis)
    dz = np.mean(np.diff(depth_axis))
    fine_dz = dz / oversample

    # Fine depth grid spans the same range as depth_axis
    d_min = depth_axis[0]
    d_max = depth_axis[-1]
    fine_depths = np.arange(d_min, d_max + fine_dz * 0.5, fine_dz)
    M = len(fine_depths)

    # Frequency axis with Blackman window
    freqs = np.linspace(f_lo, f_hi, n_freq)
    win = blackman(n_freq)

    # Two-way travel times
    tau_obs = 2.0 * depth_axis / c_ice       # (N,)
    tau_fine = 2.0 * fine_depths / c_ice      # (M,)

    # Build dictionary: A[i, j] = Σ_f w(f) * exp(-j 2π f (τ_obs[i] - τ_fine[j]))
    # This is a matrix of size (N, M) — can be large, so we build it
    # efficiently using matrix multiplication.
    #
    # Define E[i,f] = exp(-j 2π f[f] τ_obs[i])   shape (N, n_freq)
    #        F[j,f] = exp(-j 2π f[f] τ_fine[j])   shape (M, n_freq)
    #
    # Then A[i,j] = Σ_f w[f] E[i,f] * conj(F[j,f])
    #             = Σ_f w[f] exp(-j 2π f[f] (τ_obs[i] - τ_fine[j]))  ✓
    #             = (E ⊙ w) @ F^H   →  (N, n_freq) @ (n_freq, M) = (N, M)

    phase_obs = -2j * np.pi * tau_obs[:, np.newaxis] * freqs[np.newaxis, :]   # (N, n_freq)
    E = np.exp(phase_obs)  # (N, n_freq)

    phase_fine = -2j * np.pi * tau_fine[:, np.newaxis] * freqs[np.newaxis, :]  # (M, n_freq)
    F = np.exp(phase_fine)  # (M, n_freq)

    # A = (E ⊙ w) @ F^H = (N, n_freq) @ (n_freq, M) = (N, M)
    Ew = E * win[np.newaxis, :]   # (N, n_freq), element-wise multiply with window
    A = Ew @ F.conj().T           # (N, M)

    # Normalise columns to unit L2 norm (standard for CS dictionaries)
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1.0
    A /= col_norms

    return A, fine_depths


# ════════════════════════════════════════════════════════════════════
#  Step 2: FISTA Solver
# ════════════════════════════════════════════════════════════════════

def _prox_complex_l1(z: np.ndarray, threshold: float) -> np.ndarray:
    """
    Proximal operator for the complex L1 norm.

    prox_{λ||·||_1}(z) = z · max(0, 1 - λ/|z|)

    This is the complex soft-thresholding operator. It shrinks the
    magnitude of each element towards zero while preserving its phase.

    Parameters
    ----------
    z : complex array
        Input vector.
    threshold : float or array
        Shrinkage threshold (= λ / L). May be scalar or per-element
        (for weighted L1 problems).

    Returns
    -------
    Complex array of same shape as z.
    """
    mag = np.abs(z)
    # Avoid division by zero: where mag == 0, the output is 0 anyway
    scale = np.maximum(0.0, 1.0 - threshold / np.maximum(mag, 1e-30))
    return z * scale


def estimate_lipschitz(A: np.ndarray, n_iter: int = 30) -> float:
    """
    Estimate the Lipschitz constant L = ||A^H A||_2 via power iteration.

    The Lipschitz constant of the gradient ∇f(x) = A^H(Ax - y) is the
    spectral norm of A^H A (its largest eigenvalue). This determines
    the optimal step size for FISTA: step = 1/L.

    Parameters
    ----------
    A : complex array, shape (N, M)
    n_iter : int
        Number of power iterations (30 is usually sufficient).

    Returns
    -------
    L : float
        Estimated Lipschitz constant.
    """
    M = A.shape[1]
    # Random starting vector
    x = np.random.randn(M) + 1j * np.random.randn(M)
    x /= np.linalg.norm(x)

    for _ in range(n_iter):
        # y = A^H A x
        y = A.conj().T @ (A @ x)
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-30:
            break
        x = y / norm_y

    # Rayleigh quotient
    L = float(np.real(x.conj() @ (A.conj().T @ (A @ x))))
    return L


def estimate_lambda(
    A: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Estimate a good regularisation parameter λ.

    The maximum useful λ is λ_max = ||A^H y||_∞, which is the smallest λ
    for which the all-zeros solution is optimal (no reflectors recovered).

    A good default is  λ = α · λ_max  where α ∈ (0, 1):
        α = 0.5  →  very sparse (few strong reflectors)
        α = 0.1  →  moderately sparse (default)
        α = 0.01 →  recover weak reflectors (risk of noise artefacts)

    Parameters
    ----------
    A : complex array, shape (N, M)
    y : complex array, shape (N,)
    alpha : float
        Fraction of λ_max. Default 0.1.

    Returns
    -------
    lam : float
        Recommended regularisation parameter.
    """
    correlator = A.conj().T @ y  # (M,) or (M, B)
    lam_max = np.max(np.abs(correlator), axis=0)
    return alpha * lam_max


def fista_bpdn(
    A: np.ndarray,
    y: np.ndarray,
    lam: float,
    L: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for BPDN.

    Solves:  min_x  (1/2)||y - Ax||_2^2  +  λ ||x||_1

    via accelerated proximal gradient descent with Nesterov momentum.

    The algorithm alternates between:
        1. Gradient step:     z_k = x_k - (1/L) A^H (A x_k - y)
        2. Proximal step:     x_{k+1} = prox_{λ/L ||·||_1}(z_k)
        3. Momentum update:   x_k is extrapolated using Nesterov's scheme

    Parameters
    ----------
    A : complex array, shape (N, M)
        Dictionary matrix.
    y : complex array, shape (N,)
        Observed signal.
    lam : float
        Regularisation parameter (λ).
    L : float or None
        Lipschitz constant of the gradient. If None, estimated automatically.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on relative change in x.
    verbose : bool
        Print convergence info every 50 iterations.

    Returns
    -------
    x : complex array, shape (M,)
        Recovered sparse vector.
    info : dict
        Convergence information:
            'n_iter': number of iterations run
            'converged': whether tolerance was reached
            'objective': final objective value
            'objectives': list of objective values per iteration
            'n_nonzero': number of non-zero entries in x
    """
    N, M = A.shape

    # Lipschitz constant
    if L is None:
        L = estimate_lipschitz(A)
    step = 1.0 / L

    # Precompute A^H y (used every iteration)
    AHy = A.conj().T @ y   # (M,)

    # Initialisation
    x_k = np.zeros(M, dtype=np.complex128)
    z_k = x_k.copy()   # momentum variable
    t_k = 1.0

    objectives = []
    threshold = lam * step

    for k in range(max_iter):
        # Gradient at z_k:  ∇f(z_k) = A^H(A z_k - y) = A^H A z_k - A^H y
        grad = A.conj().T @ (A @ z_k) - AHy

        # Proximal gradient step
        x_new = _prox_complex_l1(z_k - step * grad, threshold)

        # Nesterov momentum
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2))
        momentum = (t_k - 1.0) / t_new
        z_k = x_new + momentum * (x_new - x_k)

        # Convergence check
        diff = np.linalg.norm(x_new - x_k)
        norm_new = np.linalg.norm(x_new)
        rel_change = diff / (norm_new + 1e-30)

        # Objective value (for monitoring)
        residual = y - A @ x_new
        obj = 0.5 * np.sum(np.abs(residual) ** 2) + lam * np.sum(np.abs(x_new))
        objectives.append(float(obj))

        if verbose and (k + 1) % 50 == 0:
            nnz = np.sum(np.abs(x_new) > 1e-10 * np.max(np.abs(x_new)))
            print(f"    FISTA iter {k+1:4d}: obj={obj:.6e}, "
                  f"rel_change={rel_change:.2e}, nnz={nnz}")

        x_k = x_new
        t_k = t_new

        if rel_change < tol and k > 10:
            if verbose:
                print(f"    FISTA converged at iter {k+1}")
            break

    nnz = int(np.sum(np.abs(x_k) > 1e-10 * np.max(np.abs(x_k) + 1e-30)))
    info = {
        'n_iter': k + 1,
        'converged': rel_change < tol,
        'objective': float(objectives[-1]) if objectives else float('inf'),
        'objectives': objectives,
        'n_nonzero': nnz,
        'lipschitz': L,
        'lambda': lam,
    }
    return x_k, info


# ════════════════════════════════════════════════════════════════════
#  Weighted FISTA, Debiasing, and Iteratively Reweighted L1
# ════════════════════════════════════════════════════════════════════

def fista_weighted_bpdn(
    A: np.ndarray,
    y: np.ndarray,
    lam: float,
    weights: np.ndarray,
    L: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    FISTA for the weighted L1 problem:

        min_x  (1/2)||y - Ax||² + λ Σ_i w_i |x_i|

    Identical to fista_bpdn except the soft-threshold is per-element
    (threshold_i = λ · w_i / L). Used as the inner solve for IRW-L1.

    Parameters
    ----------
    weights : real array, shape (M,)
        Non-negative per-element weights. weights = 1 recovers plain BPDN.
    """
    N, M = A.shape
    if L is None:
        L = estimate_lipschitz(A)
    step = 1.0 / L

    AHy = A.conj().T @ y
    x_k = np.zeros(M, dtype=np.complex128)
    z_k = x_k.copy()
    t_k = 1.0
    threshold = lam * step * weights   # per-element

    objectives = []
    for k in range(max_iter):
        grad = A.conj().T @ (A @ z_k) - AHy
        x_new = _prox_complex_l1(z_k - step * grad, threshold)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2))
        z_k = x_new + ((t_k - 1.0) / t_new) * (x_new - x_k)

        diff = np.linalg.norm(x_new - x_k)
        rel_change = diff / (np.linalg.norm(x_new) + 1e-30)

        residual = y - A @ x_new
        obj = 0.5 * np.sum(np.abs(residual) ** 2) + lam * np.sum(weights * np.abs(x_new))
        objectives.append(float(obj))

        if verbose and (k + 1) % 50 == 0:
            print(f"    wFISTA iter {k+1:4d}: obj={obj:.6e}, rel={rel_change:.2e}")

        x_k = x_new
        t_k = t_new
        if rel_change < tol and k > 10:
            break

    info = {
        'n_iter': k + 1,
        'converged': rel_change < tol,
        'objective': float(objectives[-1]) if objectives else float('inf'),
        'objectives': objectives,
        'lipschitz': L,
        'lambda': lam,
    }
    return x_k, info


def debias_lasso(
    A: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    threshold_ratio: float = 1e-3,
) -> np.ndarray:
    """
    Debiased LASSO: refit complex amplitudes by ordinary least-squares
    on the support recovered by FISTA.

    LASSO/BPDN shrinks amplitudes towards zero (the price of L1 sparsity).
    Once the support S = {i : |x_i| > τ·max|x|} is identified, the OLS
    solution on that support gives unbiased amplitudes:

        x_debiased[S] = pinv(A[:, S]) @ y
        x_debiased[~S] = 0

    Parameters
    ----------
    A : (N, M) complex
    y : (N,) complex
    x : (M,) complex — FISTA solution
    threshold_ratio : float
        Relative threshold for support detection (default 0.1% of peak).

    Returns
    -------
    x_debiased : (M,) complex
    """
    peak = float(np.max(np.abs(x)))
    if peak < 1e-30:
        return x.copy()
    S = np.abs(x) > threshold_ratio * peak
    if not np.any(S):
        return x.copy()

    x_db = np.zeros_like(x)
    A_S = A[:, S]
    # lstsq is more stable than pinv for tall/skinny columns
    sol, *_ = np.linalg.lstsq(A_S, y, rcond=None)
    x_db[S] = sol
    return x_db


def fista_irw_l1(
    A: np.ndarray,
    y: np.ndarray,
    lam: float,
    n_reweight: int = 3,
    eps: Optional[float] = None,
    L: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Iteratively Reweighted L1 (Candès–Wakin–Boyd 2008).

    Approaches the L0 (true-sparsity) optimum by solving a sequence of
    weighted L1 problems, where each iteration's weights penalise
    already-small entries more heavily:

        Iter 0:           min ||y - Ax||² + λ ||x||_1
        Iter k+1: w_i ← 1 / (|x_i^{(k)}| + ε)
                  min ||y - Ax||² + λ Σ w_i |x_i|

    Weights are mean-normalised each iteration so λ keeps its meaning.

    Parameters
    ----------
    n_reweight : int
        Number of outer reweighting passes (k=0 is plain FISTA;
        k=3 is the standard recommendation).
    eps : float or None
        Stability parameter. Default: 0.1 × first-pass median(|x_nz|),
        which adapts to the signal scale.

    Returns
    -------
    x : (M,) complex
    info : dict with keys 'n_iter' (sum across passes), 'reweight_iters'
           (list of inner iter counts), 'eps', 'lambda'.
    """
    if L is None:
        L = estimate_lipschitz(A)

    M = A.shape[1]
    weights = np.ones(M, dtype=np.float64)
    reweight_iters: List[int] = []
    total_iters = 0
    last_info: dict = {}
    x = np.zeros(M, dtype=np.complex128)

    for outer in range(n_reweight + 1):
        x, info = fista_weighted_bpdn(
            A, y, lam, weights, L=L, max_iter=max_iter, tol=tol, verbose=False
        )
        reweight_iters.append(info['n_iter'])
        total_iters += info['n_iter']
        last_info = info

        # Skip the final reweight (we wouldn't use it)
        if outer == n_reweight:
            break

        mag = np.abs(x)
        if eps is None:
            nz = mag[mag > 1e-12 * mag.max()] if mag.max() > 0 else mag
            eps = 0.1 * float(np.median(nz)) if nz.size > 0 else 1e-4
            if eps < 1e-12:
                eps = 1e-4

        weights = 1.0 / (mag + eps)
        # Mean-normalise so λ stays interpretable across iterations
        weights *= M / weights.sum()

        if verbose:
            nnz = int(np.sum(mag > 1e-6 * mag.max()))
            print(f"    IRW pass {outer+1}/{n_reweight}: "
                  f"inner_iters={info['n_iter']}, nnz={nnz}, eps={eps:.3e}")

    info_out = {
        'n_iter': total_iters,
        'reweight_iters': reweight_iters,
        'eps': eps if eps is not None else float('nan'),
        'lipschitz': L,
        'lambda': lam,
        'converged': last_info.get('converged', False),
    }
    return x, info_out


def fista_irw_l1_debiased(
    A: np.ndarray,
    y: np.ndarray,
    lam: float,
    n_reweight: int = 3,
    eps: Optional[float] = None,
    L: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    threshold_ratio: float = 1e-3,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Convenience wrapper: IRW-L1 followed by OLS debiasing on the recovered
    support. Returns (x_debiased, info).
    """
    x, info = fista_irw_l1(
        A, y, lam, n_reweight=n_reweight, eps=eps, L=L,
        max_iter=max_iter, tol=tol, verbose=verbose,
    )
    x_db = debias_lasso(A, y, x, threshold_ratio=threshold_ratio)
    info['debiased'] = True
    info['support_size'] = int(np.sum(np.abs(x_db) > 0))
    return x_db, info


# ════════════════════════════════════════════════════════════════════
#  GPU-Accelerated FISTA (PyTorch MPS)
# ════════════════════════════════════════════════════════════════════

import threading
_mps_lock = threading.Lock()

# Check for PyTorch + MPS once at import time
try:
    import torch as _torch
    _HAS_MPS = _torch.backends.mps.is_available()
except ImportError:
    _torch = None
    _HAS_MPS = False


def fista_bpdn_gpu(
    A: np.ndarray,
    y: np.ndarray,
    lam: Union[float, np.ndarray],
    L: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    GPU-accelerated FISTA via PyTorch MPS (Apple Metal).

    Supports batched processing: if y is shape (N, B), returns x of shape (M, B).
    """
    if not _HAS_MPS:
        # CPU fallback doesn't support batching natively, but we ensure it fails gracefully 
        # or we just let it fail if user passes 2D to CPU fallback.
        return fista_bpdn(A, y, lam, L=L, max_iter=max_iter, tol=tol,
                          verbose=verbose)

    device = _torch.device('mps')
    N, M = A.shape

    batched = y.ndim == 2
    if not batched:
        y = y.reshape(-1, 1)
    
    B = y.shape[1]

    if L is None:
        L = estimate_lipschitz(A)
    step = 1.0 / L

    # Serialize MPS execution to prevent Metal driver crashes
    with _mps_lock:
        try:
            A_t = _torch.tensor(A, dtype=_torch.complex64, device=device)
            y_t = _torch.tensor(y, dtype=_torch.complex64, device=device)
            AH_t = A_t.conj().T
            AHy_t = AH_t @ y_t   # (M, B)

            lam_arr = np.atleast_1d(lam)
            if lam_arr.size == 1:
                lam_arr = np.repeat(lam_arr, B)
            lam_t = _torch.tensor(lam_arr, dtype=_torch.float32, device=device).unsqueeze(0) # (1, B)

            x_k = _torch.zeros((M, B), dtype=_torch.complex64, device=device)
            z_k = x_k.clone()
            t_k = 1.0
            threshold_t = lam_t * step

            objectives = []
            rel_change = float('inf')

            for k in range(max_iter):
                # Gradient: A^H(A z_k) - A^H y
                grad = AH_t @ (A_t @ z_k) - AHy_t

                # Proximal gradient step
                u = z_k - step * grad
                mag = u.abs()
                scale = _torch.clamp(1.0 - threshold_t / _torch.clamp(mag, min=1e-30), min=0.0)
                x_new = u * scale

                # Nesterov momentum
                t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_k ** 2) ** 0.5)
                momentum = (t_k - 1.0) / t_new
                z_k = x_new + momentum * (x_new - x_k)

                # Convergence check (every 10 iterations to save sync)
                if (k + 1) % 10 == 0 or k == max_iter - 1:
                    diffs = (x_new - x_k).abs().sum(dim=0)
                    norms = x_new.abs().sum(dim=0)
                    rel_changes = diffs / (norms + 1e-30)
                    rel_change = rel_changes.max().item()

                    if verbose and (k + 1) % 50 == 0:
                        nnz = int((x_new.abs() > 1e-10 * x_new.abs().max(dim=0)[0].unsqueeze(0)).sum().item())
                        print(f"    FISTA-GPU iter {k+1:4d}: max rel_change={rel_change:.2e}, sum_nnz={nnz}")

                    if rel_change < tol and k > 10:
                        if verbose:
                            print(f"    FISTA-GPU converged at iter {k+1}")
                        break

                x_k = x_new
                t_k = t_new

            # Transfer back to CPU
            x_np = x_k.cpu().numpy().astype(np.complex128)
            nnz = int(np.sum(np.abs(x_np) > 1e-10 * np.max(np.abs(x_np), axis=0, keepdims=True) + 1e-30))

        finally:
            _torch.mps.empty_cache()

    info = {
        'n_iter': k + 1,
        'converged': rel_change < tol,
        'n_nonzero_total': nnz,
        'lipschitz': L,
        'device': 'mps',
        'batched': batched
    }
    
    if not batched:
        x_np = x_np[:, 0]
        
    return x_np, info


def estimate_lambda_mmv(
    A: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    λ_max for MMV-BPDN with L2,1 penalty:

        λ_max = max_m ||A[:,m]^H Y||_2

    For Y of shape (N, B), this is the L2 norm across time of each
    atom's correlation with the data. Below λ_max the all-zeros
    solution is no longer optimal.
    """
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    corr = A.conj().T @ Y         # (M, B)
    row_norms = np.linalg.norm(corr, axis=1)
    lam_max = float(np.max(row_norms))
    return alpha * lam_max


def fista_mmv_bpdn(
    A: np.ndarray,
    Y: np.ndarray,
    lam: float,
    L: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    MMV-BPDN via FISTA — joint sparse recovery across time bursts.

    Solves:

        min_X  (1/2) ||Y - A X||²_F  +  λ Σ_m ||X[m,:]||_2

    The L2,1 row-norm penalty couples all time bursts: each atom m is
    either active in every burst or in none of them. Complex amplitudes
    within an active row are unconstrained (so phase can evolve = layer
    motion is preserved).

    Parameters
    ----------
    A : (N, M) complex
    Y : (N, B) complex — B time bursts of length N
        (a 1-D array is auto-promoted to (N, 1) but you should use
        fista_bpdn for single-burst problems.)
    lam : float
        L2,1 regularisation. Use estimate_lambda_mmv to set it.
    L : float or None
        Lipschitz constant of A^H A (independent of B).

    Returns
    -------
    X : (M, B) complex
    info : dict
    """
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    N, B = Y.shape
    M = A.shape[1]

    if L is None:
        L = estimate_lipschitz(A)
    step = 1.0 / L

    AHY = A.conj().T @ Y              # (M, B)

    X_k = np.zeros((M, B), dtype=np.complex128)
    Z_k = X_k.copy()
    t_k = 1.0
    threshold = lam * step

    objectives: List[float] = []
    rel_change = float('inf')

    for k in range(max_iter):
        grad = A.conj().T @ (A @ Z_k) - AHY      # (M, B)
        U    = Z_k - step * grad

        # Row-wise L2 soft-threshold (block prox of L2,1)
        row_norm = np.linalg.norm(U, axis=1)      # (M,)
        scale    = np.maximum(0.0, 1.0 - threshold / np.maximum(row_norm, 1e-30))
        X_new    = U * scale[:, np.newaxis]

        t_new   = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2))
        Z_k     = X_new + ((t_k - 1.0) / t_new) * (X_new - X_k)

        diff_norm = np.linalg.norm(X_new - X_k)
        rel_change = diff_norm / (np.linalg.norm(X_new) + 1e-30)

        residual = Y - A @ X_new
        obj = 0.5 * float(np.sum(np.abs(residual) ** 2)) + \
              lam * float(np.sum(np.linalg.norm(X_new, axis=1)))
        objectives.append(obj)

        if verbose and (k + 1) % 50 == 0:
            active = int(np.sum(np.linalg.norm(X_new, axis=1) > 1e-10 * np.linalg.norm(X_new)))
            print(f"    MMV-FISTA iter {k+1:4d}: obj={obj:.6e}, "
                  f"rel={rel_change:.2e}, n_atoms_active={active}")

        X_k = X_new
        t_k = t_new
        if rel_change < tol and k > 10:
            break

    info = {
        'n_iter': k + 1,
        'converged': rel_change < tol,
        'objective': objectives[-1] if objectives else float('inf'),
        'objectives': objectives,
        'lipschitz': L,
        'lambda': lam,
    }
    return X_k, info


def fista_mmv_bpdn_gpu(
    A: np.ndarray,
    Y: np.ndarray,
    lam: float,
    L: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """GPU (PyTorch MPS) MMV-BPDN. See fista_mmv_bpdn for the math."""
    if not _HAS_MPS:
        return fista_mmv_bpdn(A, Y, lam, L=L, max_iter=max_iter, tol=tol,
                               verbose=verbose)

    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    device = _torch.device('mps')
    N, B = Y.shape
    M    = A.shape[1]

    if L is None:
        L = estimate_lipschitz(A)
    step = 1.0 / L

    with _mps_lock:
        try:
            A_t   = _torch.tensor(A, dtype=_torch.complex64, device=device)
            Y_t   = _torch.tensor(Y, dtype=_torch.complex64, device=device)
            AH_t  = A_t.conj().T
            AHY_t = AH_t @ Y_t                        # (M, B)

            X_k = _torch.zeros((M, B), dtype=_torch.complex64, device=device)
            Z_k = X_k.clone()
            t_k = 1.0
            threshold = lam * step

            rel_change = float('inf')
            for k in range(max_iter):
                grad = AH_t @ (A_t @ Z_k) - AHY_t
                U    = Z_k - step * grad

                # Row-wise L2 norm over complex magnitudes, then soft-threshold
                row_norm = U.abs().pow(2).sum(dim=1).sqrt()        # (M,)
                scale    = _torch.clamp(
                    1.0 - threshold / _torch.clamp(row_norm, min=1e-30),
                    min=0.0,
                )                                                   # (M,)
                X_new = U * scale.unsqueeze(1)

                t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_k ** 2) ** 0.5)
                Z_k   = X_new + ((t_k - 1.0) / t_new) * (X_new - X_k)

                if (k + 1) % 10 == 0 or k == max_iter - 1:
                    diff   = (X_new - X_k).abs().pow(2).sum().sqrt()
                    nrm    = X_new.abs().pow(2).sum().sqrt()
                    rel_change = (diff / (nrm + 1e-30)).item()
                    if verbose and (k + 1) % 50 == 0:
                        active = int((row_norm > 1e-10 * row_norm.max()).sum().item())
                        print(f"    MMV-GPU iter {k+1:4d}: "
                              f"rel={rel_change:.2e}, n_active={active}")
                    if rel_change < tol and k > 10:
                        break

                X_k = X_new
                t_k = t_new

            X_np = X_k.cpu().numpy().astype(np.complex128)

        finally:
            _torch.mps.empty_cache()

    info = {
        'n_iter': k + 1,
        'converged': rel_change < tol,
        'lipschitz': L,
        'lambda': lam,
        'device': 'mps',
        'joint_recovery': 'mmv_l21',
    }
    return X_np, info


def fista_mmv_bpdn_warm_gpu(
    A: np.ndarray,
    Y: np.ndarray,
    lam: float,
    X_prev: Optional[np.ndarray] = None,
    mu: float = 0.0,
    L: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    MMV-BPDN with a quadratic prior toward a previous solution.

    Solves:

        min_X  (1/2)||Y - A X||²_F
              + (μ/2) ||X - X_prev||²_F          (only if X_prev given)
              + λ Σ_m ||X[m,:]||_2

    The quadratic prior couples this window to the previous one,
    encouraging temporally smooth support changes. μ=0 reduces to plain
    MMV. Large μ pins the solution to X_prev.

    Note: shapes of X_prev and Y must match in M; the number of bursts
    in Y can differ from X_prev's. If they differ, X_prev is broadcast
    across the new bursts (so the prior is "stay close to the last
    snapshot averaged across its own bursts").

    Returns (X, info).
    """
    if not _HAS_MPS:
        raise RuntimeError("fista_mmv_bpdn_warm_gpu requires PyTorch MPS")

    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    device = _torch.device('mps')
    N, B  = Y.shape
    M     = A.shape[1]

    if L is None:
        L = estimate_lipschitz(A)

    # If quadratic prior present, the gradient adds μI → Lipschitz grows by μ
    L_eff = L + mu
    step  = 1.0 / L_eff

    use_prior = (X_prev is not None) and (mu > 0)
    if use_prior:
        if X_prev.shape[0] != M:
            raise ValueError(f"X_prev shape {X_prev.shape} incompatible with A {A.shape}")
        # Broadcast across new B if necessary (column-wise mean across X_prev)
        if X_prev.shape[1] != B:
            X_prev_broadcast = np.tile(
                X_prev.mean(axis=1, keepdims=True), (1, B)
            )
        else:
            X_prev_broadcast = X_prev

    with _mps_lock:
        try:
            A_t   = _torch.tensor(A, dtype=_torch.complex64, device=device)
            Y_t   = _torch.tensor(Y, dtype=_torch.complex64, device=device)
            AH_t  = A_t.conj().T
            AHY_t = AH_t @ Y_t

            if use_prior:
                Xp_t = _torch.tensor(X_prev_broadcast, dtype=_torch.complex64,
                                       device=device)
                # Warm start from prior
                X_k = Xp_t.clone()
            else:
                Xp_t = None
                X_k = _torch.zeros((M, B), dtype=_torch.complex64, device=device)

            Z_k = X_k.clone()
            t_k = 1.0
            threshold = lam * step

            rel_change = float('inf')
            for k in range(max_iter):
                grad_data = AH_t @ (A_t @ Z_k) - AHY_t        # (M, B)
                if use_prior:
                    grad = grad_data + mu * (Z_k - Xp_t)
                else:
                    grad = grad_data
                U = Z_k - step * grad

                row_norm = U.abs().pow(2).sum(dim=1).sqrt()
                scale    = _torch.clamp(
                    1.0 - threshold / _torch.clamp(row_norm, min=1e-30),
                    min=0.0,
                )
                X_new = U * scale.unsqueeze(1)

                t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_k ** 2) ** 0.5)
                Z_k   = X_new + ((t_k - 1.0) / t_new) * (X_new - X_k)

                if (k + 1) % 10 == 0 or k == max_iter - 1:
                    diff   = (X_new - X_k).abs().pow(2).sum().sqrt()
                    nrm    = X_new.abs().pow(2).sum().sqrt()
                    rel_change = (diff / (nrm + 1e-30)).item()
                    if verbose and (k + 1) % 50 == 0:
                        active = int((row_norm > 1e-10 * row_norm.max()).sum().item())
                        print(f"    warm-MMV iter {k+1:4d}: "
                              f"rel={rel_change:.2e}, active={active}")
                    if rel_change < tol and k > 10:
                        break

                X_k = X_new
                t_k = t_new

            X_np = X_k.cpu().numpy().astype(np.complex128)
        finally:
            _torch.mps.empty_cache()

    info = {
        'n_iter': k + 1,
        'converged': rel_change < tol,
        'lipschitz_eff': L_eff,
        'mu': mu,
        'device': 'mps',
    }
    return X_np, info


def fista_group_bpdn_gpu(
    A_list: List[np.ndarray],
    y_list: List[np.ndarray],
    lam: float,
    L_list: Optional[List[float]] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], dict]:
    """
    Joint Sparse Recovery (Multi-Band Group LASSO) via PyTorch MPS.
    
    Solves for K subbands simultaneously, forcing them to share the same
    non-zero depth locations (support) via a mixed L2,1 norm penalty,
    but allowing independent complex amplitudes (phases).
    
    A_list: List of K dictionaries (N_k x M)
    y_list: List of K data tensors (N_k x B)
    Returns list of K recovered sparse tensors (M x B).
    """
    if not _HAS_MPS:
        raise RuntimeError("fista_group_bpdn_gpu requires PyTorch MPS")
        
    device = _torch.device('mps')
    K = len(A_list)
    
    # Ensure batched
    for k in range(K):
        if y_list[k].ndim == 1:
            y_list[k] = y_list[k].reshape(-1, 1)
            
    M = A_list[0].shape[1]
    B = y_list[0].shape[1]
    
    if L_list is None:
        L_list = [estimate_lipschitz(A) for A in A_list]
        
    # In FISTA for Group LASSO, the step size must be bounded by the maximum Lipschitz 
    # constant across all operators for a uniform gradient step.
    step = 1.0 / max(L_list)
    
    with _mps_lock:
        try:
            A_t = []
            AH_t = []
            AHy_t = []
            
            for k in range(K):
                At = _torch.tensor(A_list[k], dtype=_torch.complex64, device=device)
                yt = _torch.tensor(y_list[k], dtype=_torch.complex64, device=device)
                AHt = At.conj().T
                
                A_t.append(At)
                AH_t.append(AHt)
                AHy_t.append(AHt @ yt)
                
            x_k = _torch.zeros((K, M, B), dtype=_torch.complex64, device=device)
            z_k = x_k.clone()
            t_k = 1.0
            
            lam_arr = np.atleast_1d(lam)
            if lam_arr.size == 1:
                lam_arr = np.repeat(lam_arr, B)
            lam_t = _torch.tensor(lam_arr, dtype=_torch.float32, device=device).unsqueeze(0) # (1, B)
            threshold_t = lam_t * step
            
            rel_change = float('inf')
            
            for iter_idx in range(max_iter):
                # 1. Gradient step per subband
                u = _torch.zeros_like(z_k)
                for k in range(K):
                    grad_k = AH_t[k] @ (A_t[k] @ z_k[k]) - AHy_t[k]
                    u[k] = z_k[k] - step * grad_k
                    
                # 2. Group LASSO block soft-thresholding across K
                # u shape is (K, M, B). We compute L2 norm across K (dim=0).
                norm_u = u.abs().norm(p=2, dim=0)  # Shape: (M, B)
                
                scale = _torch.clamp(1.0 - threshold_t / _torch.clamp(norm_u, min=1e-30), min=0.0)
                # target scale shape (1, M, B) to broadcast over K
                x_new = u * scale.unsqueeze(0)
                
                # 3. Nesterov momentum
                t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_k ** 2) ** 0.5)
                momentum = (t_k - 1.0) / t_new
                z_k = x_new + momentum * (x_new - x_k)
                
                # 4. Convergence check (every 10 iterations)
                if (iter_idx + 1) % 10 == 0 or iter_idx == max_iter - 1:
                    diffs = (x_new - x_k).abs().sum(dim=(0, 1))
                    norms = x_new.abs().sum(dim=(0, 1))
                    rel_changes = diffs / (norms + 1e-30)
                    rel_change = rel_changes.max().item()
                    
                    if verbose and (iter_idx + 1) % 50 == 0:
                        # Nonzero if any subband is active (they are structurally locked together)
                        nnz = int((norm_u > 1e-10 * norm_u.max(dim=0)[0].unsqueeze(0)).sum().item())
                        print(f"    Joint FISTA iter {iter_idx+1:4d}: max rel_change={rel_change:.2e}, sum_nnz={nnz}")
                        
                    if rel_change < tol and iter_idx > 10:
                        if verbose:
                            print(f"    Joint FISTA converged at iter {iter_idx+1}")
                        break
                        
                x_k = x_new
                t_k = t_new
                
            x_np_list = [x_k[k].cpu().numpy().astype(np.complex128) for k in range(K)]
            
        finally:
            _torch.mps.empty_cache()
            
    info = {
        'n_iter': iter_idx + 1,
        'converged': rel_change < tol,
        'lipschitz_max': max(L_list),
        'device': 'mps',
        'joint_recovery': True
    }
    
    return x_np_list, info


# ════════════════════════════════════════════════════════════════════
#  Peak Extraction
# ════════════════════════════════════════════════════════════════════

def extract_components(
    x: np.ndarray,
    fine_depths: np.ndarray,
    min_amplitude_db: float = -25.0,
    merge_distance: Optional[float] = None,
) -> List[Tuple[float, complex]]:
    """
    Extract reflector positions and amplitudes from the sparse vector x.

    Because the oversampled dictionary has highly correlated adjacent columns,
    FISTA often spreads energy across several neighbouring bins for a single
    physical reflector. This function merges nearby activated entries into
    single reflectors using amplitude-weighted centroid positions.

    Parameters
    ----------
    x : complex array, shape (M,)
        Recovered sparse vector from FISTA.
    fine_depths : array, shape (M,)
        Fine depth grid corresponding to x.
    min_amplitude_db : float
        Minimum amplitude (dB relative to peak) to keep a component.
    merge_distance : float or None
        Maximum depth separation (m) to merge adjacent entries.
        Default: 0.5 × Rayleigh limit.

    Returns
    -------
    components : list of (depth_m, complex_amplitude)
        Extracted reflectors, sorted by depth. Each component's depth is
        the amplitude-weighted centroid of the merged cluster, and its
        amplitude is the coherent sum of the cluster members.
    """
    if merge_distance is None:
        merge_distance = RAYLEIGH_LIMIT * 0.5

    amplitudes = np.abs(x)
    peak = np.max(amplitudes)
    if peak < 1e-30:
        return []

    threshold = peak * 10 ** (min_amplitude_db / 20.0)
    active_idx = np.where(amplitudes > threshold)[0]

    if len(active_idx) == 0:
        return []

    # Sort by depth
    active_idx = active_idx[np.argsort(fine_depths[active_idx])]

    # Cluster adjacent entries that are within merge_distance
    clusters = []
    current_cluster = [active_idx[0]]
    for i in range(1, len(active_idx)):
        if fine_depths[active_idx[i]] - fine_depths[active_idx[i - 1]] <= merge_distance:
            current_cluster.append(active_idx[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [active_idx[i]]
    clusters.append(current_cluster)

    # For each cluster, compute amplitude-weighted centroid depth
    # and coherent sum of complex amplitudes
    components = []
    for cluster in clusters:
        c_amps = amplitudes[cluster]
        c_depths = fine_depths[cluster]
        c_complex = x[cluster]

        # Amplitude-weighted centroid
        total_amp = np.sum(c_amps)
        centroid = np.sum(c_amps * c_depths) / total_amp

        # Coherent sum of complex amplitudes
        total_complex = np.sum(c_complex)

        components.append((float(centroid), complex(total_complex)))

    # Sort by depth
    components.sort(key=lambda c: c[0])
    return components


# ════════════════════════════════════════════════════════════════════
#  1D and 2D Interfaces
# ════════════════════════════════════════════════════════════════════

def cs_1d(
    spectrum: np.ndarray,
    depth_axis: np.ndarray,
    f_lo: float = f_start,
    f_hi: float = f_end,
    oversample: int = 4,
    lambda_alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_amplitude_db: float = -25.0,
    A: Optional[np.ndarray] = None,
    fine_depths: Optional[np.ndarray] = None,
    L: Optional[float] = None,
    verbose: bool = False,
    use_gpu: bool = True,
) -> Tuple[List[Tuple[float, complex]], np.ndarray, dict]:
    """
    Compressed sensing recovery for a single range profile.

    Parameters
    ----------
    spectrum : complex array, shape (N,)
        Observed range-compressed echo (one time step).
    depth_axis : array, shape (N,)
        Depth values (m).
    f_lo, f_hi : float
        Frequency band edges (Hz).
    oversample : int
        Dictionary oversampling factor.
    lambda_alpha : float
        Sparsity parameter (fraction of λ_max).
    max_iter, tol : FISTA parameters.
    min_amplitude_db : float
        Minimum component amplitude (dB below peak).
    A, fine_depths, L : optional precomputed values
        Pass these to avoid recomputing the dictionary for each time step.
    verbose : bool
    use_gpu : bool
        Use GPU-accelerated FISTA if PyTorch MPS is available (default True).

    Returns
    -------
    components : list of (depth_m, complex_amplitude)
    residual : complex array, shape (N,)
        Reconstruction residual  y - A @ x_recovered.
    info : dict
        FISTA convergence information.
    """
    y = spectrum.astype(np.complex128)

    # Build dictionary if not provided
    if A is None or fine_depths is None:
        A, fine_depths = build_dictionary(depth_axis, f_lo, f_hi, oversample)

    # Estimate Lipschitz constant if not provided
    if L is None:
        L = estimate_lipschitz(A)

    # Estimate λ
    lam = estimate_lambda(A, y, alpha=lambda_alpha)

    # Solve (GPU if available and requested)
    solver = fista_bpdn_gpu if (use_gpu and _HAS_MPS) else fista_bpdn
    x, info = solver(A, y, lam, L=L, max_iter=max_iter, tol=tol,
                     verbose=verbose)

    # Extract components
    components = extract_components(x, fine_depths, min_amplitude_db)

    # Residual
    residual = y - A @ x

    return components, residual.astype(np.complex64), info


def cs_batched(
    y_batch: np.ndarray,
    depth_axis: np.ndarray,
    f_lo: float = f_start,
    f_hi: float = f_end,
    oversample: int = 4,
    lambda_alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_amplitude_db: float = -25.0,
    A: Optional[np.ndarray] = None,
    fine_depths: Optional[np.ndarray] = None,
    L: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[List[List[Tuple[float, complex]]], dict]:
    """
    Batched compressed sensing recovery for multiple range profiles.
    Requires PyTorch MPS GPU support.

    Parameters
    ----------
    y_batch : complex array, shape (N, B)
        Observed range-compressed echoes for B time steps.
    depth_axis : array, shape (N,)
        Depth values (m).
    
    Returns
    -------
    batch_components : list of B lists of (depth_m, complex_amplitude)
    info : dict
        FISTA convergence information.
    """
    if not _HAS_MPS:
        raise RuntimeError("Batched processing requires PyTorch MPS GPU support.")
        
    y = y_batch.astype(np.complex128)

    # Build dictionary if not provided
    if A is None or fine_depths is None:
        A, fine_depths = build_dictionary(depth_axis, f_lo, f_hi, oversample)

    # Estimate Lipschitz constant if not provided
    if L is None:
        L = estimate_lipschitz(A)

    # Estimate λ for each time step in the batch
    lam = estimate_lambda(A, y, alpha=lambda_alpha)  # Shape (B,)

    # Solve all time steps in parallel on GPU
    x_batch, info = fista_bpdn_gpu(A, y, lam, L=L, max_iter=max_iter, tol=tol, verbose=verbose)

    # Extract components for each time step
    B = y.shape[1]
    batch_components = []
    for b in range(B):
        comps = extract_components(x_batch[:, b], fine_depths, min_amplitude_db)
        batch_components.append(comps)

    return batch_components, info


def cs_batched_joint(
    y_batch_dict: Dict[str, np.ndarray],
    depth_axis: np.ndarray,
    oversample: int = 4,
    lambda_alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_amplitude_db: float = -25.0,
    verbose: bool = False,
) -> Tuple[Dict[str, List[List[Tuple[float, complex]]]], dict]:
    """
    Batched joint compressed sensing recovery across multiple subbands mapping 
    to the same physical depths (Group LASSO).
    Requires PyTorch MPS GPU support.
    
    Parameters
    ----------
    y_batch_dict : Dict[str, complex array]
        Observed range-compressed echoes for B time steps.
        Keys must be subband names (e.g., 'full', 'low', 'high').
        Arrays must be shape (N_k, B).
    depth_axis : array, shape (N,)
        Depth values (m).
    """
    if not _HAS_MPS:
        raise RuntimeError("Batched joint processing requires PyTorch MPS GPU support.")
        
    subbands = list(y_batch_dict.keys())
    K = len(subbands)
    
    A_list = []
    fine_depths = None
    y_list = []
    L_list = []
    
    for sb in subbands:
        f_lo, f_hi = SUBBANDS[sb]
        A, fd = build_dictionary(depth_axis, f_lo, f_hi, oversample)
        if fine_depths is None:
            fine_depths = fd
        A_list.append(A)
        y_list.append(y_batch_dict[sb].astype(np.complex128))
        L_list.append(estimate_lipschitz(A))
        
    M = A_list[0].shape[1]
    B = y_list[0].shape[1]
    
    # Estimate joint lambda based on the maximum correlated energy block
    lam_max_batch = np.zeros(B)
    for b in range(B):
        sq_corr = np.zeros(M)
        for k in range(K):
            corr_k = np.abs(A_list[k].conj().T @ y_list[k][:, b])
            sq_corr += corr_k**2
        lam_max_batch[b] = np.sqrt(np.max(sq_corr))
        
    lam = lambda_alpha * lam_max_batch  # Shape (B,)
    
    x_batch_list, info = fista_group_bpdn_gpu(
        A_list, y_list, lam, L_list=L_list, max_iter=max_iter, tol=tol, verbose=verbose
    )
    
    # Extract components per subband per time step
    results = {}
    for k, sb in enumerate(subbands):
        batch_components = []
        x_k = x_batch_list[k]
        for b in range(B):
            comps = extract_components(x_k[:, b], fine_depths, min_amplitude_db)
            batch_components.append(comps)
        results[sb] = batch_components
        
    return results, info


def cs_2d(
    raw_complex: np.ndarray,
    depth_axis: np.ndarray,
    time_axis: np.ndarray,
    f_lo: float = f_start,
    f_hi: float = f_end,
    oversample: int = 4,
    lambda_alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_amplitude_db: float = -25.0,
    progress_interval: int = 50,
) -> Tuple[List[List[Tuple[float, complex]]], np.ndarray, dict]:
    """
    Apply compressed sensing to every time step of a 2D echogram.

    Parameters
    ----------
    raw_complex : complex array, shape (N, n_times)
    depth_axis : array, shape (N,)
    time_axis : array, shape (n_times,)
    f_lo, f_hi : float
    oversample, lambda_alpha, max_iter, tol, min_amplitude_db : CS params
    progress_interval : int

    Returns
    -------
    all_components : list of lists
        all_components[t] = list of (depth, complex_amplitude)
    residual_2d : complex array, shape (N, n_times)
    summary : dict
        Processing statistics.
    """
    n_depth, n_times = raw_complex.shape
    residual_2d = np.zeros_like(raw_complex)

    # Build dictionary ONCE (shared across all time steps)
    print(f"  Building {oversample}× oversampled dictionary "
          f"({n_depth} × {n_depth * oversample})...")
    t0_dict = timer.time()
    A, fine_depths = build_dictionary(depth_axis, f_lo, f_hi, oversample)
    print(f"  Dictionary built in {timer.time() - t0_dict:.1f}s")

    # Estimate Lipschitz constant ONCE
    print("  Estimating Lipschitz constant...")
    L = estimate_lipschitz(A)
    print(f"  L = {L:.2e}")

    all_components = []
    total_comps = 0
    total_iters = 0
    t0 = timer.time()

    for ti in range(n_times):
        spectrum = raw_complex[:, ti]
        comps, resid, info = cs_1d(
            spectrum, depth_axis, f_lo, f_hi,
            oversample=oversample,
            lambda_alpha=lambda_alpha,
            max_iter=max_iter,
            tol=tol,
            min_amplitude_db=min_amplitude_db,
            A=A, fine_depths=fine_depths, L=L,
        )
        all_components.append(comps)
        residual_2d[:, ti] = resid
        total_comps += len(comps)
        total_iters += info['n_iter']

        if (ti + 1) % progress_interval == 0 or ti + 1 == n_times:
            elapsed = timer.time() - t0
            rate = (ti + 1) / elapsed
            eta = (n_times - ti - 1) / rate if rate > 0 else 0
            avg_comps = total_comps / (ti + 1)
            print(f"    [{ti+1:4d}/{n_times}] "
                  f"avg {avg_comps:.1f} comps/step, "
                  f"{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = timer.time() - t0
    summary = {
        'total_time_s': elapsed,
        'avg_components_per_step': total_comps / n_times,
        'avg_iterations_per_step': total_iters / n_times,
        'oversample': oversample,
        'lambda_alpha': lambda_alpha,
        'n_fine_depths': len(fine_depths),
    }
    print(f"  CS complete: {n_times} steps in {elapsed:.1f}s, "
          f"avg {summary['avg_components_per_step']:.1f} comps/step")

    return all_components, residual_2d, summary


# ════════════════════════════════════════════════════════════════════
#  Raw Beat Signal CS (Pre-FFT Super-Resolution)
# ════════════════════════════════════════════════════════════════════

def build_beat_dictionary(
    t: np.ndarray,
    fine_depths: np.ndarray,
    K_hz: float,
    ci: float,
    win: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build FMCW beat-signal dictionary for pre-FFT compressed sensing.

    Each column models the windowed analytic beat signal for a unit-amplitude
    reflector at a candidate depth:

        A[n, j] = win[n] · exp(j · 2π · f_beat_j · t[n])
        f_beat_j = K_hz · 2·d_j / ci    (Hz)

    Unlike build_dictionary (which models the PSF in the already-FFT'd range
    profile), this dictionary acts directly on the N_ADC_SAMPLES-length beat
    signal.  With N ≈ 40000 observations (vs ~1087 range bins post-FFT), M
    can be much larger relative to N while still satisfying the RIP, enabling
    true super-resolution recovery down to the noise floor.

    Parameters
    ----------
    t : array (N,)
        Sample times (s) — t[n] = n / fs.
    fine_depths : array (M,)
        Candidate reflector depths (m).
    K_hz : float
        Chirp rate Hz/s — K_hz = BW / T.
    ci : float
        Wave speed in ice (m/s).
    win : array (N,) or None
        Window function applied to each column.  Use the same window as for
        the Hilbert-transform of the beat signal.

    Returns
    -------
    A : complex64 array (N, M), column-normalised.
    f_beats : float64 array (M,)   beat frequencies in Hz.
    """
    f_beats = K_hz * 2.0 * fine_depths / ci               # (M,)
    phase   = 2j * np.pi * t[:, np.newaxis] * f_beats[np.newaxis, :]
    A = np.exp(phase).astype(np.complex64)
    if win is not None:
        A *= win[:, np.newaxis].astype(np.float32)
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1.0
    A /= col_norms
    return A, f_beats


def _fmcw_phase_correction(
    fine_depths: np.ndarray,
    fc: float,
    K_rad: float,
    ci: float,
) -> np.ndarray:
    """Phase correction vector matching fmcw_range spec_cor convention.

    phiref[j] = 2π · fc · τ_j  −  K · τ_j² / 2
    where τ_j = 2·d_j / ci,  K in rad/s/s.

    Multiply raw CS result x by exp(−j · phiref) to get the same phase
    as the spec_cor array stored in the zarr file.
    """
    tau    = 2.0 * fine_depths / ci
    phiref = 2.0 * np.pi * fc * tau - 0.5 * K_rad * tau ** 2
    return phiref


def cs_raw_burst(
    vdat: np.ndarray,
    data: 'ApRESData',
    depth_start: float,
    depth_end: float,
    oversample: int = 4,
    lambda_alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_amplitude_db: float = -25.0,
    A: Optional[np.ndarray] = None,
    fine_depths: Optional[np.ndarray] = None,
    L: Optional[float] = None,
    verbose: bool = False,
    use_gpu: bool = True,
    apply_phase_correction: bool = True,
) -> Tuple[List[Tuple[float, complex]], np.ndarray, np.ndarray, dict]:
    """Run compressed sensing directly on a raw FMCW beat signal.

    Unlike cs_1d (which operates on an already range-compressed echo),
    this function operates on the raw ADC voltage samples, giving true
    super-resolution by fitting complex exponentials directly to the
    beat signal.

    Processing steps
    ----------------
    1. Average and DC-remove all subbursts: v̄[n] = mean(vdat, axis=0) − DC
    2. Hilbert transform → analytic signal + Blackman window: y[n]
    3. Build fine depth grid at oversample × native Rayleigh resolution
    4. FISTA solve: min ||y − A·x||² + λ||x||₁
    5. Apply fmcw_range-equivalent phase correction to match zarr convention

    Memory note: A has shape (N_ADC_SAMPLES, M).  For narrow depth windows
    (< 50 m) M is small and this is fast.  For wide windows (> 200 m) the
    matrix can exceed 1 GB — consider splitting the depth range.

    Parameters
    ----------
    vdat : real array (NSubBursts, N_ADC_SAMPLES)
        Raw ADC voltages from fmcw_load.
    data : ApRESData
        Metadata object from fmcw_load (fs, f0, f1, bandwidth, chirp_grad, ci).
    depth_start, depth_end : float
        Depth window to process (m).
    oversample : int
        Grid oversampling factor (relative to native Rayleigh bin ~0.42 m).
    apply_phase_correction : bool
        If True, apply fmcw_range-equivalent phase correction so component
        phases are directly comparable to the zarr spec_cor convention.

    Returns
    -------
    components : list of (depth_m, complex_amplitude)
    fine_depths : array (M,)
    x : complex array (M,) — full sparse solution
    info : dict — FISTA convergence info
    """
    from scipy.signal import hilbert as _hilbert

    # 1. Average subbursts, remove DC
    v = np.mean(vdat - np.mean(vdat, axis=1, keepdims=True), axis=0).astype(np.float64)
    N = len(v)

    # 2. Window + analytic signal
    win = blackman(N)
    y   = (_hilbert(v) * win).astype(np.complex128)

    # 3. Fine depth grid
    K_hz      = data.bandwidth / data.chirp_time
    ci        = data.ci
    native_dz = ci / (2.0 * data.bandwidth)
    fine_dz   = native_dz / oversample
    t         = data.t[:N]

    if fine_depths is None:
        fine_depths = np.arange(depth_start, depth_end + fine_dz * 0.5, fine_dz)

    # 4. Dictionary (reuse pre-built if provided)
    if A is None:
        A, _ = build_beat_dictionary(t, fine_depths, K_hz, ci, win=win)
    if L is None:
        L = estimate_lipschitz(A)

    lam    = estimate_lambda(A, y, alpha=lambda_alpha)
    solver = fista_bpdn_gpu if (use_gpu and _HAS_MPS) else fista_bpdn
    x, info = solver(A, y, lam, L=L, max_iter=max_iter, tol=tol, verbose=verbose)

    # 5. Phase correction
    if apply_phase_correction:
        fc     = (data.f0 + data.f1) / 2.0
        phiref = _fmcw_phase_correction(fine_depths, fc, data.chirp_grad, ci)
        x      = x * np.exp(-1j * phiref)

    components = extract_components(x, fine_depths, min_amplitude_db)
    return components, fine_depths, x, info


def cs_raw_timeseries(
    data_dir: str,
    depth_start: float = 600.0,
    depth_end: float = 1100.0,
    oversample: int = 4,
    lambda_alpha: float = 0.1,
    max_iter: int = 300,
    tol: float = 1e-5,
    min_amplitude_db: float = -30.0,
    step: int = 1,
    verbose: bool = True,
    use_gpu: bool = True,
) -> dict:
    """Process all .DAT files in a directory using raw beat signal CS.

    Pre-FFT alternative to cs_2d: reads raw .DAT files and runs CS
    directly on the ADC beat signal instead of the zarr range profiles.
    The dictionary is built once from the first file and reused for all
    subsequent bursts (valid as long as chirp parameters are unchanged).

    Parameters
    ----------
    data_dir : str
        Path to directory containing .DAT files (e.g. data/apres/raw).
    depth_start, depth_end : float
        Depth window of interest (m).
    oversample : int
        Oversampling factor relative to native Rayleigh bin (~0.42 m).
    step : int
        Process every Nth file (1 = all files).

    Returns
    -------
    dict with keys:
        'components'  — list (N_files) of list of (depth_m, complex_amp)
        'time_days'   — float array (N_files,)
        'timestamps'  — list of datetime
        'fine_depths' — depth grid (M,)
        'depth_start', 'depth_end', 'oversample', 'lambda_alpha'
    """
    if fmcw_load is None:
        raise ImportError("apres.io not found — ensure the src/ directory is on sys.path")

    data_path = Path(data_dir)
    all_files = sorted(data_path.glob('*.DAT'))
    files = all_files[::step]
    if not files:
        raise ValueError(f"No .DAT files found in {data_dir}")

    if verbose:
        print(f"Found {len(all_files)} .DAT files, processing {len(files)}")
        print(f"Depth window: {depth_start}–{depth_end} m, oversample={oversample}")

    # Load first file to build shared dictionary + Lipschitz estimate
    data0     = fmcw_load(str(files[0]))
    K_hz      = data0.bandwidth / data0.chirp_time
    ci        = data0.ci
    native_dz = ci / (2.0 * data0.bandwidth)
    fine_dz   = native_dz / oversample
    fine_depths = np.arange(depth_start, depth_end + fine_dz * 0.5, fine_dz)
    N   = data0.n_adc_samples
    t   = data0.t[:N]
    win = blackman(N)
    M   = len(fine_depths)
    dict_mem_mb = N * M * 8 / 1e6

    if verbose:
        print(f"  Native bin: {native_dz:.3f} m → fine grid: {fine_dz:.4f} m, M={M}")
        print(f"  Building dictionary A ({N}×{M}, ~{dict_mem_mb:.0f} MB)...")
    t0_d = timer.time()
    A, _ = build_beat_dictionary(t, fine_depths, K_hz, ci, win=win)
    L    = estimate_lipschitz(A)
    if verbose:
        print(f"  Dictionary ready in {timer.time()-t0_d:.1f}s, L={L:.3e}")

    all_components: List = []
    timestamps: List     = []
    t_start = timer.time()

    for i, filepath in enumerate(files):
        try:
            data_i = fmcw_load(str(filepath))
            comps, _, _, _ = cs_raw_burst(
                data_i.vdat, data_i,
                depth_start, depth_end,
                oversample=oversample,
                lambda_alpha=lambda_alpha,
                max_iter=max_iter,
                tol=tol,
                min_amplitude_db=min_amplitude_db,
                A=A, fine_depths=fine_depths, L=L,
                verbose=False, use_gpu=use_gpu,
            )
            all_components.append(comps)
            timestamps.append(data_i.time_stamp)
        except Exception as e:
            if verbose:
                print(f"  Error {filepath.name}: {e}")
            all_components.append([])
            timestamps.append(None)

        if verbose and ((i + 1) % 50 == 0 or i + 1 == len(files)):
            elapsed = timer.time() - t_start
            rate    = (i + 1) / elapsed
            eta     = (len(files) - i - 1) / rate if rate > 0 else 0
            avg_c   = float(np.mean([len(c) for c in all_components]))
            print(f"  [{i+1:4d}/{len(files)}] {elapsed:.0f}s elapsed  "
                  f"~{eta:.0f}s remaining  avg {avg_c:.1f} comps/burst")

    valid_ts = [ts for ts in timestamps if ts is not None]
    if valid_ts:
        t0 = valid_ts[0]
        time_days = np.array(
            [(ts - t0).total_seconds() / 86400 if ts else np.nan for ts in timestamps])
    else:
        time_days = np.arange(len(files), dtype=float)

    return {
        'components':   all_components,
        'time_days':    time_days,
        'timestamps':   timestamps,
        'fine_depths':  fine_depths,
        'depth_start':  depth_start,
        'depth_end':    depth_end,
        'oversample':   oversample,
        'lambda_alpha': lambda_alpha,
    }


# ════════════════════════════════════════════════════════════════════
#  Synthetic Validation (Step 3)
# ════════════════════════════════════════════════════════════════════

def run_synthetic_test(save_path: str = 'output/apres/cs_synthetic_validation.png'):
    """
    Validate compressed sensing against FFT and CLEAN on synthetic data.

    Three tests:
    1. Super-resolution:  Two reflectors 0.5× Rayleigh limit apart
    2. Amplitude accuracy: 5 reflectors spanning 20 dB dynamic range
    3. Noise robustness:   Recovery at SNR = 40, 20, 10, 0 dB

    Produces a multi-panel comparison figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from apres.clean import clean_1d, precompute_psf_bank

    rng = np.random.default_rng(42)
    f_lo, f_hi = SUBBANDS['full']
    n_freq = 500
    freqs = np.linspace(f_lo, f_hi, n_freq)
    win = blackman(n_freq)

    def synth_spectrum(depth_axis, reflector_depths, reflector_amps, snr_db=30):
        """Generate synthetic range profile from known reflectors."""
        tau_axis = 2.0 * depth_axis / c_ice
        spec = np.zeros(len(depth_axis), dtype=np.complex128)
        for d, a in zip(reflector_depths, reflector_amps):
            tau_r = 2.0 * d / c_ice
            dt = tau_axis - tau_r
            ph = -2j * np.pi * freqs[np.newaxis, :] * dt[:, np.newaxis]
            psf = np.exp(ph).dot(win)
            spec += a * psf / np.max(np.abs(psf))
        # Add noise
        signal_power = np.mean(np.abs(spec) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(spec))
                                             + 1j * rng.standard_normal(len(spec)))
        return spec + noise

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # ── Test 1: Super-resolution ──────────────────────────────────
    # Two reflectors separated by 0.5 × Rayleigh limit
    center = 500.0
    sep = RAYLEIGH_LIMIT * 0.5  # ~0.21 m
    true_depths = [center - sep / 2, center + sep / 2]
    true_amps = [1.0, 0.8]
    depth_axis = np.arange(center - 5, center + 5, RAYLEIGH_LIMIT / 8)
    N = len(depth_axis)

    spec = synth_spectrum(depth_axis, true_depths, true_amps, snr_db=30)

    # FFT (just the magnitude of the spectrum itself)
    fft_amp = np.abs(spec)

    # CLEAN
    psf_func = precompute_psf_bank(depth_axis, f_lo, f_hi)
    clean_comps, clean_resid = clean_1d(spec, depth_axis, f_lo, f_hi,
                                         psf_func=psf_func,
                                         max_components=10, threshold_db=-30)

    # CS
    cs_comps, cs_resid, cs_info = cs_1d(spec, depth_axis, f_lo, f_hi,
                                         oversample=8, lambda_alpha=0.05,
                                         max_iter=1000, verbose=False)

    # Plot: spectrum + recovered positions
    ax = axes[0, 0]
    ax.plot(depth_axis, 20 * np.log10(fft_amp / np.max(fft_amp) + 1e-30),
            'b-', lw=1, alpha=0.7, label='FFT')
    ax.set_title(f'Super-Resolution Test\n(sep = {sep:.2f}m, Rayleigh = {RAYLEIGH_LIMIT:.2f}m)',
                 fontsize=10)
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Amplitude (dB)')
    for d in true_depths:
        ax.axvline(d, color='green', ls='--', lw=1.5, alpha=0.7)
    ax.legend(fontsize=8)
    ax.set_xlim(center - 2, center + 2)

    ax = axes[0, 1]
    ax.plot(depth_axis, 20 * np.log10(fft_amp / np.max(fft_amp) + 1e-30),
            'b-', lw=0.8, alpha=0.4, label='FFT')
    for d, a, _ in clean_comps:
        ax.axvline(d, color='red', lw=2, alpha=0.8)
    for d in true_depths:
        ax.axvline(d, color='green', ls='--', lw=1.5, alpha=0.7)
    ax.set_title(f'CLEAN ({len(clean_comps)} components)', fontsize=10)
    ax.set_xlabel('Depth (m)')
    ax.set_xlim(center - 2, center + 2)

    ax = axes[0, 2]
    ax.plot(depth_axis, 20 * np.log10(fft_amp / np.max(fft_amp) + 1e-30),
            'b-', lw=0.8, alpha=0.4, label='FFT')
    for d, a in cs_comps:
        ax.axvline(d, color='orange', lw=2, alpha=0.8)
    for d in true_depths:
        ax.axvline(d, color='green', ls='--', lw=1.5, alpha=0.7)
    ax.set_title(f'CS FISTA ({len(cs_comps)} comps, {cs_info["n_iter"]} iters)',
                 fontsize=10)
    ax.set_xlabel('Depth (m)')
    ax.set_xlim(center - 2, center + 2)

    # ── Test 2: Amplitude accuracy ────────────────────────────────
    true_depths_2 = [500.0, 501.5, 503.5, 506.0, 509.0]
    true_amps_2 = [1.0, 0.5, 0.25, 0.1, 0.05]  # 26 dB range
    depth_axis_2 = np.arange(498, 512, RAYLEIGH_LIMIT / 8)

    spec_2 = synth_spectrum(depth_axis_2, true_depths_2, true_amps_2, snr_db=30)
    fft_amp_2 = np.abs(spec_2)

    psf_func_2 = precompute_psf_bank(depth_axis_2, f_lo, f_hi)
    clean_comps_2, _ = clean_1d(spec_2, depth_axis_2, f_lo, f_hi,
                                 psf_func=psf_func_2,
                                 max_components=10, threshold_db=-30)
    cs_comps_2, _, cs_info_2 = cs_1d(spec_2, depth_axis_2, f_lo, f_hi,
                                      oversample=8, lambda_alpha=0.05,
                                      max_iter=1000)

    ax = axes[1, 0]
    ax.plot(depth_axis_2, 20 * np.log10(fft_amp_2 / np.max(fft_amp_2) + 1e-30),
            'b-', lw=1, alpha=0.7, label='FFT')
    for d in true_depths_2:
        ax.axvline(d, color='green', ls='--', lw=1.5, alpha=0.7)
    ax.set_title('Amplitude Accuracy Test\n(5 reflectors, 26 dB range)', fontsize=10)
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Amplitude (dB)')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(depth_axis_2, 20 * np.log10(fft_amp_2 / np.max(fft_amp_2) + 1e-30),
            'b-', lw=0.8, alpha=0.4)
    for d, a, _ in clean_comps_2:
        ax.axvline(d, color='red', lw=2, alpha=0.8)
    for d in true_depths_2:
        ax.axvline(d, color='green', ls='--', lw=1.5, alpha=0.7)
    ax.set_title(f'CLEAN ({len(clean_comps_2)} comps)', fontsize=10)
    ax.set_xlabel('Depth (m)')

    ax = axes[1, 2]
    ax.plot(depth_axis_2, 20 * np.log10(fft_amp_2 / np.max(fft_amp_2) + 1e-30),
            'b-', lw=0.8, alpha=0.4)
    for d, a in cs_comps_2:
        ax.axvline(d, color='orange', lw=2, alpha=0.8)
    for d in true_depths_2:
        ax.axvline(d, color='green', ls='--', lw=1.5, alpha=0.7)
    ax.set_title(f'CS FISTA ({len(cs_comps_2)} comps)', fontsize=10)
    ax.set_xlabel('Depth (m)')

    # ── Test 3: Noise robustness ──────────────────────────────────
    snrs = [40, 20, 10, 0]
    true_d3 = [500.0, 501.0, 503.0]
    true_a3 = [1.0, 0.6, 0.3]
    depth_axis_3 = np.arange(498, 506, RAYLEIGH_LIMIT / 8)

    cs_counts = []
    clean_counts = []
    cs_errors = []
    clean_errors = []

    for snr in snrs:
        spec_3 = synth_spectrum(depth_axis_3, true_d3, true_a3, snr_db=snr)
        psf_func_3 = precompute_psf_bank(depth_axis_3, f_lo, f_hi)
        cl_3, _ = clean_1d(spec_3, depth_axis_3, f_lo, f_hi,
                            psf_func=psf_func_3, max_components=10,
                            threshold_db=-30)
        cs_3, _, _ = cs_1d(spec_3, depth_axis_3, f_lo, f_hi,
                            oversample=8, lambda_alpha=0.1,
                            max_iter=1000)

        # Count correctly detected reflectors (within 0.3m of true)
        def count_correct(comps, true_depths, tol=0.3):
            found = 0
            err = []
            for td in true_depths:
                dists = [abs(c[0] - td) for c in comps]
                if dists and min(dists) < tol:
                    found += 1
                    err.append(min(dists))
            return found, np.mean(err) if err else float('nan')

        if isinstance(cl_3[0], tuple) and len(cl_3[0]) == 3:
            cl_depths = [(d, a) for d, a, _ in cl_3]
        else:
            cl_depths = cl_3

        cn, ce = count_correct(cl_depths, true_d3)
        csn, cse = count_correct(cs_3, true_d3)
        clean_counts.append(cn)
        cs_counts.append(csn)
        clean_errors.append(ce)
        cs_errors.append(cse)

    ax = axes[2, 0]
    x_pos = np.arange(len(snrs))
    w = 0.35
    ax.bar(x_pos - w / 2, clean_counts, w, color='red', alpha=0.7, label='CLEAN')
    ax.bar(x_pos + w / 2, cs_counts, w, color='orange', alpha=0.7, label='CS')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{s} dB' for s in snrs])
    ax.set_ylabel('Correct detections (of 3)')
    ax.set_title('Noise Robustness\n(3 reflectors)', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 3.5)
    ax.axhline(3, color='green', ls='--', lw=1, alpha=0.5)

    ax = axes[2, 1]
    ax.bar(x_pos - w / 2, [e * 100 for e in clean_errors], w,
           color='red', alpha=0.7, label='CLEAN')
    ax.bar(x_pos + w / 2, [e * 100 for e in cs_errors], w,
           color='orange', alpha=0.7, label='CS')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{s} dB' for s in snrs])
    ax.set_ylabel('Mean depth error (cm)')
    ax.set_title('Depth Accuracy vs. SNR', fontsize=10)
    ax.legend(fontsize=8)

    # ── FISTA convergence example ─────────────────────────────────
    ax = axes[2, 2]
    ax.semilogy(cs_info['objectives'], 'k-', lw=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective value')
    ax.set_title(f'FISTA Convergence\n({cs_info["n_iter"]} iters, '
                 f'nnz={cs_info["n_nonzero"]})', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Compressed Sensing (FISTA/BPDN) vs FFT + CLEAN — Synthetic Validation',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSynthetic validation saved to {save_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
#  Real Data Comparison (Step 4)
# ════════════════════════════════════════════════════════════════════

def run_real_comparison(
    data_path: str,
    depth_start: float = 900.0,
    depth_end: float = 910.0,
    subband: str = 'full',
    oversample: int = 4,
    lambda_alpha: float = 0.1,
    time_index: int = 0,
    save_path: str = 'output/apres/cs_vs_clean_efz.png',
):
    """
    Run CS and FFT+CLEAN on a single time step of real ApRES data and
    produce a side-by-side comparison plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from apres.clean import clean_1d, precompute_psf_bank

    print(f"Loading {data_path}...")
    if data_path.endswith('.zarr'):
        import zarr
        root = zarr.open(data_path, 'r')
        Rcoarse = np.array(root['Rcoarse']).flatten()
        raw = np.array(root['raw_complex'])
        time_days = np.array(root['time_days']).flatten()
    else:
        from scipy.io import loadmat
        mat = loadmat(data_path)
        Rcoarse = mat['Rcoarse'].flatten()
        raw = np.array(mat['RawImageComplex'])
        time_days = mat['TimeInDays'].flatten()

    # Extract depth window
    mask = (Rcoarse >= depth_start) & (Rcoarse <= depth_end)
    depth_axis = Rcoarse[mask]
    spectrum = raw[mask, time_index].astype(np.complex128)
    N = len(depth_axis)
    print(f"  Depth window: {depth_axis[0]:.1f} – {depth_axis[-1]:.1f} m, "
          f"{N} bins, time index {time_index}")

    f_lo, f_hi = SUBBANDS[subband]

    # FFT amplitude
    fft_amp = np.abs(spectrum)
    fft_db = 20 * np.log10(fft_amp / np.max(fft_amp) + 1e-30)

    # CLEAN
    print("  Running CLEAN...")
    t0 = timer.time()
    psf_func = precompute_psf_bank(depth_axis, f_lo, f_hi)
    clean_comps, clean_resid = clean_1d(
        spectrum, depth_axis, f_lo, f_hi,
        psf_func=psf_func, max_components=30, threshold_db=-30,
    )
    t_clean = timer.time() - t0
    print(f"  CLEAN: {len(clean_comps)} components in {t_clean:.2f}s")

    # Compressed Sensing
    print(f"  Running CS (oversample={oversample}, α={lambda_alpha})...")
    t0 = timer.time()
    cs_comps, cs_resid, cs_info = cs_1d(
        spectrum, depth_axis, f_lo, f_hi,
        oversample=oversample, lambda_alpha=lambda_alpha,
        max_iter=1000, verbose=True,
    )
    t_cs = timer.time() - t0
    print(f"  CS: {len(cs_comps)} components in {t_cs:.2f}s "
          f"({cs_info['n_iter']} iters)")

    # Residual energies
    orig_energy = np.sum(np.abs(spectrum) ** 2)
    clean_resid_energy = np.sum(np.abs(clean_resid) ** 2)
    cs_resid_energy = np.sum(np.abs(cs_resid) ** 2)
    clean_explained = (1 - clean_resid_energy / orig_energy) * 100
    cs_explained = (1 - cs_resid_energy / orig_energy) * 100

    print(f"\n  Signal explained:")
    print(f"    CLEAN: {clean_explained:.1f}%")
    print(f"    CS:    {cs_explained:.1f}%")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Spectra + detected components
    ax = axes[0, 0]
    ax.plot(depth_axis, fft_db, 'b-', lw=0.8, label='FFT')
    ax.set_title('FFT Range Profile', fontsize=11)
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Amplitude (dB)')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(depth_axis, fft_db, 'b-', lw=0.5, alpha=0.4)
    for d, a, _ in clean_comps:
        adb = 20 * np.log10(np.abs(a) / np.max(fft_amp) + 1e-30)
        ax.plot(d, adb, 'rv', ms=8, alpha=0.8)
    ax.set_title(f'FFT + CLEAN ({len(clean_comps)} comps, {t_clean:.2f}s)\n'
                 f'{clean_explained:.1f}% explained', fontsize=11)
    ax.set_xlabel('Depth (m)')

    ax = axes[0, 2]
    ax.plot(depth_axis, fft_db, 'b-', lw=0.5, alpha=0.4)
    for d, a in cs_comps:
        adb = 20 * np.log10(np.abs(a) / np.max(fft_amp) + 1e-30)
        ax.plot(d, adb, 'o', color='orange', ms=6, alpha=0.8)
    ax.set_title(f'CS FISTA ({len(cs_comps)} comps, {t_cs:.2f}s)\n'
                 f'{cs_explained:.1f}% explained', fontsize=11)
    ax.set_xlabel('Depth (m)')

    # Row 2: Residuals and convergence
    clean_resid_db = 20 * np.log10(np.abs(clean_resid) / np.max(fft_amp) + 1e-30)
    cs_resid_db = 20 * np.log10(np.abs(cs_resid) / np.max(fft_amp) + 1e-30)

    ax = axes[1, 0]
    ax.plot(depth_axis, fft_db, 'b-', lw=0.5, alpha=0.3, label='Original')
    ax.plot(depth_axis, clean_resid_db, 'r-', lw=0.8, label='CLEAN residual')
    ax.plot(depth_axis, cs_resid_db, '-', color='orange', lw=0.8, label='CS residual')
    ax.set_title('Residuals', fontsize=11)
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Amplitude (dB)')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogy(cs_info['objectives'], 'k-', lw=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective')
    ax.set_title(f'FISTA Convergence ({cs_info["n_iter"]} iters)', fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    # Histogram of component depths
    if clean_comps:
        ax.hist([d for d, _, _ in clean_comps], bins=20, alpha=0.5,
                color='red', label=f'CLEAN (n={len(clean_comps)})')
    if cs_comps:
        ax.hist([d for d, _ in cs_comps], bins=40, alpha=0.5,
                color='orange', label=f'CS (n={len(cs_comps)})')
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Count')
    ax.set_title('Component Distribution', fontsize=11)
    ax.legend(fontsize=8)

    plt.suptitle(
        f'Compressed Sensing vs FFT+CLEAN — Real ApRES Data\n'
        f'{subband.capitalize()} band, {depth_start:.0f}–{depth_end:.0f}m, '
        f't={time_days[time_index]:.1f} days',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Compressed Sensing range estimation for ApRES data (FISTA/BPDN)')

    parser.add_argument('--test', action='store_true',
                        help='Run synthetic validation test')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to processed ApRES data (.zarr or .mat) for post-FFT CS')
    parser.add_argument('--raw-dir', type=str, default=None,
                        help='Path to directory of raw .DAT files for pre-FFT super-resolution CS')
    parser.add_argument('--depth-start', type=float, default=900.0)
    parser.add_argument('--depth-end', type=float, default=910.0)
    parser.add_argument('--subband', type=str, default='full',
                        choices=['full', 'low', 'high'])
    parser.add_argument('--oversample', type=int, default=4,
                        help='Dictionary oversampling factor')
    parser.add_argument('--lambda-alpha', type=float, default=0.1,
                        help='Sparsity parameter (fraction of lambda_max)')
    parser.add_argument('--time-index', type=int, default=0,
                        help='Time step / file index (for single-step mode)')
    parser.add_argument('--step', type=int, default=1,
                        help='Process every Nth file (raw mode only)')
    parser.add_argument('--compare-clean', action='store_true',
                        help='Also run CLEAN and produce comparison plots (post-FFT mode)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output plot / results path')

    args = parser.parse_args()

    if args.test:
        save = args.output or 'output/apres/cs_synthetic_validation.png'
        run_synthetic_test(save_path=save)
        return

    # ── Pre-FFT raw beat signal CS ──────────────────────────────────
    if args.raw_dir:
        if args.time_index != 0:
            # Single-burst mode: process one file and print summary
            files = sorted(Path(args.raw_dir).glob('*.DAT'))
            if args.time_index >= len(files):
                print(f"File index {args.time_index} out of range ({len(files)} files)")
                return
            dat_file = files[args.time_index]
            print(f"Pre-FFT CS on {dat_file.name} ...")
            data = fmcw_load(str(dat_file))
            comps, fine_depths, x, info = cs_raw_burst(
                data.vdat, data,
                depth_start=args.depth_start,
                depth_end=args.depth_end,
                oversample=args.oversample,
                lambda_alpha=args.lambda_alpha,
                verbose=True,
            )
            print(f"\n{len(comps)} components recovered ({info['n_iter']} iters):")
            for d, a in comps:
                print(f"  {d:.3f} m   amp={np.abs(a):.4f}   phase={np.angle(a):.3f} rad")
        else:
            # Full time-series mode
            save = args.output or 'output/apres/cs_raw_timeseries.npz'
            result = cs_raw_timeseries(
                data_dir=args.raw_dir,
                depth_start=args.depth_start,
                depth_end=args.depth_end,
                oversample=args.oversample,
                lambda_alpha=args.lambda_alpha,
                step=args.step,
                verbose=True,
            )
            out = Path(save)
            out.parent.mkdir(parents=True, exist_ok=True)
            # Flatten component list to arrays for NPZ storage
            t_arr, d_arr, a_arr = [], [], []
            for ti, comps in enumerate(result['components']):
                for d, a in comps:
                    t_arr.append(result['time_days'][ti])
                    d_arr.append(d)
                    a_arr.append(a)
            np.savez(str(out),
                     time_days=np.array(t_arr),
                     depths=np.array(d_arr),
                     amplitudes=np.array(a_arr, dtype=np.complex64),
                     fine_depths=result['fine_depths'])
            print(f"\nSaved {len(t_arr)} total components → {out}")
        return

    # ── Post-FFT CS (existing paths) ────────────────────────────────
    if args.data and args.compare_clean:
        save = args.output or 'output/apres/cs_vs_clean_efz.png'
        run_real_comparison(
            data_path=args.data,
            depth_start=args.depth_start,
            depth_end=args.depth_end,
            subband=args.subband,
            oversample=args.oversample,
            lambda_alpha=args.lambda_alpha,
            time_index=args.time_index,
            save_path=save,
        )
        return

    if args.data:
        print("Use --compare-clean for comparison against CLEAN, "
              "or --test for synthetic validation.")
        return

    parser.print_help()


if __name__ == '__main__':
    main()
