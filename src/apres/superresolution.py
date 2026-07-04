import numpy as np
from scipy.signal import decimate
import scipy.linalg as la
from apres.io import ApRESData

def _itc_likelihood(s: np.ndarray, N: int) -> tuple[np.ndarray, int]:
    """Shared likelihood computation for MDL/AIC family criteria.

    Returns (log_lik_array, L) where log_lik[k] is the Wax–Kailath
    log-likelihood term for assuming k signals among the L singular
    values.  log_lik[L] is set to 0 by convention (no noise eigenvalues).
    """
    L = len(s)
    eigenvalues = s ** 2
    log_lik = np.full(L + 1, np.nan)
    for k in range(L):
        noise_eigs = eigenvalues[k:]
        m = L - k
        if m <= 0:
            log_lik[k] = 0.0
            continue
        ari = float(np.mean(noise_eigs))
        geo = float(np.exp(np.mean(np.log(noise_eigs + 1e-30))))
        if ari <= 0 or geo <= 0:
            log_lik[k] = np.inf
            continue
        ratio = geo / ari
        log_lik[k] = -N * m * np.log(ratio)
    log_lik[L] = 0.0
    return log_lik, L


def estimate_mdl(s: np.ndarray, N: int) -> int:
    """Wax–Kailath MDL (BIC-style) order estimate.

    MDL(k) = −N(L−k) log[ geo_mean(λ_k…) / ari_mean(λ_k…) ]
             + ½ k(2L−k) log(N)

    Picks k that minimises MDL.

    Bug-fix history: an earlier version initialised mdl_vals to 0.0 and
    skipped the k = L-1 case via `continue`, leaving that slot at zero —
    so argmin always returned L-1 (the maximum order).  We now compute
    every k explicitly and initialise to +inf.
    """
    log_lik, L = _itc_likelihood(s, N)
    mdl = np.full(L + 1, np.inf)
    for k in range(L + 1):
        if not np.isfinite(log_lik[k]):
            continue
        n_free  = k * (2 * L - k)
        penalty = 0.5 * n_free * np.log(N)
        mdl[k]  = log_lik[k] + penalty
    return int(np.argmin(mdl))


def estimate_aic(s: np.ndarray, N: int) -> int:
    """Akaike Information Criterion order estimate.

    AIC(k) = −N(L−k) log[geo/ari] + k(2L−k)

    Lighter penalty than MDL — tends to over-estimate model order,
    useful as an upper bound or when you'd rather false-positive than
    false-negative on a weak signal.
    """
    log_lik, L = _itc_likelihood(s, N)
    aic = np.full(L + 1, np.inf)
    for k in range(L + 1):
        if not np.isfinite(log_lik[k]):
            continue
        n_free = k * (2 * L - k)
        aic[k] = log_lik[k] + n_free
    return int(np.argmin(aic))


def estimate_aicc(s: np.ndarray, N: int) -> int:
    """Corrected AIC for finite samples (Hurvich–Tsai).

    Adds a 2·n_free·(n_free+1)/(N − n_free − 1) correction; defaults to
    AIC when N is large.  Avoids the AIC's tendency to over-fit on
    short signals.
    """
    log_lik, L = _itc_likelihood(s, N)
    aicc = np.full(L + 1, np.inf)
    for k in range(L + 1):
        if not np.isfinite(log_lik[k]):
            continue
        n_free  = k * (2 * L - k)
        denom   = N - n_free - 1
        corr    = (2.0 * n_free * (n_free + 1) / denom) if denom > 0 else np.inf
        aicc[k] = log_lik[k] + n_free + corr
    return int(np.argmin(aicc))


def estimate_gap(s: np.ndarray, N: int | None = None) -> int:
    """Eigenvalue-gap detector: pick k at the largest jump in log σ.

    Robust to noise level, requires no statistical assumptions.  Returns
    k = (argmax of −Δ log σ) + 1.  Tends to be conservative (small k)
    if the gap is at the very first eigenvalue.
    """
    if len(s) < 2:
        return len(s)
    log_s = np.log(np.maximum(s, 1e-30))
    drops = log_s[:-1] - log_s[1:]               # positive
    return int(np.argmax(drops)) + 1


def estimate_threshold(s: np.ndarray, eps: float = 0.01) -> int:
    """Numerical-rank order estimate.

    Smallest k such that σ_{k+1} < eps · σ_max.  Equivalent to the
    pseudo-inverse rule used in linear regression.  No statistical
    interpretation, but the most predictable behaviour.
    """
    if len(s) == 0:
        return 0
    s_norm = s / max(s[0], 1e-30)
    below  = s_norm < eps
    if not below.any():
        return len(s)
    return int(np.argmax(below))


def estimate_order(s: np.ndarray, N: int, method: str = 'mdl',
                    threshold_eps: float = 0.01) -> int:
    """Dispatch model-order estimation by method name.

    method ∈ {'mdl', 'aic', 'aicc', 'gap', 'threshold'}.
    """
    method = method.lower()
    if   method == 'mdl':       return estimate_mdl(s, N)
    elif method == 'aic':       return estimate_aic(s, N)
    elif method == 'aicc':      return estimate_aicc(s, N)
    elif method == 'gap':       return estimate_gap(s)
    elif method == 'threshold': return estimate_threshold(s, eps=threshold_eps)
    raise ValueError(f"Unknown method: {method}")

def matrix_pencil_poles(x: np.ndarray, M: int = None, L_param: int = None,
                          order_method: str = 'mdl',
                          threshold_eps: float = 0.01) -> tuple:
    """
    Extract frequencies and amplitudes using the Matrix Pencil Method.

    Args:
        x: Complex 1D array of signal samples
        M: Number of signals/targets (None to auto-estimate)
        L_param: Pencil parameter, default is N//3
        order_method: when M is None, criterion used to pick the order.
            'mdl'        — Wax–Kailath MDL (BIC-style)
            'aic'        — Akaike Information Criterion
            'aicc'       — finite-sample-corrected AIC
            'gap'        — largest gap in log σ (no statistical model)
            'threshold'  — σ_k < eps · σ_max  (eps from threshold_eps)
        threshold_eps: relative cutoff for the 'threshold' method.

    Returns:
        poles, amplitudes
    """
    N = len(x)
    if L_param is None:
        L_param = N // 3

    # Standard choice is N/3 <= L <= 2N/3. We'll use L_param.
    # Form Hankel matrix Y of size (N-L) x (L+1)
    rows = N - L_param
    cols = L_param + 1

    Y = np.zeros((rows, cols), dtype=np.complex128)
    for i in range(rows):
        Y[i, :] = x[i:i+cols]

    # SVD of Y
    U, s, Vh = la.svd(Y, full_matrices=False)
    V = Vh.T.conj()

    if M is None:
        M = estimate_order(s, N, method=order_method,
                            threshold_eps=threshold_eps)

    M = max(1, min(M, cols - 2)) # Ensure at least 1, max cols-2
        
    # Extract signal subspace
    Vs = V[:, :M]
    
    # Create shifted matrices
    V1 = Vs[:-1, :]
    V2 = Vs[1:, :]
    
    # Calculate generalized eigenvalues (poles)
    V1_pinv = la.pinv(V1)
    poles = la.eigvals(V1_pinv @ V2)
    
    # Estimate amplitudes via Least Squares: x = Z * a
    # Z matrix size: N x M
    Z = np.zeros((N, M), dtype=np.complex128)
    for n in range(N):
        Z[n, :] = poles**n
        
    amps, _, _, _ = la.lstsq(Z, x)
    
    return poles, amps

def fmcw_matrix_pencil(data: ApRESData, depth_min: float, depth_max: float,
                        M: int = None, order_method: str = 'mdl',
                        return_svd: bool = False):
    """
    Apply Matrix Pencil Method to a specific depth window by digitally
    downconverting and decimating the raw ApRES signal.

    Args:
        data: ApRESData object
        depth_min: Minimum depth of window (m)
        depth_max: Maximum depth of window (m)
        M: Number of expected sub-resolution targets (e.g., 1 for lake surface). None for MDL.
        order_method: when M is None, criterion used to pick the order
            ('mdl' default; also 'aic', 'aicc', 'gap', 'threshold').
        return_svd: if True, also return the decimated signal and its
            Hankel SVD for downstream order-selection analysis.

    Returns:
        dict: containing specific target depths and powers (and optionally
        'decimated_signal' + 'singular_values' when return_svd=True)
    """
    # 1. Take mean across subbursts to improve SNR before MPM
    # Shape of vdat is (n_subbursts, n_samples)
    vif = np.mean(data.vdat, axis=0)
    vif = vif - np.mean(vif)
    
    N = len(vif)
    fs = data.sampling_freq
    dt = 1.0 / fs
    t = np.arange(N) * dt
    
    # 2. Physics logic: Beat frequency corresponds to range
    # fb = (K / pi / c_ice) * R
    K_hz_per_sec = data.chirp_grad / (2 * np.pi) 
    # K in datastructure is angular rad/s/s. K_hz_per_sec = B / T
    
    # Exact B from data
    B = data.bandwidth
    T = data.chirp_time
    K_hz_per_sec = B / T
    
    # Map depth to frequency
    f_min = (2 * K_hz_per_sec / data.ci) * depth_min
    f_max = (2 * K_hz_per_sec / data.ci) * depth_max
    
    f_center = (f_min + f_max) / 2.0
    f_width = f_max - f_min
    
    # 3. Digital Downconversion (shift f_center to DC)
    # Multiply by exp(-j*2*pi*f_center*t)
    analytic_signal = vif * np.exp(-1j * 2 * np.pi * f_center * t)

    # 3b. Tight low-pass filter matched to window bandwidth
    # The DDC shifts the window to baseband but doesn't reject out-of-band
    # reflectors.  A Butterworth filter at f_width/2 ensures only in-window
    # frequencies survive, so MPM poles map to physical depths.
    from scipy.signal import butter, sosfiltfilt
    f_cutoff = f_width / 2 * 1.2   # slight margin above Nyquist of window
    sos = butter(6, f_cutoff, btype='low', fs=fs, output='sos')
    analytic_signal = sosfiltfilt(sos, analytic_signal)

    # 4. Decimation
    fs_new_target = max(f_width * 8, 10.0) # 8× oversampling
    decimation_factor = int(fs / fs_new_target)
    
    if decimation_factor > 1:
        # Avoid decimation > 13 in a single step per scipy recommendations.
        # We'll do it in multiple steps if necessary.
        factors = []
        df = decimation_factor
        while df > 10:
            factors.append(10)
            df = df // 10
        if df > 1:
            factors.append(df)
            
        decimated_signal = analytic_signal
        for d in factors:
            decimated_signal = decimate(decimated_signal, d, zero_phase=True)
            
        fs_new = fs / np.prod(factors)
    else:
        decimated_signal = analytic_signal
        fs_new = fs
        
    # Crop edges to remove decimation filter transients (ringing)
    crop_len = int(len(decimated_signal) * 0.15)
    if crop_len > 0:
        decimated_signal = decimated_signal[crop_len:-crop_len]
        
    # 5. Matrix Pencil Method
    poles, amps = matrix_pencil_poles(decimated_signal, M=M,
                                       order_method=order_method)

    # Optional: expose the SVD of the Hankel matrix for downstream
    # order-selection analysis.
    _svd_data = None
    if return_svd:
        N_sig = len(decimated_signal)
        L_p = N_sig // 3
        Y = np.zeros((N_sig - L_p, L_p + 1), dtype=np.complex128)
        for i in range(N_sig - L_p):
            Y[i, :] = decimated_signal[i:i + L_p + 1]
        _svd_data = dict(
            decimated_signal=decimated_signal,
            singular_values=la.svd(Y, compute_uv=False),
            N=N_sig,
        )

    # Reconstruct signal from all poles and compute explained variance (R²)
    N_dec = len(decimated_signal)
    Z_full = np.zeros((N_dec, len(poles)), dtype=np.complex128)
    for n in range(N_dec):
        Z_full[n, :] = poles ** n
    x_hat = Z_full @ amps
    ss_res = np.sum(np.abs(decimated_signal - x_hat) ** 2)
    ss_tot = np.sum(np.abs(decimated_signal - np.mean(decimated_signal)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # 6. Translate poles back to absolute frequencies and ranges
    target_depths = []
    target_powers = []
    target_phases = []
    target_amps = []

    dt_new = 1.0 / fs_new
    for i in range(len(poles)):
        # pole z = exp(s * dt_new) = exp((alpha + j 2 pi f) * dt_new)
        # f = angle(pole) / (2 pi dt_new)
        f_offset = np.angle(poles[i]) / (2 * np.pi * dt_new)

        f_abs = f_center + f_offset

        # Convert absolute frequency back to depth
        depth = f_abs * data.ci / (2 * K_hz_per_sec)

        # The DDC sign convention inverts depth within the window;
        # mirror around the window centre to restore correct ordering.
        depth = depth_min + depth_max - depth

        # Keep only poles within the requested depth window (strict).
        # A small ±2 m margin allows for minor aliasing at window edges.
        if not (depth_min - 2.0 <= depth <= depth_max + 2.0):
            continue

        # Reject poles with |z| far from 1: physical sinusoids are undamped
        # (|z| ≈ 1).  Poles with |z| significantly different from unity are
        # exponentially growing/decaying artefacts.
        mag = abs(poles[i])
        if not (0.9 <= mag <= 1.1):
            continue

        target_depths.append(depth)

        # Relative power is sufficient for ranking/filtering.
        power = 10 * np.log10(np.abs(amps[i])**2 + 1e-15)
        target_powers.append(power)

        phase = np.angle(amps[i])
        target_phases.append(phase)

        target_amps.append(amps[i])

    # Sort by depth
    sort_idx = np.argsort(target_depths)

    out = {
        'depths': np.array(target_depths)[sort_idx],
        'powers': np.array(target_powers)[sort_idx],
        'phases': np.array(target_phases)[sort_idx],
        'amps': np.array(target_amps)[sort_idx],  # complex amplitudes of kept poles
        'poles': poles,
        'M_used': len(poles),
        'r_squared': r_squared,          # fraction of signal power explained by all M poles
    }
    if return_svd and _svd_data is not None:
        out['decimated_signal'] = _svd_data['decimated_signal']
        out['singular_values']  = _svd_data['singular_values']
        out['N_decimated']      = _svd_data['N']
    return out


