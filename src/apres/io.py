"""
ApRES FMCW Radar Processing in Python

This module provides Python implementations of the MATLAB ApRES processing 
functions for phase-sensitive FMCW radar data.

Based on MATLAB code by:
- Craig Stewart (fmcw_load, fmcw_range, fmcw_phase2range)
- Keith Nicholls (RMB2/RMB5 format updates)
- Nicole Bienert, Sean Peters, Paul Summers (mainCode_simple)

References:
- Brennan et al. (2013) - Phase-sensitive FMCW radar processing

Author: Python translation for SiegVent2023 project
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import re


@dataclass
class ApRESData:
    """Container for ApRES burst data and metadata."""
    filename: str
    vdat: np.ndarray  # Voltage data [NSubBursts, N_ADC_SAMPLES]
    chirp_time: float  # Chirp time in seconds
    chirp_num: int  # Number of chirps
    sampling_freq: float  # Sampling frequency (Hz)
    f0: float  # Start frequency (Hz)
    f1: float  # End frequency (Hz)
    bandwidth: float  # Bandwidth (Hz)
    chirp_grad: float  # Chirp gradient K (rad/s/s)
    er: float  # Relative permittivity
    ci: float  # Wave speed in medium (m/s)
    lambdac: float  # Center wavelength in medium (m)
    t: np.ndarray  # Time vector
    file_format: int  # File format version
    # Additional header info
    n_adc_samples: int
    n_subbursts: int
    time_stamp: Optional[datetime] = None


def detect_file_format(filename: str) -> int:
    """
    Determine ApRES file format from burst header.
    
    Returns:
        fmt: 5 = RMB2 after Oct 2014 (SW_Issue=)
             4 = After Oct 2013 (SubBursts in burst:)
             3 = Jan 2013 (*** Burst Header ***)
             2 = Prototype Nov 2012 (RADAR TIME)
    """
    with open(filename, 'r', errors='ignore') as f:
        header = f.read(1500)
    
    if 'SW_Issue=' in header:
        return 5
    elif 'SubBursts in burst:' in header:
        return 4
    elif '*** Burst Header ***' in header:
        return 3
    elif 'RADAR TIME' in header:
        return 2
    else:
        raise ValueError(f"Unknown file format: {filename}")


def parse_header_value(header: str, key: str, dtype=int, first_only: bool = True):
    """Extract a value from the header given a key.
    
    Args:
        header: Header text
        key: Parameter name to find
        dtype: Type to convert value to
        first_only: If True and value is comma-separated, return only first value
    """
    pattern = rf'{key}=([^\n]+)'
    match = re.search(pattern, header)
    if match:
        value = match.group(1).strip().strip('"')
        # Handle comma-separated values
        if ',' in value and first_only:
            value = value.split(',')[0]
        try:
            return dtype(value)
        except (ValueError, TypeError):
            return value
    return None


def load_burst_rmb5(filename: str, burst_num: int = 1) -> Tuple[np.ndarray, dict]:
    """
    Load burst data from RMB5 format file.
    
    Based on LoadBurstRMB5.m by Keith Nicholls (2014-10-22)
    
    Args:
        filename: Path to .DAT file
        burst_num: Which burst to load (1-indexed)
    
    Returns:
        vdat: Voltage data array [NSubBursts, N_ADC_SAMPLES]
        header_info: Dictionary with header parameters
    """
    with open(filename, 'rb') as f:
        file_content = f.read()
    
    # Find all burst markers
    search_str = b'*** Burst Header ***'
    burst_locs = []
    start = 0
    while True:
        loc = file_content.find(search_str, start)
        if loc == -1:
            break
        burst_locs.append(loc)
        start = loc + 1
    
    if burst_num > len(burst_locs):
        raise ValueError(f"Burst {burst_num} not found. File has {len(burst_locs)} bursts.")
    
    # Read header for the requested burst
    burst_start = burst_locs[burst_num - 1]
    
    # Determine header end - look for "*** End Header ***" or the start of binary data
    # The header ends after "End Header" line
    header_end_str = b'*** End Header ***'
    header_end = file_content.find(header_end_str, burst_start)
    if header_end == -1:
        # Fallback: estimate header size
        header_end = burst_start + 1000
    else:
        header_end += len(header_end_str) + 2  # Skip past the marker and newline
    
    header_text = file_content[burst_start:header_end].decode('ascii', errors='ignore')
    
    # Parse header values
    header_info = {}
    header_info['n_adc_samples'] = parse_header_value(header_text, 'N_ADC_SAMPLES', int)
    header_info['n_subbursts'] = parse_header_value(header_text, 'NSubBursts', int)
    header_info['attenuator1'] = parse_header_value(header_text, 'Attenuator 1', float)
    header_info['attenuator2'] = parse_header_value(header_text, 'Attenuator 2', float)
    header_info['af_gain'] = parse_header_value(header_text, 'AFGain', int)
    header_info['tx_ant'] = parse_header_value(header_text, 'TxAnt', int)
    header_info['rx_ant'] = parse_header_value(header_text, 'RxAnt', int)
    header_info['reg00'] = parse_header_value(header_text, 'Reg00', str)
    header_info['reg01'] = parse_header_value(header_text, 'Reg01', str)
    
    # Parse start/stop frequencies (more reliable than DDS registers)
    header_info['start_freq'] = parse_header_value(header_text, 'StartFreq', float)
    header_info['stop_freq'] = parse_header_value(header_text, 'StopFreq', float)
    
    # Parse time stamp
    time_match = re.search(r'Time stamp=(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', header_text)
    if time_match:
        header_info['time_stamp'] = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
    
    # Default values if not in header
    if header_info['n_adc_samples'] is None:
        header_info['n_adc_samples'] = 40000
    if header_info['n_subbursts'] is None:
        header_info['n_subbursts'] = 100
    
    # Read voltage data (binary)
    n_samples = header_info['n_adc_samples']
    n_subbursts = header_info['n_subbursts']
    
    # Data starts after header
    data_start = header_end
    bytes_per_sample = 2  # 16-bit integers
    total_samples = n_samples * n_subbursts
    
    # Read raw data as uint16
    raw_data = np.frombuffer(
        file_content[data_start:data_start + total_samples * bytes_per_sample],
        dtype=np.uint16
    )
    
    # Reshape to [NSubBursts, N_ADC_SAMPLES]
    if len(raw_data) >= total_samples:
        vdat = raw_data[:total_samples].reshape(n_subbursts, n_samples).astype(np.float64)
        # Convert to voltage (centered around zero)
        vdat = vdat - np.mean(vdat)
    else:
        raise ValueError(f"Insufficient data: expected {total_samples}, got {len(raw_data)}")
    
    return vdat, header_info


def get_dds_parameters(reg00: str, reg01: str) -> Tuple[float, float, float, float]:
    """
    Extract DDS frequency parameters from hex register values.
    
    Based on fmcw_ParametersRMB2.m
    
    Returns:
        f0: Start frequency (Hz)
        f1: End frequency (Hz)
        T: Chirp time (s)
        fc: Center frequency (Hz)
    """
    # DDS clock frequency
    fs_dds = 1e9  # 1 GHz
    
    # Default values (typical for ApRES)
    f0 = 2e8  # 200 MHz
    f1 = 4e8  # 400 MHz
    T = 1.0   # 1 second chirp
    
    if reg00 and reg01:
        try:
            # Reg00 contains start frequency
            # Reg01 contains frequency step
            reg00_val = int(reg00, 16)
            reg01_val = int(reg01, 16)
            
            # Extract frequency tuning word (48-bit)
            ftw0 = reg00_val & 0xFFFFFFFFFFFF
            f0 = ftw0 * fs_dds / (2**48)
            
            # Frequency step calculation
            df = reg01_val * fs_dds / (2**48)
            # Assume 1 second chirp with ~200 MHz bandwidth
            f1 = f0 + 2e8  # Default bandwidth
        except:
            pass
    
    fc = (f0 + f1) / 2
    return f0, f1, T, fc


def fmcw_load(filename: str, er: float = 3.18) -> ApRESData:
    """
    Load FMCW radar burst data and metadata.
    
    Based on fmcw_load.m by Craig Stewart
    
    Args:
        filename: Path to .DAT file
        er: Relative permittivity of medium (default 3.18 for ice)
    
    Returns:
        ApRESData object containing voltage data and metadata
    """
    c = 3e8  # Speed of light (m/s)
    
    # Detect file format
    fmt = detect_file_format(filename)
    
    if fmt == 5:
        vdat, header = load_burst_rmb5(filename)
    else:
        raise NotImplementedError(f"File format {fmt} not yet implemented")
    
    # Get frequency parameters - prefer StartFreq/StopFreq from header
    start_freq = header.get('start_freq')
    stop_freq = header.get('stop_freq')
    
    if start_freq and stop_freq:
        f0 = start_freq
        f1 = stop_freq
        T = 1.0  # Chirp time (typically 1 second)
        fc = (f0 + f1) / 2
    else:
        # Fallback to DDS register parsing
        f0, f1, T, fc = get_dds_parameters(
            header.get('reg00', ''),
            header.get('reg01', '')
        )
    
    # Calculate derived parameters
    bandwidth = f1 - f0
    fs = 40000  # Sampling frequency (Hz) - fixed for ApRES
    
    # Chirp gradient K = 2*pi*B/T (rad/s/s)
    K = 2 * np.pi * bandwidth / T
    
    # Wave speed in medium
    ci = c / np.sqrt(er)
    
    # Center wavelength in medium
    lambdac = ci / fc
    
    # Time vector
    n_samples = header['n_adc_samples']
    t = np.arange(n_samples) / fs
    
    return ApRESData(
        filename=filename,
        vdat=vdat,
        chirp_time=T,
        chirp_num=header['n_subbursts'],
        sampling_freq=fs,
        f0=f0,
        f1=f1,
        bandwidth=bandwidth,
        chirp_grad=K,
        er=er,
        ci=ci,
        lambdac=lambdac,
        t=t,
        file_format=fmt,
        n_adc_samples=header['n_adc_samples'],
        n_subbursts=header['n_subbursts'],
        time_stamp=header.get('time_stamp')
    )


def fmcw_range(
    data: ApRESData,
    pad_factor: int = 2,
    max_range: float = 2000,
    window_func: Callable = np.blackman,
    subband: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process FMCW radar data to obtain range profile.
    
    Based on fmcw_range.m by Craig Stewart, following Brennan et al. 2013.
    
    Args:
        data: ApRESData object from fmcw_load
        pad_factor: Zero-padding factor for FFT
        max_range: Maximum range to compute (m)
        window_func: Window function (default: Blackman)
        subband: None for full, 'low' for first half, 'high' for second half
    
    Returns:
        Rcoarse: Coarse range vector (m)
        Rfine: Fine range correction for each burst [NSubBursts, N_bins]
        spec_cor: Phase-corrected spectra [NSubBursts, N_bins]
        spec: Raw spectra [NSubBursts, N_bins]
    """
    vdat = data.vdat
    n_subbursts, n_samples = vdat.shape
    
    # Calculate number of FFT points
    nfft = n_samples * pad_factor
    
    # Generate window based on subband selection
    if subband == 'low':
        half_n = n_samples // 2
        half_win = window_func(half_n)
        win = np.zeros(n_samples, dtype=float)
        win[:half_n] = half_win
    elif subband == 'high':
        half_n = n_samples // 2
        half_win = window_func(half_n)
        win = np.zeros(n_samples, dtype=float)
        win[-half_n:] = half_win
    else:
        win = window_func(n_samples)
        
    win_rms = np.sqrt(np.mean(win**2)) if np.mean(win**2) > 0 else 1.0  # RMS of window for scaling
    
    # Phase center offset - shift signal so phase center is at t=0
    # For subbands, we still shift by n_samples//2 so the Rcoarse alignment is exactly the same
    xn = n_samples // 2
    
    # Extract parameters
    K = data.chirp_grad
    fs = data.sampling_freq
    ci = data.ci
    B = data.bandwidth
    fc = (data.f0 + data.f1) / 2  # Center frequency
    lambdac = data.lambdac
    
    # Number of frequency bins to keep
    nf = nfft // 2
    
    # Coarse range vector: R = n * ci / (2 * B * p)  (eq 14 rearranged)
    n_vec = np.arange(nf)
    Rcoarse = n_vec * ci / (2 * B * pad_factor)
    
    # Limit to max range
    max_bin = np.searchsorted(Rcoarse, max_range) + 1
    max_bin = min(max_bin, nf)
    
    # Reference phase for bin center correction (eq 17 from Brennan et al. 2013)
    # phi_ref = 2*pi*fc*tau - K*tau^2/2, where tau = n/(B*p)
    tau = n_vec[:max_bin] / (B * pad_factor)
    phiref = 2 * np.pi * fc * tau - (K * tau**2) / 2
    
    # Preallocate output arrays
    spec = np.zeros((n_subbursts, max_bin), dtype=complex)
    spec_cor = np.zeros((n_subbursts, max_bin), dtype=complex)
    
    # Process each subburst
    for ii in range(n_subbursts):
        vif = vdat[ii, :] - np.mean(vdat[ii, :])  # De-mean
        vif = win * vif  # Apply window
        
        # Zero-pad to nfft length
        vif_pad = np.zeros(nfft)
        vif_pad[:n_samples] = vif
        
        # Circular shift so phase center is at start (critical for phase accuracy)
        vif_pad = np.roll(vif_pad, -xn)
        
        # FFT with scaling for padding and window
        fftvif = np.fft.fft(vif_pad) * np.sqrt(2 * pad_factor) / nfft
        fftvif = fftvif / win_rms  # Scale for window RMS
        
        # Extract positive frequencies up to max_bin
        spec[ii, :] = fftvif[:max_bin]
        
        # Apply phase correction (conjugate of reference phase)
        comp = np.exp(-1j * phiref)
        spec_cor[ii, :] = comp * fftvif[:max_bin]
    
    # Crop Rcoarse to max_bin
    Rcoarse = Rcoarse[:max_bin]
    
    # Calculate fine range from corrected phase (eq 15 with full correction term)
    # Rfine = phi / ((4*pi/lambdac) - (4*Rcoarse*K/ci^2))
    Rfine = np.zeros((n_subbursts, max_bin), dtype=float)
    
    for i in range(max_bin):
        rc = Rcoarse[i]
        phi = np.angle(spec_cor[:, i])
        
        # Full equation including chirp gradient term (from fmcw_phase2range.m)
        denom = (4 * np.pi / lambdac) - (4 * rc * K / ci**2)
        if abs(denom) > 1e-10:
            Rfine[:, i] = phi / denom
        else:
            Rfine[:, i] = lambdac * phi / (4 * np.pi)
    
    return Rcoarse, Rfine, spec_cor, spec


def process_apres_file(
    filename: str,
    er: float = 3.18,
    max_range: float = 2000,
    pad_factor: int = 2,
    subband: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, ApRESData]:
    """
    High-level function to process a single ApRES file.
    
    Args:
        filename: Path to .DAT file
        er: Relative permittivity (default 3.18 for ice)
        max_range: Maximum range to compute (m)
        pad_factor: FFT zero-padding factor
        subband: Optional subband extraction ('low' or 'high')
    
    Returns:
        Rcoarse: Range vector (m)
        spec_avg: Average spectrum magnitude
        data: Full ApRESData object
    """
    data = fmcw_load(filename, er)
    Rcoarse, Rfine, spec_cor, spec = fmcw_range(data, pad_factor, max_range, subband=subband)
    
    # Average across all subbursts
    spec_avg = np.mean(np.abs(spec_cor), axis=0)
    
    return Rcoarse, spec_avg, data


def process_timeseries(
    data_folder: str,
    er: float = 3.18,
    max_range: float = 2000,
    pad_factor: int = 8,
    verbose: bool = True,
    step: int = 1,
    keep_complex: bool = False,
    subband: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Process all ApRES files in a folder to create a time series.
    
    Based on mainCode_simple.m workflow.
    
    Args:
        data_folder: Path to folder containing .DAT files
        er: Relative permittivity
        max_range: Maximum range to compute
        pad_factor: FFT zero-padding factor (default 8 for smooth results)
                   Higher values = finer range bins, less sawtooth artifact
                   pad_factor=8 gives ~0.05m bins vs ~0.21m for pad_factor=2
        verbose: Print progress
        step: Process every Nth file (default 1 = all files)
        keep_complex: Whether to keep complex array
        subband: 'low' or 'high' or None for processing sub-bands
    
    Returns:
        range_img: 2D array of range profiles [n_bins, n_files]
        rfine_avg: 2D array of average fine range corrections [n_bins, n_files]
        Rcoarse: Range vector (m)
        time_days: Time in days since first measurement
        timestamps: List of datetime objects
        (optional) range_img_complex: 2D complex array [n_bins, n_files]
    """
    data_path = Path(data_folder)
    all_files = sorted(data_path.glob('*.DAT'))
    files = all_files[::step]
    
    if len(files) == 0:
        raise ValueError(f"No .DAT files found in {data_folder}")
    
    if verbose:
        print(f"Found {len(all_files)} ApRES files, processing {len(files)} (every {step})")
        print(f"Using pad_factor={pad_factor} (bin spacing ~{0.21*2/pad_factor:.3f} m)")
    
    # Process first file to get dimensions
    data = fmcw_load(str(files[0]), er)
    Rcoarse, Rfine, spec_cor, spec = fmcw_range(data, pad_factor, max_range, subband=subband)
    n_bins = len(Rcoarse)
    n_files = len(files)
    
    # Initialize arrays
    range_img = np.zeros((n_bins, n_files))
    rfine_avg = np.zeros((n_bins, n_files))
    range_img_complex = np.zeros((n_bins, n_files), dtype=np.complex128) if keep_complex else None
    range_img[:, 0] = np.mean(np.abs(spec_cor), axis=0)
    rfine_avg[:, 0] = np.mean(Rfine, axis=0)
    if keep_complex:
        range_img_complex[:, 0] = np.mean(spec_cor, axis=0)
    timestamps = [data.time_stamp]
    
    # Process remaining files
    for i, filepath in enumerate(files[1:], 1):
        if verbose and i % 50 == 0:
            print(f"Processing file {i+1}/{n_files}...")
        
        try:
            data = fmcw_load(str(filepath), er)
            Rcoarse, Rfine, spec_cor, spec = fmcw_range(data, pad_factor, max_range, subband=subband)
            range_img[:, i] = np.mean(np.abs(spec_cor), axis=0)
            rfine_avg[:, i] = np.mean(Rfine, axis=0)
            if keep_complex:
                range_img_complex[:, i] = np.mean(spec_cor, axis=0)
            timestamps.append(data.time_stamp)
        except Exception as e:
            if verbose:
                print(f"Error processing {filepath.name}: {e}")
            timestamps.append(None)
    
    # Calculate time in days from first measurement
    valid_ts = [t for t in timestamps if t is not None]
    if len(valid_ts) > 0:
        t0 = valid_ts[0]
        time_days = np.array([
            (t - t0).total_seconds() / 86400 if t else np.nan 
            for t in timestamps
        ])
    else:
        time_days = np.arange(n_files)
    
    if keep_complex:
        return range_img, rfine_avg, Rcoarse, time_days, timestamps, range_img_complex
    return range_img, rfine_avg, Rcoarse, time_days, timestamps


def extract_fine_range(
    range_img: np.ndarray,
    rfine_avg: np.ndarray,
    Rcoarse: np.ndarray,
    min_range: float = 1000,
    lambdac: float = 0.5608,
    unwrap_phase: bool = True,
    correction_factor: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract fine range estimates from time series data.
    
    Based on RangeEstFine.m by Paul Summers.
    
    This finds the peak in each profile (above min_range), extracts the
    corresponding fine range value, and computes total range.
    
    Args:
        range_img: 2D array of range profiles [n_bins, n_files]
        rfine_avg: 2D array of fine range corrections [n_bins, n_files]
        Rcoarse: Coarse range vector (m)
        min_range: Minimum range to search for peak (m)
        lambdac: Center wavelength in medium (m), default for ice at 300 MHz
        unwrap_phase: If True, apply phase unwrapping to remove phase jumps.
                      This is the physically correct approach for continuous
                      range changes. Default True.
        correction_factor: Fraction of λ/2 to use for correction. Default 0.5
                          means corrections of λ/4, accounting for two-way path.
    
    Returns:
        range_total: Total range (coarse + fine) for each file
        range_coarse: Coarse range (peak bin) for each file
        peak_bins: Index of peak bin for each file
    """
    n_bins, n_files = range_img.shape
    half_lambda = lambdac / 2
    
    # The correction step size (default λ/4 for two-way path)
    correction_step = half_lambda * correction_factor
    
    # Find threshold bin (minimum range to search)
    thres_bin = np.searchsorted(Rcoarse, min_range)
    
    # Find peak in each profile
    peak_bins = np.argmax(range_img[thres_bin:, :], axis=0) + thres_bin
    
    # Extract coarse range and fine range at peak
    range_coarse = Rcoarse[peak_bins]
    rfine_pick = np.array([rfine_avg[peak_bins[i], i] for i in range(n_files)])
    
    # Raw total range (may have phase jumps from wrapping)
    range_total = range_coarse + rfine_pick
    
    if unwrap_phase:
        # Phase unwrapping: detect and correct phase jumps
        # The fine range Rfine is derived from phase, which wraps periodically.
        # When the true range crosses a phase boundary, there's a sudden jump.
        # We detect these jumps and accumulate corrections to remove them.
        
        # Work with differences to find discontinuities
        diff = np.diff(range_total)
        
        # Threshold for detecting a phase wrap (jump of ~λ/2 in raw data)
        # Use 0.7 * λ/2 to allow for noise while catching real wraps
        wrap_thresh = 0.7 * half_lambda
        
        # Accumulate phase corrections
        correction = np.zeros(n_files)
        cumulative = 0.0
        
        for i in range(len(diff)):
            if diff[i] > wrap_thresh:
                # Positive jump = phase wrapped down, subtract correction_step
                cumulative -= correction_step
            elif diff[i] < -wrap_thresh:
                # Negative jump = phase wrapped up, add correction_step
                cumulative += correction_step
            correction[i + 1] = cumulative
        
        range_total = range_total + correction
    
    return range_total, range_coarse, peak_bins


def save_range_results(
    filename: str,
    range_total: np.ndarray,
    time_days: np.ndarray,
    timestamps: list,
    format: str = 'both'
):
    """
    Save range time series results to file.
    
    Args:
        filename: Output filename (without extension)
        range_total: Range values (m)
        time_days: Time in days since first measurement
        timestamps: List of datetime objects
        format: 'mat', 'csv', or 'both'
    """
    import scipy.io as sio
    
    # Prepare data
    data_dict = {
        'range': range_total,
        'timeInDays': time_days,
    }
    
    if format in ('mat', 'both'):
        sio.savemat(f"{filename}.mat", data_dict)
        print(f"Saved: {filename}.mat")
    
    if format in ('csv', 'both'):
        import csv
        with open(f"{filename}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'time_days', 'range_m'])
            for i, ts in enumerate(timestamps):
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if ts else ''
                writer.writerow([ts_str, time_days[i], range_total[i]])
        print(f"Saved: {filename}.csv")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        Rcoarse, spec_avg, data = process_apres_file(filename)
        print(f"Loaded: {filename}")
        print(f"Time stamp: {data.time_stamp}")
        print(f"Subbursts: {data.n_subbursts}")
        print(f"Samples: {data.n_adc_samples}")
        print(f"Range bins: {len(Rcoarse)}")
        print(f"Max range: {Rcoarse[-1]:.1f} m")
    else:
        print("Usage: python apres_python.py <filename.DAT>")
