"""
2-D forward model for ApRES phase-sensitive radar.

Co-moving frame
---------------
The radar sits at the origin (x=0, z=0). Reflectors live in the half-
space z>0 (depth positive downward) at arbitrary lateral offsets x.
Over time, each reflector drifts at its own velocity (vx, vz) relative
to the radar. Because the radar co-moves with the ice surface, a
reflector embedded in ice with perfect plug-flow has vx=0 in this
frame; only ice-internal shear (or geographically-fixed scatterers)
produces non-zero vx.

The simulator returns a range-compressed complex echogram S(z, t) with
the same shape as the real zarr data, so downstream analysis (carrier
detrend, mean subtract, coherence, phase slope) applies unchanged.

Physical ingredients
--------------------
- Slant range from radar to scatterer: R = sqrt(x^2 + z^2).
- Range-compressed complex signal at depth bin z for a reflector at R:
      S(z) = A * W(z - R) * exp( +j * k * (R - z) ),
  where k = 4 pi / lambda_c. This matches the output of
  `apres.io.fmcw_range` after its built-in phase correction:
  the +j*k*R term makes the phase at a fixed bin track the
  reflector's range (the tracking signal), and the -j*k*z term is
  the spatial carrier that detrending cancels (see `carrier_detrend`).
- Amplitude envelope around the slant range modelled as a Gaussian
  approximation of the Blackman-windowed 200 MHz FMCW PSF.
- Layers are discretised into sub-wavelength segments; the specular /
  diffraction behaviour emerges automatically from the coherent sum
  (Huygens principle).

Usage
-----
>>> scene = [
...     Layer(np.array([[-250, 800], [250, 800]]), vz=1.0),    # flat layer
...     Point(x0=0, z0=800, vx=-6, vz=1.0, A=2.0),            # shearing point
... ]
>>> S = simulate(scene, time_days=np.linspace(0, 312, 300),
...              depths=np.linspace(795, 805, 400))
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, replace
from typing import Callable, List, Sequence, Union, Tuple

# ── Physical constants ────────────────────────────────────────────────
C_ICE    = 1.68e8                  # wave speed in ice (m/s)
F_C      = 3.0e8                   # radar centre frequency (Hz)
B_HZ     = 2.0e8                   # chirp bandwidth (Hz)
LAMBDA_C = C_ICE / F_C             # ~0.56 m
# Blackman-windowed 200 MHz FMCW PSF has 3dB width ~0.8 m; the
# 1/e width of a matched Gaussian is ~0.4 m.
DEFAULT_PSF_SIGMA = 0.4
# Antenna beam pattern: amplitude weight ~ exp(-θ²/(2 σ_θ²)), where
# θ = atan2(x, z) is the off-nadir angle.  For ApRES the 3-dB beamwidth
# is ~30°, giving σ_θ ≈ 0.22 rad.  Setting σ_θ = None (or ∞) disables
# the beam pattern (back-compat with the original simulator).
DEFAULT_SIGMA_THETA = 0.22


# ── Reflector types ───────────────────────────────────────────────────
@dataclass
class Point:
    """Single point scatterer.

    Position at time t (years): (x0 + vx*t, z0 + vz*t).
    `A` is the complex reflectivity — magnitude scales the echo,
    phase sets an arbitrary initial offset.
    """
    x0: float
    z0: float
    vx: float = 0.0          # lateral velocity relative to radar (m/yr)
    vz: float = 0.0          # vertical velocity (m/yr, positive = down)
    A:  complex = 1.0 + 0.0j

    def position(self, t_yr: float) -> Tuple[float, float]:
        return self.x0 + self.vx * t_yr, self.z0 + self.vz * t_yr


@dataclass
class Layer:
    """Piecewise-linear layer (polyline).

    `vertices` is an (N, 2) array of (x, z) control points. The layer
    drifts as a rigid body at (vx, vz). At simulation time it is
    discretised into sub-segments of length ≤ `sample_spacing`; each
    sub-segment acts as a weak point scatterer with amplitude
    A_per_m * segment_length.

    Specular reflection (flat or dipping) and diffraction edges both
    emerge naturally from the coherent sum of these Huygens wavelets —
    no separate Fresnel-zone machinery needed.
    """
    vertices: np.ndarray
    vx: float = 0.0
    vz: float = 0.0
    A_per_m: complex = 1.0 + 0.0j
    sample_spacing: float = 0.1      # (m) along the arc

    def _sample_rest(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Discretise the polyline at rest (t=0). Returns x, z, dA per sub-point."""
        v = np.asarray(self.vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError("Layer.vertices must have shape (N, 2)")

        xs_list, zs_list, dA_list = [], [], []
        for i in range(len(v) - 1):
            x0, z0 = v[i]
            x1, z1 = v[i + 1]
            seg_len = float(np.hypot(x1 - x0, z1 - z0))
            if seg_len == 0.0:
                continue
            n = max(1, int(np.ceil(seg_len / self.sample_spacing)))
            # Midpoint quadrature — each sub-sample represents one cell
            ts = (np.arange(n) + 0.5) / n
            xs_list.append(x0 + (x1 - x0) * ts)
            zs_list.append(z0 + (z1 - z0) * ts)
            dA_list.append(np.full(n, self.A_per_m * seg_len / n,
                                   dtype=np.complex128))

        if not xs_list:
            empty = np.zeros(0)
            return empty, empty, empty.astype(np.complex128)
        return (np.concatenate(xs_list),
                np.concatenate(zs_list),
                np.concatenate(dA_list))

    def sample(self, t_yr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, zs, dA = self._sample_rest()
        return xs + self.vx * t_yr, zs + self.vz * t_yr, dA


@dataclass
class BackgroundSpeckle:
    """Distributed random scatterers filling a depth × lateral window.

    Real ApRES echograms show a spatial phase carrier at ~−22 rad/m
    (i.e. ~one wrap per λ_c/2 in depth) because the ice is a continuous
    weakly-scattering medium. With an *isolated* reflector in otherwise
    empty space, the simulator would produce a flat phase profile that
    does not match the real-data look. A realistic background is
    created by scattering many weak points across the depth window.

    The scatterer realisation is drawn once at t=0 and then drifts
    rigidly at (vx, vz); it is reproducible via `seed`.
    """
    z_min: float
    z_max: float
    x_halfwidth: float = 20.0        # lateral extent (m)
    n_scatterers: int = 800
    amplitude_rms: float = 0.003     # per-scatterer complex std
    vx: float = 0.0
    vz: float = 0.0
    seed: int = 0

    def to_points(self) -> List["Point"]:
        """Expand into individual Point scatterers.

        Amplitudes are real-valued (Gaussian): real ice reflectivity is
        real (impedance mismatch), and it is this realness that causes
        the −(4π/λ_c) rad/m spatial phase carrier in the echogram.
        Using complex-random amplitudes would destroy that carrier and
        make the phase image look like white noise.
        """
        rng = np.random.default_rng(self.seed)
        xs = rng.uniform(-self.x_halfwidth, self.x_halfwidth, self.n_scatterers)
        zs = rng.uniform(self.z_min, self.z_max, self.n_scatterers)
        amps_real = self.amplitude_rms * rng.standard_normal(self.n_scatterers)
        return [Point(x0=float(xs[i]), z0=float(zs[i]),
                      vx=self.vx, vz=self.vz, A=complex(amps_real[i]))
                for i in range(self.n_scatterers)]


Reflector = Union[Point, Layer, BackgroundSpeckle]


def _expand_scene(scene: List[Reflector]) -> List[Union[Point, Layer]]:
    """Flatten aggregate reflectors (BackgroundSpeckle) into primitives."""
    out: List[Union[Point, Layer]] = []
    for r in scene:
        if isinstance(r, BackgroundSpeckle):
            out.extend(r.to_points())
        else:
            out.append(r)
    return out


# ── Forward model ─────────────────────────────────────────────────────
def _contribute_points(S: np.ndarray, depths: np.ndarray,
                       xs: np.ndarray, zs: np.ndarray, dA: np.ndarray,
                       time_yr: np.ndarray,
                       lambdac: float, psf_sigma: float) -> None:
    """Add the contribution of N point-like scatterers (each with its
    own trajectory) to S. If `xs/zs/dA` are 1-D of length N and
    `time_yr` has length T, all arrays must broadcast consistently.

    Used internally: for a Point reflector, N=1 and the trajectory is
    evaluated at each of T times; for a Layer, N=n_subsegments and the
    same geometry is reused across times with a rigid shift.
    """
    # Here we accept already-evaluated positions (xs, zs) of shape
    # either (N,) or (N, T). dA shape (N,) or (N, T) likewise.
    raise NotImplementedError  # kept for reference — actual paths below


def simulate(scene: List[Reflector],
             time_days: np.ndarray,
             depths: np.ndarray,
             lambdac: float = LAMBDA_C,
             psf_sigma: float = DEFAULT_PSF_SIGMA,
             sigma_theta: float | None = DEFAULT_SIGMA_THETA,
             noise_sigma: float = 0.0,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """Run the 2-D forward model.

    Parameters
    ----------
    scene : list of Point / Layer
    time_days : (n_t,) array of times in days since t=0
    depths    : (n_z,) array of depth bin centres (m)
    lambdac   : carrier wavelength in ice (default 0.56 m)
    psf_sigma : 1/e width of the Gaussian PSF in range (default 0.4 m)
    sigma_theta : antenna beam angular std σ_θ in rad. Amplitude weight is
        exp(-θ²/(2σ_θ²)) with θ = atan2(x, z).  Set to None to disable
        (recovers the original isotropic model).  This is what produces the
        beam-sweep decorrelation: a lateral drift of vx changes θ over
        time, modulating the amplitude.  Default 0.22 rad (~30° 3-dB).
    noise_sigma : if > 0, add circular complex Gaussian noise of this std
    rng       : numpy Generator for reproducibility (default: default_rng())

    Returns
    -------
    S : complex ndarray, shape (n_z, n_t)
    """
    depths  = np.asarray(depths, dtype=float)
    time_yr = np.asarray(time_days, dtype=float) / 365.25
    n_t, n_z = len(time_yr), len(depths)
    S = np.zeros((n_z, n_t), dtype=np.complex128)

    k = 4.0 * np.pi / lambdac   # two-way phase per metre of slant range

    # Spatial-carrier factor — same for every reflector, computed once.
    # Phase = -k*z at every depth bin, mirroring the -22 rad/m carrier
    # in real fmcw_range output.
    carrier_z = np.exp(-1j * k * depths)[:, None]             # (n_z, 1)

    use_beam = sigma_theta is not None and np.isfinite(sigma_theta) \
               and sigma_theta > 0
    two_st2  = 2.0 * (sigma_theta ** 2) if use_beam else None

    for reflector in _expand_scene(scene):
        if isinstance(reflector, Point):
            x_t = reflector.x0 + reflector.vx * time_yr       # (n_t,)
            z_t = reflector.z0 + reflector.vz * time_yr
            R_t = np.sqrt(x_t * x_t + z_t * z_t)
            # Envelope across all (depth, time) bins
            dz  = depths[:, None] - R_t[None, :]              # (n_z, n_t)
            env = np.exp(-(dz / psf_sigma) ** 2)
            # Antenna beam weight: exp(-θ²/(2σ_θ²)), θ = atan2(x, z).
            # For typical scenes z >> |x|, so θ ≈ x/z is accurate, but use
            # atan2 to stay correct for steep dipping reflectors too.
            if use_beam:
                theta = np.arctan2(x_t, z_t)                  # (n_t,)
                beam  = np.exp(-(theta ** 2) / two_st2)       # (n_t,)
                env   = env * beam[None, :]
            # Phase = exp(+j k R) in time × exp(-j k z) in depth
            phs_t = np.exp(+1j * k * R_t)[None, :]            # (1, n_t)
            S    += reflector.A * env * carrier_z * phs_t

        elif isinstance(reflector, Layer):
            # Sample the polyline ONCE at rest, then shift rigidly each t
            xs0, zs0, dA = reflector._sample_rest()           # (n_sub,)
            if xs0.size == 0:
                continue
            # Broadcast: (n_sub, n_t)
            xs = xs0[:, None] + reflector.vx * time_yr[None, :]
            zs = zs0[:, None] + reflector.vz * time_yr[None, :]
            R  = np.sqrt(xs * xs + zs * zs)                   # (n_sub, n_t)
            if use_beam:
                theta_all = np.arctan2(xs, zs)                # (n_sub, n_t)
                beam_all  = np.exp(-(theta_all ** 2) / two_st2)
            for i in range(xs0.size):
                Ri    = R[i]                                  # (n_t,)
                dz    = depths[:, None] - Ri[None, :]
                env   = np.exp(-(dz / psf_sigma) ** 2)
                if use_beam:
                    env = env * beam_all[i][None, :]
                phs_t = np.exp(+1j * k * Ri)[None, :]
                S    += dA[i] * env * carrier_z * phs_t

        else:
            raise TypeError(f"Unknown reflector type: {type(reflector).__name__}")

    if noise_sigma > 0:
        if rng is None:
            rng = np.random.default_rng()
        S += noise_sigma * (rng.standard_normal(S.shape)
                            + 1j * rng.standard_normal(S.shape))
    return S


# ── Post-processing helpers (match real-data pipeline) ────────────────
def carrier_detrend(S: np.ndarray, depths: np.ndarray,
                    grad: float | None = None,
                    lambdac: float = LAMBDA_C) -> np.ndarray:
    """Subtract the spatial carrier phase.

    A reflector at depth R contributes phase -4π R / λ_c, which grows
    linearly with depth at rate -4π/λ_c ≈ -22.4 rad/m. Removing this
    gradient exposes the time-varying structure. Pass `grad` explicitly
    to match the empirical value estimated from real data (e.g.
    -22.2 rad/m at the SiegVent site).
    """
    if grad is None:
        grad = -4.0 * np.pi / lambdac
    return S * np.exp(-1j * grad * depths[:, None])


def mean_subtract(S: np.ndarray) -> np.ndarray:
    """Remove the complex temporal mean at each depth bin."""
    return S - S.mean(axis=1, keepdims=True)


# ── Depth-dependent velocity profile ─────────────────────────────────
@dataclass
class VelocityProfile:
    """Depth-dependent velocity v(z) → (vx, vz), both in m/yr.

    Used to override per-reflector velocities so a whole scene moves
    according to a shared depth profile (e.g. linearly-increasing vz
    with depth, as seen in the EFZ). See `apply_velocity_profile`.
    """
    vx_fn: Callable[[float], float]
    vz_fn: Callable[[float], float]

    def __call__(self, z: float) -> Tuple[float, float]:
        return float(self.vx_fn(z)), float(self.vz_fn(z))

    @classmethod
    def polynomial(cls,
                   vx_coeffs: Sequence[float],
                   vz_coeffs: Sequence[float],
                   z_ref: float = 0.0) -> "VelocityProfile":
        """Polynomial in (z − z_ref). Coeffs ordered [c0, c1, c2, …]
        so that v(z) = c0 + c1·(z−z_ref) + c2·(z−z_ref)² + …
        """
        vx_c = np.asarray(vx_coeffs, dtype=float)
        vz_c = np.asarray(vz_coeffs, dtype=float)

        def _poly(c, z):
            dz = z - z_ref
            return float(np.polyval(c[::-1], dz))

        return cls(vx_fn=lambda z: _poly(vx_c, z),
                   vz_fn=lambda z: _poly(vz_c, z))


def _reflector_depth(r: Reflector) -> float:
    """Representative depth for profile lookup."""
    if isinstance(r, Point):
        return float(r.z0)
    if isinstance(r, Layer):
        v = np.asarray(r.vertices, dtype=float)
        return float(v[:, 1].mean())
    if isinstance(r, BackgroundSpeckle):
        return 0.5 * (float(r.z_min) + float(r.z_max))
    raise TypeError(f"Unknown reflector type: {type(r).__name__}")


def apply_velocity_profile(scene: List[Reflector],
                           profile: VelocityProfile) -> List[Reflector]:
    """Return a new scene with each reflector's (vx, vz) overridden by
    `profile` evaluated at its representative depth.

    The velocity is set once at t=0 and then treated as constant for
    that reflector's trajectory, which is accurate when drift over the
    simulated time span (meters) is small compared to the profile's
    length scale (hundreds of metres).
    """
    out: List[Reflector] = []
    for r in scene:
        vx, vz = profile(_reflector_depth(r))
        out.append(replace(r, vx=vx, vz=vz))
    return out
