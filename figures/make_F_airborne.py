"""
make_F_airborne — airborne echogram over the Mercer Ice Stream (Fig. 1),
generated from the CReSIS L1B product so it matches the paper style.

Source: data/airborne/Data_20171204_06_010.mat (CReSIS 2017 Antarctica
survey, the frame reproduced in the original Fig. 1).  Fields used:
  Data        linear power, (n_range, n_trace)  [h5py reads transposed]
  Time        fast-time two-way travel time (s), (n_range,)
  Surface     per-trace surface two-way time (s)
  Latitude/Longitude   per-trace geolocation

Processing
----------
  • along-track distance from the geodesic between trace fixes;
  • depth below the local ice surface via per-trace surface-referencing
    with the ice permittivity e_r = 3.15 (v_ice = c / sqrt(e_r)), so the
    surface sits at 0 m everywhere (elevation-flattened);
  • amplitude in dB (10 log10 power), RdBu_r colormap (paper amplitude
    convention), depth axis matching the ApRES echograms.

Output:
  figs/Echogram_Mercer.pdf / .png
"""

from __future__ import annotations
import sys
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_style as ps
ps.apply_style()

MAT = ROOT / 'data' / 'airborne' / 'Data_20171204_06_010.mat'
OUT_DIR = ROOT / 'figs'

C = 299_792_458.0          # m/s
EPS_R = 3.15               # ice permittivity (matches original axis label)
V_ICE = C / np.sqrt(EPS_R)
DEPTH_MAX = 1800.0         # m below surface to display
DZ = 1.0                   # depth-grid spacing (m)
EFZ_TOP, BED = 600.0, 1094.0   # paper zone boundaries (ApRES site)


def haversine_km(lat, lon):
    """Cumulative great-circle distance along the track (km)."""
    R = 6371.0
    la = np.radians(lat); lo = np.radians(lon)
    dla = np.diff(la); dlo = np.diff(lo)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    seg = 2 * R * np.arcsin(np.sqrt(a))
    return np.concatenate([[0.0], np.cumsum(seg)])


print(f'Loading {MAT.relative_to(ROOT)} ...')
with h5py.File(str(MAT), 'r') as f:
    Data = np.array(f['Data']).T          # (n_range, n_trace)
    t = np.array(f['Time']).ravel()       # (n_range,) two-way time
    surf = np.array(f['Surface']).ravel() # (n_trace,)
    lat = np.array(f['Latitude']).ravel()
    lon = np.array(f['Longitude']).ravel()
n_range, n_trace = Data.shape
print(f'  {n_trace} traces x {n_range} range samples')

# along-track distance
dist = haversine_km(lat, lon)
print(f'  track length {dist[-1]:.1f} km')

# surface-reference each trace onto a common depth grid (ice depth)
depth_grid = np.arange(0.0, DEPTH_MAX + DZ, DZ)
echo = np.empty((len(depth_grid), n_trace), dtype=np.float32)
for j in range(n_trace):
    d_col = (t - surf[j]) * V_ICE / 2.0           # depth of each sample (m)
    echo[:, j] = np.interp(depth_grid, d_col, Data[:, j],
                           left=np.nan, right=np.nan)

echo_db = 10.0 * np.log10(echo + 1e-30)

# ── drop the leading artifact section ─────────────────────────────────
# Two acquisition-dropout blocks sit in the first ~13 km of the frame;
# rather than inpaint them we simply show the clean section beyond, and
# re-zero the along-track distance.
CROP_START_KM = 13.5
keep = dist >= CROP_START_KM
echo_db = echo_db[:, keep]
dist = dist[keep] - dist[keep][0]
print(f'  cropped to the clean section: {dist[-1]:.1f} km '
      f'({echo_db.shape[1]} traces)')

vmin = float(np.nanpercentile(echo_db, 45))
vmax = float(np.nanpercentile(echo_db, 99.7))
print(f'  dB display range [{vmin:.0f}, {vmax:.0f}]')

# ── figure ────────────────────────────────────────────────────────────
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(ps.FIG_W_SINGLE, 3.4))
fig.subplots_adjust(left=0.10, right=0.93, top=0.93, bottom=0.13)
im = ax.imshow(echo_db, aspect='auto', cmap=ps.CMAP_AMP, vmin=vmin, vmax=vmax,
               extent=[dist[0], dist[-1], depth_grid[-1], depth_grid[0]],
               interpolation='nearest')
ax.set_ylim(DEPTH_MAX, 0)
ax.set_xlabel('Along-track distance (km)', labelpad=2)
ax.set_ylabel('Depth (m)', labelpad=2)
ps.panel_colorbar(ax, im, label='Power (dB)', label_top=True)

# ── EFZ extent marker on the left (plain white line) ─────────────────
z0, z1 = 700.0, 1100.0
xl = dist[0] + 0.05 * (dist[-1] - dist[0])     # left side
cap = 0.012 * (dist[-1] - dist[0])             # end-cap half-width
ax.plot([xl, xl], [z0, z1], color='white', lw=1.6, zorder=5)
ax.plot([xl - cap, xl + cap], [z0, z0], color='white', lw=1.6, zorder=5)
ax.plot([xl - cap, xl + cap], [z1, z1], color='white', lw=1.6, zorder=5)
ax.text(xl + 0.035 * (dist[-1] - dist[0]), 0.5 * (z0 + z1), 'Echo-Free Zone',
        ha='left', va='center', fontsize=8, color='white', zorder=6)

for ext in ('pdf', 'png'):
    out = OUT_DIR / f'Echogram_Mercer.{ext}'
    fig.savefig(out, dpi=300 if ext == 'png' else None)
    print(f'Saved {out.relative_to(ROOT)}')
