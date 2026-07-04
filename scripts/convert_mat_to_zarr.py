#!/usr/bin/env python3
"""
Convert ApRES .mat files to chunked .zarr directories.
Zarr eliminates the need to load 2GB arrays into RAM all at once and allows rapid 
lazy loading for backend web apps.

Usage:
  python convert_mat_to_zarr.py --input data/ImageP2_python.mat
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import zarr


def convert_mat_to_zarr(mat_path: str, zarr_path: str = None) -> None:
    path = Path(mat_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    if zarr_path is None:
        zarr_path = path.with_suffix('.zarr')
        
    print(f"Loading {path.name} into memory... (this might take a few GBs of RAM)")
    try:
        mat_data = loadmat(str(path))
    except Exception as e:
        print(f"Failed to load .mat file: {e}")
        return

    # Extract original arrays
    raw_image = np.array(mat_data.get('RawImage', []), dtype=np.float32)
    raw_complex = None
    if 'RawImageComplex' in mat_data:
        raw_complex = np.array(mat_data['RawImageComplex'], dtype=np.complex64)
    rcoarse = mat_data.get('Rcoarse', np.array([])).flatten().astype(np.float32)
    time_days = mat_data.get('TimeInDays', np.array([])).flatten().astype(np.float32)
    lambdac = float(np.array(mat_data.get('lambdac', 0)).flatten()[0])

    print(f"Creating Zarr store at: {zarr_path}")
    # Clears any existing store at that location
    root = zarr.open_group(str(zarr_path), mode='w')

    # 1D arrays don't need chunking, they are small.
    root.create_array('Rcoarse', data=rcoarse)
    root.create_array('time_days', data=time_days)
    root.attrs['lambdac'] = lambdac

    # 2D arrays need careful chunking to allow rapid spatial subsets in Dash
    # If the array is e.g. (Depth=20000, Time=2000), a chunk size of (500, 100)
    # means Dash only reads ~500KB blobs at a time
    chunk_depth = 500
    chunk_time = 100

    if raw_image.size > 0:
        print(f"Writing RawImage {raw_image.shape} to Zarr...")
        # Automatically bounds chunk size if array is smaller than the chunk
        cd = min(chunk_depth, raw_image.shape[0])
        ct = min(chunk_time, raw_image.shape[1])
        root.create_array('range_img', data=raw_image, chunks=(cd, ct))

    if raw_complex is not None and raw_complex.size > 0:
        print(f"Writing RawImageComplex {raw_complex.shape} to Zarr...")
        cd = min(chunk_depth, raw_complex.shape[0])
        ct = min(chunk_time, raw_complex.shape[1])
        root.create_array('raw_complex', data=raw_complex, chunks=(cd, ct))

    print("Conversion complete!")
    print(f"The web application can now open {zarr_path} lazily using zarr.open(mode='r')")


def main():
    parser = argparse.ArgumentParser(description="Convert ApRES .mat files to .zarr")
    parser.add_argument("--input", type=str, required=True, help="Input .mat file")
    parser.add_argument("--output", type=str, default=None, help="Output .zarr name (optional)")
    args = parser.parse_args()
    
    convert_mat_to_zarr(args.input, args.output)


if __name__ == "__main__":
    main()
