#!/usr/bin/env python3
"""
Convert raw ApRES .DAT files into ImageP2_python.mat with complex data preserved.
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.io import savemat
import zarr

from apres.io import process_timeseries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process ApRES .DAT files and save ImageP2_python.mat",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="Folder containing .DAT files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ImageP2_python.mat",
        help="Output .mat file path",
    )
    parser.add_argument("--er", type=float, default=3.18, help="Relative permittivity")
    parser.add_argument("--max-range", type=float, default=2000.0, help="Max range (m)")
    parser.add_argument("--pad-factor", type=int, default=8, help="FFT padding factor")
    parser.add_argument("--step", type=int, default=1, help="Process every Nth file")
    parser.add_argument(
        "--no-complex",
        action="store_true",
        help="Do not store complex RawImage (magnitude only)",
    )
    parser.add_argument(
        "--subband",
        type=str,
        default=None,
        choices=["low", "high"],
        help="Subband to process (low, high, or omitted for full band)",
    )
    parser.add_argument(
        "--zarr",
        action="store_true",
        help="Also export outputs as chunked Zarr arrays for efficient web loading."
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_complex = not args.no_complex

    result = process_timeseries(
        args.data_folder,
        er=args.er,
        max_range=args.max_range,
        pad_factor=args.pad_factor,
        step=args.step,
        keep_complex=keep_complex,
        subband=args.subband,
    )

    if keep_complex:
        range_img, rfine_avg, rcoarse, time_days, timestamps, range_img_complex = result
    else:
        range_img, rfine_avg, rcoarse, time_days, timestamps = result
        range_img_complex = None

    lambdac = 1.0 / np.sqrt(args.er)
    
    mat_dict = {
        "RawImage": range_img,
        "RfineBarTime": rfine_avg,
        "Rcoarse": rcoarse,
        "TimeInDays": time_days,
        "lambdac": lambdac,
    }
    if range_img_complex is not None:
        mat_dict["RawImageComplex"] = range_img_complex

    savemat(str(output_path), mat_dict)
    print(f"Saved: {output_path}")

    if args.zarr:
        zarr_path = output_path.with_suffix(".zarr")
        print(f"Creating Zarr store at: {zarr_path}")
        root = zarr.open_group(str(zarr_path), mode='w')

        root.create_array('Rcoarse', data=rcoarse.astype(np.float32))
        root.create_array('time_days', data=time_days.astype(np.float32))
        root.attrs['lambdac'] = lambdac

        chunk_depth, chunk_time = 500, 100
        cd = min(chunk_depth, range_img.shape[0])
        ct = min(chunk_time, range_img.shape[1])
        root.create_array('range_img', data=range_img.astype(np.float32),
                           chunks=(cd, ct))

        if range_img_complex is not None:
            cd = min(chunk_depth, range_img_complex.shape[0])
            ct = min(chunk_time, range_img_complex.shape[1])
            root.create_array('raw_complex', data=range_img_complex.astype(np.complex64),
                               chunks=(cd, ct))

        print(f"Saved Zarr: {zarr_path}")


if __name__ == "__main__":
    main()
