"""Bounded-workspace merging for uncompressed, memory-mappable TIFFs.

Only the float32 result (and RAW validity mask) is full-size in RAM. Source
mappings are closed after each patch so touched input pages do not accumulate.
No halo is needed: exposure merging is independent at every pixel.
"""
from pathlib import Path

import numpy as np
import tifffile


def validate_stack(files, tile_size):
    files = [Path(p) for p in files]
    if not files:
        raise ValueError("Need at least one TIFF")
    if isinstance(tile_size, bool) or not isinstance(tile_size, (int, np.integer)) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    shape = dtype = None
    for path in files:
        array = open_tiff(path)
        try:
            if shape is None:
                shape, dtype = array.shape, array.dtype
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"TIFF shapes and dtypes must match: {path}")
            if array.dtype != np.uint16 or array.ndim not in (2, 3):
                raise ValueError(f"Expected uint16 grayscale/Bayer or RGB TIFF: {path}")
            if array.ndim == 3 and array.shape[-1] != 3:
                raise ValueError(f"Expected interleaved RGB TIFF: {path}")
        finally:
            array._mmap.close()
    return files, shape


def open_tiff(path):
    try:
        return tifffile.memmap(path, mode="r")
    except ValueError as exc:
        raise ValueError(
            f"Cannot read patches directly from {path}. Use an uncompressed, "
            "contiguous TIFF (such as dcraw -4 -T output); this path deliberately "
            "does not decode an entire compressed image into RAM."
        ) from exc


def read_patch(path, ys, xs):
    array = open_tiff(path)
    try:
        return array[ys, xs].copy()
    finally:
        array._mmap.close()


def patches(shape, tile_size):
    for y in range(0, shape[0], tile_size):
        for x in range(0, shape[1], tile_size):
            yield slice(y, min(y + tile_size, shape[0])), slice(x, min(x + tile_size, shape[1]))


def validate_times(times, count):
    times = np.asarray(times, dtype=float)
    if times.shape != (count,) or not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError("Need one finite positive exposure time per TIFF")
    return times


def merge_tiff_stack(files, merge_function, *, tile_size=256, exposure_times=None,
                     config="linear", weight="uniform", gain=None,
                     additive_variance=None, progress=True):
    """Normalize uint16 patches and use the existing RGB merge unchanged.

    Returns a full-size float32 array; workspace scales with tile_size**2,
    not frame count or image size. Does not apply a JPEG response curve.
    """
    files, shape = validate_stack(files, tile_size)
    times = validate_times(exposure_times if exposure_times is not None
                           else [2**i / 2048.0 for i in range(len(files))], len(files))
    result = np.empty(shape, dtype=np.float32)
    total = ((shape[0] + tile_size - 1) // tile_size) * ((shape[1] + tile_size - 1) // tile_size)
    for index, (ys, xs) in enumerate(patches(shape, tile_size), 1):
        def frames():
            for path in files:
                frame = read_patch(path, ys, xs).astype(np.float32)
                frame /= 65535.0
                yield frame
        result[ys, xs] = merge_function(
            frames(), config=config, weight=weight, exposure_times=times,
            gain=gain, additive_variance=additive_variance, verbose=False)
        if progress and (index == 1 or index % 25 == 0 or index == total):
            print(f"Merged patch {index}/{total}", flush=True)
    return result


def merge_raw_tiff_stack(files, times, gains, additive, *, black_level=150.,
                         saturation=4000., tile_size=256, progress=True):
    """Patchwise RG/GB merge, preserving signed samples and float64 sums.

    Files must already be in increasing exposure order, for the shortest-frame
    fallback. Bayer phase uses absolute coordinates, even for odd tile sizes.
    """
    files, shape = validate_stack(files, tile_size)
    if len(shape) != 2:
        raise ValueError("Expected 2D Bayer TIFFs")
    times = validate_times(times, len(files))
    if np.any(np.diff(times) < 0):
        raise ValueError("RAW frames must be sorted by exposure time")
    gains, additive = np.asarray(gains, dtype=float), np.asarray(additive, dtype=float)
    if gains.shape != (3,) or additive.shape != (3,) or not np.isfinite([gains, additive]).all() or np.any(gains <= 0) or np.any(additive <= 0):
        raise ValueError("Need three finite positive RGB gain and variance coefficients")
    if not np.isfinite([black_level, saturation]).all() or saturation <= black_level:
        raise ValueError("Saturation must exceed the finite black level")
    hdr = np.empty(shape, dtype=np.float32)
    usable = np.empty(shape, dtype=bool)
    total = ((shape[0] + tile_size - 1) // tile_size) * ((shape[1] + tile_size - 1) // tile_size)
    for index, (ys, xs) in enumerate(patches(shape, tile_size), 1):
        size = (ys.stop - ys.start, xs.stop - xs.start)
        numerator = np.zeros(size, dtype=np.float64)
        denominator = np.zeros(size, dtype=np.float64)
        for i, (path, t) in enumerate(zip(files, times)):
            raw = read_patch(path, ys, xs)
            if i == 0:
                fallback = (raw.astype(np.float32) - black_level) / t
            for dy, dx, c in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)]:
                phase = (slice((dy - ys.start) % 2, None, 2),
                         slice((dx - xs.start) % 2, None, 2))
                sample = raw[phase]
                signal = sample.astype(np.float64) - black_level
                valid = (sample > 0) & (sample < saturation)
                variance = gains[c] * np.maximum(signal, 0.) + additive[c]
                weights = np.where(valid, t**2 / variance, 0.)
                numerator[phase] += weights * signal / t
                denominator[phase] += weights
        valid = denominator > 0
        np.divide(numerator, denominator, out=fallback, where=valid, casting="unsafe")
        hdr[ys, xs], usable[ys, xs] = fallback, valid
        if progress and (index == 1 or index % 25 == 0 or index == total):
            print(f"Merged RAW patch {index}/{total}", flush=True)
    return hdr, usable
