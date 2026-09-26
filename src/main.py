import numpy as np
from skimage.io import imread, imsave
from cp_hw2 import readHDR, read_colorchecker_gm, writeHDR, lRGB2XYZ, xyY_to_XYZ, XYZ2lRGB
import matplotlib.pyplot as plt


def read_doorStacks(path, ext="jpg"):
    imgs = []
    for i in range(16):
        img_path = f"{path}/exposure{i+1}.{ext}"
        data = imread(img_path)
        if ext == "tiff":
            data = data / (2**16 - 1)
        imgs.append(data)

        # print(data.mean(), data.max(), data.min())

    return np.stack(imgs)

def weight_function(imgs, config="uniform", exposure_time=None, gain=None, additive_variance=None):
    Z_min = 0.05
    Z_max = 0.95

    mask = (imgs > Z_min) & (imgs < Z_max)
    mask = mask.astype(np.float32)

    if config == "uniform":
        pass
    elif config == "tent":
        mask = np.minimum(imgs, 1.0-imgs) * mask
    elif config == "gaussian":
        mask = np.exp(-16 * np.square(imgs - 0.5)) * mask
    elif config == "photon":
        if exposure_time is not None:
            return mask * exposure_time
        if mask.shape[0] != 16:
            return np.ones_like(mask)
        for i in range(16):
            mask[i] *= 2**i / 2048.0
    elif config == "optimal":
        if exposure_time is None or gain is None or additive_variance is None:
            raise ValueError("Optimal weights need exposure_time, gain, and additive_variance")
        gain = np.asarray(gain, dtype=np.float32)
        additive_variance = np.asarray(additive_variance, dtype=np.float32)
        if not np.all(np.isfinite(gain)) or not np.all(np.isfinite(additive_variance)) or np.any(gain <= 0) or np.any(additive_variance <= 0):
            raise ValueError("Noise coefficients must be finite and positive")
        variance = gain * np.maximum(imgs, 0) + additive_variance
        mask = mask * exposure_time**2 / variance
    else:
        raise NotImplementedError

    return mask

def solve_g(imgs, lambda_0, weight, exposure_times=None):
    times = np.asarray(exposure_times if exposure_times is not None
                       else [2**i / 2048.0 for i in range(16)], dtype=float)
    if times.shape != (imgs.shape[0],) or not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError("Need one positive exposure time per calibration frame")
    w = weight_function(imgs / 255.0, weight,
                        exposure_time=times[:, None] if weight == "photon" else None)

    A = np.zeros((imgs.shape[0]*imgs.shape[1], 256 + imgs.shape[1]))
    b = np.ones_like(imgs).astype(np.float32)
    for i in range(imgs.shape[0]):
        b[i] *= np.log(times[i])
    b = b.reshape(-1)

    A[np.arange(A.shape[0]), imgs.reshape(-1)] = 1
    A[np.arange(A.shape[0]), 256 + np.arange(A.shape[0]) % imgs.shape[1]] = -1

    # print(b)
    A = w.reshape(-1, 1) * A
    b = w.reshape(-1) * b

    A2 = np.zeros((1, A.shape[1]))
    b2 = np.zeros(1)
    # Fix the arbitrary log-exposure offset: exp(g[242]) = 1.
    # Linearized intensities are relative exposures, not values limited to [0, 1].
    A2[:, 242] = 1

    A = np.concatenate([A, A2], axis=0)
    b = np.concatenate([b, b2], axis=0)

    A3 = np.zeros((230, A.shape[1]))
    b3 = np.zeros(230)

    A3[np.arange(230), np.arange(12, 242)] = 1
    A3[np.arange(230), np.arange(13, 243)] = -2
    A3[np.arange(230), np.arange(14, 244)] = 1

    A3 *= np.sqrt(lambda_0)
    w2 = weight_function(np.arange(13, 243) / 255.0, weight)
    A3 = A3 * w2.reshape(-1, 1)

    A = np.concatenate([A, A3], axis=0)
    b = np.concatenate([b, b3], axis=0)

    result = np.linalg.lstsq(A, b, rcond=None)
    result[0][:13] = result[0][13]
    result[0][243:256] = result[0][242]

    return result[0][:256]

def plot_g(result):
    plt.plot(np.arange(13, 243), result, color='blue', linestyle='-', marker='o', linewidth=2, label='g')

    # Labels and styling
    plt.xlabel('Pixel Intensity')
    plt.ylabel('g(I)')
    plt.title('g curve')
    plt.grid(True)
    plt.legend()

    plt.show()

def merge(imgs, linear_imgs=None, config="linear", weight="uniform",
          exposure_times=None, gain=None, additive_variance=None, verbose=True):
    """Merge arrays or a frame iterator. Intensities for weighting are in [0,1].

    Optimal RGB weights are approximate; gain/variance must use normalized units.
    linear_imgs=None means the weighting images are already linear.
    """
    from itertools import zip_longest
    times = np.asarray(exposure_times if exposure_times is not None
                       else [2**i / 2048.0 for i in range(16)], dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("Exposure times must be finite positive values")
    if config not in ("linear", "log"):
        raise ValueError("Unknown merge mode")
    if weight == "optimal" and config != "linear":
        raise ValueError("Inverse radiance-variance weights require linear merging")
    pairs = ((a, a) for a in imgs) if linear_imgs is None else zip_longest(imgs, linear_imgs)
    numerator = denominator = fallback = distance = None
    count = 0
    for i, pair in enumerate(pairs):
        if i >= len(times) or pair[0] is None or pair[1] is None:
            raise ValueError("Frame and exposure counts differ")
        observed = np.asarray(pair[0], dtype=np.float32)
        signal = np.asarray(pair[1], dtype=np.float32)
        if observed.shape != signal.shape or not np.isfinite(signal).all() or not np.isfinite(observed).all():
            raise ValueError("Frame shapes must match and contain finite values")
        if numerator is None:
            numerator = np.zeros_like(signal)
            denominator = np.zeros_like(signal)
            fallback = np.zeros_like(signal)
            distance = np.full_like(signal, np.inf)
        if signal.shape != numerator.shape:
            raise ValueError("All frames must have the same shape")
        t = times[i]
        w = weight_function(observed, weight, exposure_time=t,
                            gain=gain, additive_variance=additive_variance)
        radiance = signal / t
        d = np.abs(observed - .5)
        better = d < distance
        fallback[better] = radiance[better]
        distance[better] = d[better]
        value = radiance if config == "linear" else np.log(np.maximum(signal, 1e-8)) - np.log(t)
        numerator += w * value
        denominator += w
        count += 1
        if verbose:
            print(f"Merged {count}/{len(times)}, t={t:g}s", flush=True)
    if count != len(times):
        raise ValueError("Frame and exposure counts differ")
    valid = denominator > 0
    np.divide(numerator, denominator, out=numerator, where=valid)
    if config == "log":
        np.exp(numerator, out=numerator)
    numerator[~valid] = fallback[~valid]
    if verbose:
        print(f"Fallback channel samples: {np.mean(~valid):.4%} (closest to midrange, exposure-normalized)")
    return numerator


def merge_room_rgb(tile_size=256):
    """Shared-WB linear RGB workflow; RAW-calibrated weights are approximate."""
    from pathlib import Path
    import json, subprocess
    import tifffile
    project = Path(__file__).resolve().parents[1]
    root = project / "data/room_stack"
    files = [root / "tiff_fixed_wb" / f"exposure{i}.tiff" for i in range(1,17)]
    if not all(p.exists() for p in files):
        raise ValueError("Run src/convert_room_fixed_wb.sh first")
    records = json.loads(subprocess.check_output(["exiftool", "-j", "-n", "-ISO", "-ExposureTime", "-FNumber"] +
                        [str(root / f"exposure{i}.nef") for i in range(1,17)], text=True))
    meta = {Path(r["SourceFile"]).stem:r for r in records}
    if any(r["ISO"] != 200 for r in records) or len({r["FNumber"] for r in records}) != 1:
        raise ValueError("This RGB calibration requires ISO200 and fixed aperture")
    times = [meta[p.stem]["ExposureTime"] for p in files]
    wb = np.array([1.41015625, 1., 2.20703125], dtype=np.float32)
    # Parameters originally fitted in scaling-only 16-bit DN units, without WB.
    # This channel-wise approximation omits demosaicing and color-matrix covariance.
    gain = np.array([4.836,4.12,4.885], dtype=np.float32) * wb / 65535.
    additive = np.array([255.9,156.1,253.4], dtype=np.float32) * wb**2 / 65535.**2
    from tiled_merge import merge_tiff_stack
    hdr = merge_tiff_stack(files, merge, tile_size=tile_size, weight="optimal",
                           exposure_times=times, gain=gain, additive_variance=additive)
    out = root / "optimal_merge_rgb"
    out.mkdir(exist_ok=True)
    writeHDR(str(out / "room_optimal_fixed_wb.hdr"), hdr)
    report = {"white_balance_RGB":wb.tolist(), "exposure_times":times,
              "gain_normalized_RGB":gain.tolist(), "additive_variance_normalized_RGB":additive.tolist(),
              "mask_range":[.05,.95], "units":"linear RGB / second",
              "note":"Approximate RGB weights from RAW calibration; ignores demosaicing/color-matrix covariance. Shared WB from exposure8, not neutral-patch calibrated."}
    (out / "merge_report.json").write_text(json.dumps(report,indent=2))
    # Downsample only the display preview; the HDR remains full resolution.
    preview = np.maximum(hdr[::4,::4].copy(), 0)
    preview = tonemapping(preview, config="xyY")
    save_jpg(str(out / "preview.jpg"), preview, scale=1)
    print(f"Saved HDR and preview to {out}")
    return hdr


def optimal_raw_weights(signal, exposure_time, gain, additive_variance, valid):
    """Inverse variance of signal/time, at fixed ISO; signal is signed RAW DN."""
    if exposure_time <= 0 or gain <= 0 or additive_variance <= 0:
        raise ValueError("Exposure, gain, and additive variance must be positive")
    variance = gain * np.maximum(signal, 0.0) + additive_variance
    return np.where(valid, exposure_time**2 / variance, 0.0)

def get_patchs():
    return np.array([
        [3306, 1418, 3425, 1524],
        [3299, 1256, 3420, 1368],
        [3293, 1091, 3413, 1209],
        [3295, 935, 3407, 1042],
        [3283, 779, 3400, 885],
        [3280, 614, 3395, 729], #6
        [3463, 1414, 3579, 1527],
        [3461, 1258, 3576, 1367],
        [3448, 1095, 3569, 1208],
        [3452, 936, 3563, 1047],
        [3443, 779, 3556, 884],
        [3436, 618, 3551, 728], # 12
        [3623, 1418, 3737, 1526],
        [3615, 1257, 3731, 1366],
        [3609, 1093, 3720, 1208],
        [3606, 936, 3718, 1045],
        [3599, 779, 3712, 884],
        [3592, 627, 3703, 729], # 18
        [3774, 1411, 3894, 1523],
        [3769, 1257, 3887, 1366],
        [3760, 1093, 3880, 1208],
        [3758, 936, 3877, 1046],
        [3763, 779, 3864, 884],
        [3745, 626, 3863, 730]])

def color_correction(img):
    r, g, b = read_colorchecker_gm()
    patch_coord = get_patchs()
    values = []

    for x1, y1, x2, y2 in patch_coord:
        tmp_img = img[y1:y2, x1:x2]
        values.append(np.mean(tmp_img, axis=(0, 1)))

    values = np.stack(values)

    A = np.zeros((72,12))
    A[:24, :3] = values
    A[24:48, 3:6] = values
    A[48:, 6:9] = values
    A[:24, 9] = 1
    A[24:48, 10] = 1
    A[48:, 11] = 1

    b1 = np.zeros(72)
    b1[:24] = r.reshape(-1)
    b1[24:48] = g.reshape(-1)
    b1[48:] = b.reshape(-1)

    result = np.linalg.lstsq(A, b1)[0]

    img = img @ result[:9].reshape(3, 3).T
    img = img + result[-3:].reshape(1, 1, 3)

    # print(img.mean(), img.min(), img.max())
    img = np.clip(img, 0, None)

    x1, y1, x2, y2 = patch_coord[18]
    avg = np.mean(img[y1:y2, x1:x2], axis=(0, 1))
    img[:, :, 0] *= avg[1] / avg[0]
    img[:, :, 2] *= avg[1] / avg[2]


    return img

def tonemapping(img, config="rgb", K=0.1, B=0.95):
    epsilon = 1E-5

    print("config", config)

    if config == "rgb":
        I_m = np.exp(np.mean(np.log(img + epsilon)))
        I_new = K / I_m * img
        I_white = B * np.max(I_new)
        img_tonemapped = I_new * (1 + I_new / (I_white**2)) / (1 + I_new)
    elif config == "xyY":
        img_xyz = lRGB2XYZ(img)

        # Guard only zero denominators; adding epsilon changes dark chromaticities.
        xyz_sum = np.sum(img_xyz, axis=2)
        x = np.divide(img_xyz[:, :, 0], xyz_sum,
                      out=np.zeros_like(xyz_sum), where=xyz_sum != 0)
        y = np.divide(img_xyz[:, :, 1], xyz_sum,
                      out=np.zeros_like(xyz_sum), where=xyz_sum != 0)
        img_Y = img_xyz[:, :, 1]

        I_m = np.exp(np.mean(np.log(img_Y + epsilon)))
        I_new = K / I_m * img_Y
        I_white = B * np.max(I_new)
        if I_white == 0:
            return np.zeros_like(img_xyz)
        img_Y = I_new * (1 + I_new / (I_white**2)) / (1 + I_new)

        # The provided helper adds epsilon to y, so invert explicitly here.
        scale = np.divide(img_Y, y, out=np.zeros_like(img_Y), where=y != 0)
        X = x * scale
        Y = img_Y
        Z = (1 - x - y) * scale
        # print(X.shape, Y.shape, Z.shape)
        img_tonemapped = XYZ2lRGB(np.dstack([X, Y, Z]))


    return img_tonemapped

def save_jpg(filename, data, scale=0.8):
    data *= scale
    print(data.min(), data.max(), data.mean())
    img_display = np.clip(data, 0, 1)
    img_display = np.where(
        img_display <= 0.0031308,
        12.92 * img_display,
        1.055 * img_display ** (1 / 2.4) - 0.055,
    )
    img_display = np.round(img_display * 255).astype(np.uint8)
    imsave(filename, img_display)

def raw_channel_statistics(path, channel="g1", crop=(548, 1805, 1740, 3078), black_img=None, scale=1.0):
    """Stream D3500 mosaic TIFFs; return mean and unbiased temporal variance.

    Crop coordinates refer to the full RAW image. Channel samples retain their
    original Bayer phase. No white balance, interpolation, or clipping is applied.
    """
    import tifffile
    from pathlib import Path
    channel = channel.lower()
    offsets = {"r": (0, 0), "g": (0, 1), "g1": (0, 1), "g2": (1, 0), "b": (1, 1)}
    if channel not in offsets:
        raise ValueError("channel must be r, g, g1, g2, or b")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    y1, x1, y2, x2 = crop
    if not (0 <= y1 < y2 <= 4016 and 0 <= x1 < x2 <= 6016):
        raise ValueError("Invalid full-resolution RAW crop")
    dy, dx = offsets[channel]
    ys, xs = y1 + (dy-y1) % 2, x1 + (dx-x1) % 2
    files = sorted(p for p in Path(path).iterdir()
                   if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    if len(files) < 2:
        raise ValueError(f"Need at least two RAW TIFFs in {path}; run convert_p5_raw.sh first")
    mean = m2 = None
    for n, f in enumerate(files, 1):
        raw = tifffile.imread(f)
        if raw.ndim != 2 or raw.shape != (4016, 6016):
            raise ValueError(f"{f}: expected undemosaiced D3500 RAW, got {raw.shape}")
        data = raw[ys:y2:2, xs:x2:2].astype(np.float64) * scale
        del raw
        if black_img is not None:
            if black_img.shape != data.shape:
                raise ValueError("Dark image must use the same channel, crop, and scale")
            data -= black_img  # Keep negative fluctuations.
        if mean is None:
            mean = np.zeros_like(data)
            m2 = np.zeros_like(data)
        delta = data - mean
        mean += delta / n
        m2 += delta * (data - mean)
    print(f"Loaded {len(files)} frames, channel {channel}, {mean.size} pixels")
    return mean, m2 / (len(files) - 1)


def load_black_img(path, channel="g1", crop=(548, 1805, 1740, 3078), scale=1.0):
    # Second return value is now variance, NOT a sum of squares.
    return raw_channel_statistics(path, channel, crop, scale=scale)


def load_calibrated_img(calibration_path, black_img, channel="g1",
                        crop=(548, 1805, 1740, 3078), scale=1.0):
    return raw_channel_statistics(calibration_path, channel, crop, black_img, scale)


def fit_data(img_avg, img_var, x_threshold=5000, bin_width=1.0, min_count=1):
    """Inputs already use the chosen DN / DN² units; no extra 65535 factors."""
    if bin_width <= 0 or min_count < 1:
        raise ValueError("bin_width and min_count must be positive")
    means, variances = img_avg.ravel(), img_var.ravel()
    valid = np.isfinite(means) & np.isfinite(variances) & (means > 0) & (variances >= 0)
    _, inverse = np.unique(np.rint(means[valid] / bin_width).astype(np.int64), return_inverse=True)
    counts = np.bincount(inverse)
    x = np.bincount(inverse, weights=means[valid]) / counts
    y = np.bincount(inverse, weights=variances[valid]) / counts
    keep = counts >= min_count
    if x_threshold is not None:
        keep &= x <= x_threshold
    x, y = x[keep], y[keep]
    if x.size < 2 or np.ptp(x) == 0:
        raise ValueError("Need at least two distinct populated intensity bins")
    g, b = np.polyfit(x, y, 1)
    predicted = g*x+b
    rmse = np.sqrt(np.mean((y-predicted)**2))
    total = np.sum((y-y.mean())**2)
    r2 = 1-np.sum((y-predicted)**2)/total if total > 0 else np.nan
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=8, alpha=.4, label="Measured bin means")
    plt.plot(x, predicted, "r-", label=f"Fit: y = {g:.4g}x + {b:.4g}")
    plt.xlabel("Black-subtracted mean (chosen DN units)")
    plt.ylabel("Temporal variance (chosen DN² units)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
    print(f"Gain: {g:.6g}; additive variance intercept: {b:.6g}")
    print(f"Bins: {len(x)}, R²: {r2:.6f}, RMSE: {rmse:.6g} DN²")
    return g, b


if __name__ == "__main__":
    from run_hdr import main
    main()
