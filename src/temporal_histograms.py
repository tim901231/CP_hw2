"""Plot temporal histograms at four reproducibly random calibration pixels.

Use repeated fixed-exposure ramp captures, NOT the HDR exposure bracket.
Defaults match the ISO 200 RAW calibration paths and crop in main.py.
Only four samples per frame are read through a TIFF memory map.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile

PROJECT = Path(__file__).resolve().parents[1]
PHASES = {'r': (0, 0), 'g1': (0, 1), 'g2': (1, 0), 'b': (1, 1)}


def tiffs(directory):
    files = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in ('.tif', '.tiff') and p.is_file())
    if len(files) < 2:
        raise ValueError(f'Need at least two repeated TIFF frames in {directory}')
    return files


def random_points(shape, crop, channel, seed):
    y1, x1, y2, x2 = crop
    if len(shape) != 2 or not (0 <= y1 < y2 <= shape[0] and 0 <= x1 < x2 <= shape[1]):
        raise ValueError('Crop must be within a 2D RG/GB RAW mosaic')
    dy, dx = PHASES[channel]
    ys = np.arange(y1 + (dy-y1) % 2, y2, 2)
    xs = np.arange(x1 + (dx-x1) % 2, x2, 2)
    if len(ys)*len(xs) < 4:
        raise ValueError('Crop must contain at least four pixels of the selected channel')
    selected = np.random.default_rng(seed).choice(len(ys)*len(xs), 4, replace=False)
    return np.column_stack((ys[selected // len(xs)], xs[selected % len(xs)]))


def pixel_series(files, points, shape):
    samples = np.empty((len(files), 4), dtype=np.float64)
    for i, path in enumerate(files):
        frame = tifffile.memmap(path, mode='r')
        try:
            if frame.shape != shape or frame.dtype != np.uint16:
                raise ValueError(f'Expected matching uint16 RAW TIFF: {path}')
            samples[i] = frame[points[:, 0], points[:, 1]]
        finally:
            frame._mmap.close()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ramp-dir', type=Path, default=PROJECT/'data/calibration_iso_200_20260925_220351/raw_tiff')
    parser.add_argument('--dark-dir', type=Path, default=PROJECT/'data/black_iso_200_20260925_165853/raw_tiff')
    parser.add_argument('--crop', type=int, nargs=4, default=(829, 1419, 2792, 4502),
                        metavar=('Y1', 'X1', 'Y2', 'X2'), help='Full-image crop; upper bounds exclusive')
    parser.add_argument('--channel', choices=PHASES, default='g1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale', type=float, default=1., help='Signal scale; default native RAW DN')
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, default=PROJECT/'output/temporal_histograms')
    args = parser.parse_args()
    if not np.isfinite(args.scale) or args.scale <= 0 or args.bins < 1:
        parser.error('--scale and --bins must be positive (scale must be finite)')
    ramp, dark = tiffs(args.ramp_dir), tiffs(args.dark_dir)
    frame = tifffile.memmap(ramp[0], mode='r')
    try:
        shape = frame.shape
        points = random_points(shape, args.crop, args.channel, args.seed)
        preview = frame[::8, ::8].copy()
    finally:
        frame._mmap.close()
    raw = pixel_series(ramp, points, shape)
    dark_mean = pixel_series(dark, points, shape).mean(axis=0)
    # Equivalent at these coordinates to subtracting the complete averaged dark
    # image. Subtract one fixed dark estimate; do not pair ramp and dark frames.
    values = (raw - dark_mean) * args.scale
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    colors = ['#277da1', '#e07a20', '#328450', '#8b5aa8']
    units = 'RAW DN' if args.scale == 1. else 'scaled DN'
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    for i, (ax, color) in enumerate(zip(axes.flat, colors)):
        ax.hist(values[:, i], bins=args.bins, color=color, alpha=.85, edgecolor='white')
        ax.axvline(values[:, i].mean(), color='#333333', linestyle='--', linewidth=1)
        y, x = points[i]
        ax.set_title(f'P{i+1}: (x={x}, y={y}) | {args.channel.upper()}', fontsize=11)
        ax.set_xlabel(f'Dark-subtracted intensity ({units})')
        ax.set_ylabel('Frame count')
        ax.text(.97, .96, f'Mean = {values[:,i].mean():.2f}\nSD = {values[:,i].std(ddof=1):.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle(f'Temporal pixel histograms | {len(ramp)} ramp frames, {len(dark)} dark frames', fontsize=14)
    fig.savefig(out/'temporal_histograms.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.imshow(preview, cmap='gray', vmin=float(np.percentile(preview,1)),
              vmax=float(np.percentile(preview,99)), extent=(-.5, shape[1]-.5, shape[0]-.5, -.5))
    from matplotlib.patches import Rectangle
    y1, x1, y2, x2 = args.crop
    ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='cyan', linewidth=1))
    for i, ((y,x), color) in enumerate(zip(points, colors)):
        ax.scatter(x,y,s=90,facecolor=color,edgecolor='white')
        ax.annotate(f'P{i+1}', (x,y), xytext=(8,-12), textcoords='offset points', color='white',
                    bbox=dict(facecolor=color, edgecolor='none', alpha=.85))
    ax.set(title='Randomly selected pixels on the first ramp capture', xlabel='x (column)', ylabel='y (row)')
    fig.savefig(out/'selected_pixels.png', dpi=180)
    plt.close(fig)
    with (out/'samples.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'P1', 'P2', 'P3', 'P4'])
        for path,row in zip(ramp,values): writer.writerow([path.name,*row])
    summary = {'ramp_files': [str(p) for p in ramp], 'dark_files': [str(p) for p in dark],
               'seed':args.seed, 'crop_y1_x1_y2_x2':args.crop, 'channel':args.channel,
               'bayer_pattern':'RG/GB', 'coordinate_convention':'zero-based full RAW image',
               'signal_scale':args.scale, 'units':units, 'histogram_bins':args.bins,
               'points':[{'id':f'P{i+1}', 'x':int(x),'y':int(y),
                          'dark_mean_raw_DN':float(dark_mean[i]),
                          'mean':float(values[:,i].mean()),
                          'variance_ddof1':float(values[:,i].var(ddof=1))}
                         for i,(y,x) in enumerate(points)]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Saved histograms, pixel locations, samples, and metadata to {out}')
    print(json.dumps(summary['points'], indent=2))


if __name__ == '__main__':
    main()
