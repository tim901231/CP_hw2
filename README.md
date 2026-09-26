# Computational Photography — Assignment 2: HDR Imaging

Code for HDR merging, color correction, photographic tonemapping, noise calibration,
and the bonus experiment varying shutter speed and ISO.

## Setup

Run commands from the project root. The code was checked with Python 3.8 in the
`cp_hw1` environment. Install the Python dependencies into your environment:

```bash
python -m pip install -r requirements.txt
```

Install **dcraw** for RAW conversion and **ExifTool** for shutter/ISO metadata.
On macOS with Homebrew:

```bash
brew install dcraw exiftool
```

For a machine without a display, set `MPLBACKEND=Agg` before running Python.
Existing outputs are replaced when a command is rerun with the same output path.

## Input data

Keep the following folders relative to this README:

```text
data/
  door_stack/       exposure1.nef/.jpg/.tiff through exposure16
  room_stack/       exposure1.nef/.jpg/.tiff through exposure16
  room_fix_iso/     1.NEF through 8.NEF
  room_vary_iso/    4.NEF through 8.NEF (reuses fixed frames 1–3)
  calibration_iso_*/raw_tiff/   repeated ramp captures for each ISO
  black_iso_*/raw_tiff/         repeated dark captures for each ISO
  noise_calibration/iso_noise_profile.json
```

The regular `.tiff` stack files are developed linear RGB images. Calibration
`raw_tiff` files are undemosaiced RAW mosaics; these are different input types.
The scripts resolve default data locations relative to the project directory.

## Parts 1–4: merging, color correction, and tonemapping

If the developed TIFFs are missing, convert the original stacks:

```bash
for scene in door room; do
  for file in data/${scene}_stack/*.nef; do
    dcraw -w -q 3 -o 1 -4 -T "$file"
  done
done
```

Generate all eight combinations of merge domain (linear/log) and weight
(uniform/tent/Gaussian/photon), for each input format:

```bash
python src/run_hdr.py merge --scene door --format tiff --all-variants
python src/run_hdr.py merge --scene door --format jpg --all-variants
```

Results go to `output/reproduced/door_tiff/` and `door_jpg/`. Each combination
produces an HDR and gamma-encoded JPEG preview. JPEG runs also save the recovered
response curves; `settings.json` records settings and shutter times. Door shutter
times use the assignment schedule; room times are read from the original NEFs.
For a single result, omit `--all-variants` and choose `--domain` and `--weight`.

Apply color correction to the door result, then tonemap it in RGB and luminance:

```bash
python src/run_hdr.py color --input output/reproduced/door_jpg/door_log_gaussian.hdr --output output/reproduced/door_corrected.hdr
python src/run_hdr.py tonemap --input output/reproduced/door_corrected.hdr --output output/reproduced/door_rgb.jpg --method rgb --key 0.05 --burn 0.95
python src/run_hdr.py tonemap --input output/reproduced/door_corrected.hdr --output output/reproduced/door_xyY.jpg --method xyY --key 0.05 --burn 0.95
```

Color correction uses saved ColorChecker coordinates for the original door image;
it is not a general-purpose correction for other scenes or resized inputs.

Merge and tonemap the captured room scene:

```bash
python src/run_hdr.py merge --scene room --format tiff --domain log --weight gaussian
python src/run_hdr.py merge --scene room --format jpg --domain log --weight gaussian
python src/run_hdr.py tonemap --input output/reproduced/room_tiff/room_log_gaussian.hdr --output output/reproduced/room_tonemapped.jpg --method rgb
```

TIFF merging uses 256 × 256 patches; `--tile-size 128` reduces temporary memory.
TIFFs must be contiguous, uncompressed uint16 images. JPEG merging streams frames;
only downsampled response-calibration samples are stacked. Final HDR arrays and
rendering still require full-image memory. The merge previews use a fixed scalar
(`--preview-scale`); use `tonemap` for the photographic tonemapping result.

Use `python src/run_hdr.py --help` or a subcommand followed by `--help` for options.
`python src/main.py` accepts the same commands.

## Part 5: noise calibration and optimal merging

If RAW mosaic TIFFs have not been prepared, convert the calibration captures:

```bash
bash src/convert_dark_raw.sh data/calibration_iso_* data/black_iso_*
```

This creates `raw_tiff/` inside each capture folder, preserving the black offset
and Bayer pattern. Existing files are skipped. The default ISO 200 analysis uses
ramp set `calibration_iso_200_20260925_220351` and dark set
`black_iso_200_20260925_165853`.

Generate the mean–variance fit and four randomly selected temporal histograms:

```bash
python src/run_hdr.py noise-fit
python src/temporal_histograms.py --scale 16.612167300380228
```

The fit and measured black variance are saved in `output/noise_fit/`; histograms,
pixel locations, samples, and statistics go to `output/temporal_histograms/`.
The histogram command uses repeated fixed-exposure captures, not the HDR bracket.
Seed 42 makes selection reproducible; change it with `--seed`. The explicit scale
above expresses the histograms in the same scaled TIFF units as the report.

Generate the Part 5 fixed-white-balance RGB optimal merge:

```bash
bash src/convert_room_fixed_wb.sh
python src/run_hdr.py optimal-room --tile-size 256
```

Outputs are saved in `data/room_stack/optimal_merge_rgb/`. This is the earlier
RGB approximation using RAW-derived noise coefficients. The bonus below applies
the noise model directly to RAW samples before demosaicing.

## Bonus: HDR by varying both shutter speed and ISO

The saved `data/noise_calibration/iso_noise_profile.json` contains per-channel
`g`, additive variance, and black mean for ISO 200/400/800/1600/3200. Reuse it to
avoid repeating calibration. To rebuild it from the repeated ramp and dark TIFFs:

```bash
python src/build_iso_calibration.py
```

Gain comes from the ramp mean–variance slope; additive variance comes from the
measured temporal variance of black frames, not the ramp-fit intercept.
The profile stores native RAW units for computation. Report tables use the same
scaled TIFF convention as Part 5: with `S = 65535/(4095-150)`,
`g_TIFF = S*g_RAW` and `var_TIFF = S^2*var_RAW`. Do not substitute the scaled
numbers into the native-RAW merging calculation.

Run both eight-image brackets and generate the detailed comparisons:

```bash
python src/merge_iso_aware.py --reuse-fixed-first-three --target-mean 0.225
python src/visualize_iso_differences.py
```

The reuse flag supplies varying-ISO frames 1–3 from `room_fix_iso`, as intended
for this dataset. EXIF supplies ISO and shutter times. Missing mosaic TIFFs are
automatically decoded into the output cache. Both brackets have 10.668833 seconds
of total integration time; manual capture does not enforce equal wall-clock time.

For each pixel, the optimal estimator uses:

```text
z = observed_RAW - measured_channel_black_mean
flux = z / (g * exposure_time)
var = g * max(z, 0) + measured_black_additive_variance
weight = (g * exposure_time)^2 / var
merged_flux = sum(weight * flux) / sum(weight)
```

Signed samples are retained. Saturated/invalid samples receive zero weight;
pixels without valid samples use a flagged shortest-exposure fallback.

Outputs in `output/iso_aware_merge/` include:

- `*_bayer_electrons_per_second.tiff` and `*_valid_mask.tiff`: RAW HDR and validity.
- `*_camera_rgb_signed.tiff` and `*_camera_rgb.hdr`: demosaiced HDR results.
- `*_preview.jpg`, `*_preview.png`, and comparison images: display renders.
- `calibration_used.json` and `merge_report.json`: exact calibration and run metadata.
- `difference_analysis/`: brightened shadow crops, difference maps, and statistics.

A shared scalar sets the fixed-ISO linear mean to 0.225 before clipping and sRGB
encoding; the varying-ISO result uses that same scalar. To change brightness
without merging again:

```bash
python src/merge_iso_aware.py --render-only --target-mean 0.225
```

The bonus renders use common white balance but no camera-to-sRGB color matrix.
Dark subtraction uses per-channel offsets, not exposure-dependent dark-current
maps. Difference images include texture, motion, and brightness differences;
they do not measure temporal noise from a single bracket pair.

## Source files and checks

| File in `src/` | Purpose |
| --- | --- |
| `run_hdr.py` | Command-line runner for Parts 1–5 |
| `main.py`, `cp_hw2.py` | HDR algorithms and assignment helpers |
| `tiled_merge.py` | Memory-efficient TIFF patch loading and merging |
| `build_iso_calibration.py` | Reusable gain and black-noise calibration |
| `merge_iso_aware.py` | RAW-domain optimal merge for the bonus |
| `temporal_histograms.py` | Four-pixel temporal noise plots |
| `visualize_iso_differences.py` | Matched crops and bonus comparison figures |
| `convert_dark_raw.sh`, `convert_room_fixed_wb.sh` | RAW preparation |
| `test_iso_aware.py`, `test_tiled_merge.py` | Numerical regression tests |

```bash
MPLBACKEND=Agg python -m unittest discover -s src -p 'test_*.py' -v
```

## Report and submission files

The submitted report is `report.pdf`. The local editable LaTeX source and
figures are bundled in `output/report/hw2_latex_source.zip`; upload/extract that ZIP
in Overleaf. `bonus_section.tex` is a standalone copy already included in
`report.tex`, so do not include it a second time.

Include `src/`, this README, `requirements.txt`, the report, and the image/HDR/data
files required by the assignment. Keep the saved calibration profile if including
the bonus code. The commands above need the datasets described under Input data;
source files alone cannot reproduce the capture-dependent results.

Exploratory scripts removed during cleanup are recoverable from
`output/cleanup_backup/pre_submission_cleanup.zip`. Exclude this backup, `tmp/`,
`__pycache__/`, and `.DS_Store` from the submission.

### Data included in this Git submission

| Assignment result | Submitted file |
| --- | --- |
| Part 1 selected HDR | `data/door_jpg/door_log_gaussian.hdr` |
| Part 2 corrected HDR | `data/door_color_correction.hdr` |
| Part 4 RAW-derived HDR | `data/room_tiff/room_log_gaussian.hdr` |
| Part 4 JPEG-derived HDR | `data/room_jpg/room_log_gaussian.hdr` |
| Part 5 optimal HDR | `data/room_stack/optimal_merge_rgb/room_optimal_fixed_wb.hdr` |
| Part 4 original RAW/JPEG pair | `data/room_stack/exposure8.nef` and `exposure8.jpg` |
| Bonus fixed/varying HDRs | `output/iso_aware_merge/{fixed_iso,varying_iso}_camera_rgb.hdr` |
| Bonus original brackets | `data/room_fix_iso/1.NEF`–`8.NEF`, `data/room_vary_iso/4.NEF`–`8.NEF` |
| Saved noise calibration | `data/noise_calibration/iso_noise_profile.json` and `.csv` |

The included bonus originals and saved profile allow both bonus HDRs to be
regenerated with the command above. Run merging before the bonus visualization
command to recreate its required signed TIFF intermediates and current file paths.
The full door/room stacks, repeated calibration captures, decoded TIFF caches,
and LaTeX sources remain local and are excluded from this Git submission.
Recomputing Parts 1–5 or rebuilding calibration requires those additional datasets.
The report PDF already includes the report figures.
