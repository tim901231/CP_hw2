#!/bin/bash
# Preserve the RAW mosaic and black offset in Python-readable 16-bit TIFFs.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DCRAW="$(command -v dcraw || true)"
[[ -n "$DCRAW" ]] || { echo "dcraw was not found." >&2; exit 1; }
if [[ $# -eq 0 ]]; then
    set -- "$PROJECT_DIR/data/black_iso_200_20260925_165853" \
        "$PROJECT_DIR/data/black_iso_400_20260925_160633" \
        "$PROJECT_DIR/data/black_iso_800_20260925_161032" \
        "$PROJECT_DIR/data/black_iso_1600_20260925_163736" \
        "$PROJECT_DIR/data/black_iso_3200_20260925_164745"
fi
shopt -s nullglob nocaseglob
temp_file=""
trap '[[ -z "$temp_file" ]] || rm -f -- "$temp_file"' EXIT
for input_dir in "$@"; do
    files=("$input_dir"/*.nef)
    [[ ${#files[@]} -gt 0 ]] || { echo "No NEFs found: $input_dir" >&2; exit 1; }
    output_dir="$input_dir/raw_tiff"
    mkdir -p "$output_dir"
    count=0
    for input in "${files[@]}"; do
        name="${input##*/}"
        output="$output_dir/${name%.*}.tiff"
        count=$((count + 1))
        if [[ -s "$output" ]]; then
            echo "[$count/${#files[@]}] Already exists: $output"
            continue
        fi
        echo "[$count/${#files[@]}] Converting $input"
        temp_file="$(mktemp "$output_dir/.dcraw_XXXXXX")"
        # No WB, demosaicing, color conversion, or black-level subtraction.
        "$DCRAW" -D -4 -j -t 0 -T -c "$input" > "$temp_file"
        [[ -s "$temp_file" ]] || { echo "Empty output: $input" >&2; exit 1; }
        mv -- "$temp_file" "$output"
        temp_file=""
    done
done
echo "Done. Each set has its own raw_tiff folder."
