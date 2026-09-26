#!/bin/bash
# Exposure 8's camera WB, fixed across the whole bracket. Keep original TIFFs.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="$PROJECT_DIR/data/room_stack"
OUTPUT_DIR="$INPUT_DIR/tiff_fixed_wb"
mkdir -p "$OUTPUT_DIR"
temp_file=""
trap '[[ -z "$temp_file" ]] || rm -f -- "$temp_file"' EXIT
for i in {1..16}; do
    input="$INPUT_DIR/exposure$i.nef"
    output="$OUTPUT_DIR/exposure$i.tiff"
    if [[ -s "$output" ]]; then echo "Already exists: $output"; continue; fi
    temp_file="$(mktemp "$OUTPUT_DIR/.dcraw_XXXXXX")"
    echo "Converting exposure$i with fixed white balance"
    dcraw -r 1.41015625 1 2.20703125 1 -q 3 -o 1 -4 -T -c "$input" > "$temp_file"
    [[ -s "$temp_file" ]]
    mv -- "$temp_file" "$output"
    temp_file=""
done
