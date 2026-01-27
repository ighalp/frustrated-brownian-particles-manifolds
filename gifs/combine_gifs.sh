#!/bin/bash
# Script to create an optimized combined GIF from three geometry simulations
# Usage: ./combine_gifs.sh [output_height] [fps]
# Defaults: height=300, fps=10

HEIGHT=${1:-300}
FPS=${2:-10}
OUTPUT="combined_optimized.gif"

cd "$(dirname "$0")"

echo "Creating combined GIF with height=${HEIGHT}px, fps=${FPS}..."

# Step 1: Create a combined video with scaled inputs
# Step 2: Generate an optimized palette
# Step 3: Create the final GIF using the palette

ffmpeg -y \
  -i Dimension_reduction_on_the_sphere.gif \
  -i Dimension_reduction_on_the_cylinder.gif \
  -i Dimension_reduction_on_a_torus.gif \
  -filter_complex "
    [0:v]scale=-1:${HEIGHT}:flags=lanczos,fps=${FPS}[v0];
    [1:v]scale=-1:${HEIGHT}:flags=lanczos,fps=${FPS}[v1];
    [2:v]scale=-1:${HEIGHT}:flags=lanczos,fps=${FPS}[v2];
    [v0][v1][v2]hstack=inputs=3[stacked];
    [stacked]split[s0][s1];
    [s0]palettegen=max_colors=128:stats_mode=diff[p];
    [s1][p]paletteuse=dither=bayer:bayer_scale=3
  " \
  "$OUTPUT"

# Show result
SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
echo "Created $OUTPUT ($SIZE)"
