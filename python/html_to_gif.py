#!/usr/bin/env python3
"""
HTML Player to GIF Converter
=============================

Converts the HTML Player file (from Brownian simulation) to an animated GIF.

Usage:
    python html_to_gif.py input.html output.gif [--fps 15] [--scale 0.5]

Requirements:
    pip install Pillow

Examples:
    python html_to_gif.py brownian_player.html animation.gif
    python html_to_gif.py brownian_player.html animation.gif --fps 20
    python html_to_gif.py brownian_player.html animation.gif --fps 15 --scale 0.5
"""

import re
import base64
import argparse
import sys
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install it with:")
    print("  pip install Pillow")
    sys.exit(1)


def extract_frames_from_html(html_path):
    """Extract base64 PNG frames from HTML player file."""
    print(f"Reading {html_path}...")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the frames array in the JavaScript
    # Pattern: var frames=[...] or frames=[...]
    pattern = r'var\s+frames\s*=\s*\[(.*?)\];'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        # Try alternative pattern
        pattern = r'frames\s*=\s*\[(.*?)\];'
        match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Error: Could not find frames array in HTML file.")
        print("Make sure this is an HTML Player file from the Brownian simulation.")
        sys.exit(1)
    
    frames_str = match.group(1)
    
    # Extract all base64 data URLs
    # Pattern: "data:image/png;base64,..."
    base64_pattern = r'"(data:image/png;base64,[^"]+)"'
    base64_matches = re.findall(base64_pattern, frames_str)
    
    if not base64_matches:
        print("Error: No PNG frames found in HTML file.")
        sys.exit(1)
    
    print(f"Found {len(base64_matches)} frames.")
    return base64_matches


def decode_frame(data_url):
    """Decode a base64 data URL to PIL Image."""
    # Remove the data URL prefix
    base64_data = data_url.split(',')[1]
    image_data = base64.b64decode(base64_data)
    return Image.open(BytesIO(image_data))


def create_gif(frames_data, output_path, fps=15, scale=1.0, max_frames=None):
    """Create animated GIF from base64 frame data."""
    
    # Calculate frame duration in milliseconds
    duration = int(1000 / fps)
    
    print(f"Decoding frames...")
    images = []
    
    # Optionally limit frames
    if max_frames and len(frames_data) > max_frames:
        # Sample frames evenly
        step = len(frames_data) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        frames_to_process = [frames_data[i] for i in indices]
        print(f"Sampling {max_frames} frames from {len(frames_data)} total.")
    else:
        frames_to_process = frames_data
    
    for i, data_url in enumerate(frames_to_process):
        img = decode_frame(data_url)
        
        # Convert to RGB (GIF doesn't support RGBA well)
        if img.mode == 'RGBA':
            # Create a background and paste
            background = Image.new('RGB', img.size, (15, 23, 42))  # Dark background
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Scale if needed
        if scale != 1.0:
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        
        images.append(img)
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == len(frames_to_process) - 1:
            print(f"  Processed {i + 1}/{len(frames_to_process)} frames...")
    
    print(f"Creating GIF at {fps} fps (frame duration: {duration}ms)...")
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # Loop forever
        optimize=True
    )
    
    # Get file size
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Done! Saved to: {output_path}")
    print(f"  Frames: {len(images)}")
    print(f"  Duration: {len(images) * duration / 1000:.1f}s")
    print(f"  Size: {size_mb:.2f} MB")
    
    if size_mb > 10:
        print(f"\nTip: GIF is large. To reduce size, try:")
        print(f"  python html_to_gif.py {sys.argv[1]} {output_path} --scale 0.5")
        print(f"  python html_to_gif.py {sys.argv[1]} {output_path} --max-frames 100")


def main():
    parser = argparse.ArgumentParser(
        description='Convert HTML Player file to animated GIF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python html_to_gif.py recording.html output.gif
  python html_to_gif.py recording.html output.gif --fps 20
  python html_to_gif.py recording.html output.gif --scale 0.5
  python html_to_gif.py recording.html output.gif --max-frames 150
        """
    )
    
    parser.add_argument('input', help='Input HTML player file')
    parser.add_argument('output', help='Output GIF file')
    parser.add_argument('--fps', type=int, default=15, 
                        help='Frames per second (default: 15)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor, e.g. 0.5 for half size (default: 1.0)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to include (default: all)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.endswith('.html'):
        print("Warning: Input file doesn't have .html extension")
    
    if not args.output.endswith('.gif'):
        args.output += '.gif'
    
    if args.fps < 1 or args.fps > 60:
        print("Error: FPS should be between 1 and 60")
        sys.exit(1)
    
    if args.scale <= 0 or args.scale > 2:
        print("Error: Scale should be between 0.1 and 2.0")
        sys.exit(1)
    
    # Extract and convert
    frames_data = extract_frames_from_html(args.input)
    create_gif(frames_data, args.output, args.fps, args.scale, args.max_frames)


if __name__ == '__main__':
    main()
