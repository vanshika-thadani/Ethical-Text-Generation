"""
Run once to generate PNG icons for the extension.
Requires: Pillow (already in the venv)
"""
from PIL import Image, ImageDraw
import os

os.makedirs("icons", exist_ok=True)

for size in [16, 48, 128]:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Indigo rounded square background
    radius = max(2, size // 6)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=(99, 102, 241, 255))

    # White shield: just a centered circle (no text, no font needed)
    pad = size // 5
    draw.ellipse([pad, pad, size - pad - 1, size - pad - 1], fill=(255, 255, 255, 220))

    # Small inner indigo dot
    pad2 = size // 3
    draw.ellipse([pad2, pad2, size - pad2 - 1, size - pad2 - 1], fill=(99, 102, 241, 255))

    img.save(f"icons/icon{size}.png")
    print(f"Created icons/icon{size}.png ({size}x{size})")

print("Done.")
