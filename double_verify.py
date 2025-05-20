# viz.py
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def visualize(img_path, x1, y1, x2, y2, out_path=None):
    # load
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # draw box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # compute & draw center
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = 5
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="blue")

    # annotate coords
    text = f"({cx:.1f}, {cy:.1f})"
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    draw.text((cx + 10, cy - 10), text, fill="white", font=font)

    # show or save
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    if out_path:
        img.save(out_path)
        print(f"Saved visualization to {out_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7):
        print("Usage: python viz.py image.jpg x1 y1 x2 y2 [out.png]")
        sys.exit(1)

    img_path = sys.argv[1]
    x1, y1, x2, y2 = map(float, sys.argv[2:6])
    out_path = sys.argv[6] if len(sys.argv) == 7 else None

    visualize(img_path, x1, y1, x2, y2, out_path)