# auto_viz.py
import sys
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def visualize(img, x1, y1, x2, y2, out_path=None):
    draw = ImageDraw.Draw(img)
    # draw the red bounding box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    # compute & draw the blue center dot
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = 5
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="blue")
    # annotate the coordinates
    text = f"({cx:.1f}, {cy:.1f})"
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    draw.text((cx + 10, cy - 10), text, fill="white", font=font)

    # display
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")

    # save if requested
    if out_path:
        img.save(out_path)
        print(f"Saved annotated image to {out_path}")

def main(image_path, weights_path="runs/detect/train/weights/best.pt", conf=0.3, out_path=None):
    # 1. Load model
    model = YOLO(weights_path)

    # 2. Run inference
    results = model.predict(source=image_path, conf=conf, verbose=False)

    # 3. Grab the first result
    res = results[0]
    if not res.boxes:
        print("❌ No boxes detected.")
        return

    # 4. Get the first box's corners
    x1, y1, x2, y2 = res.boxes[0].xyxy[0].tolist()
    print(f"Detected box corners: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})")

    # 5. Load the image via PIL
    img = Image.open(image_path).convert("RGB")

    # 6. Visualize + show/save
    visualize(img, x1, y1, x2, y2, out_path)

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python auto_viz.py IMAGE.jpg [OUTPUT.png] [CONF]")
        sys.exit(1)

    img_path = sys.argv[1]
    out = None
    conf_val = 0.3

    # optional second arg = output path
    if len(sys.argv) >= 3:
        out = sys.argv[2]
    # optional third arg = confidence threshold
    if len(sys.argv) == 4:
        conf_val = float(sys.argv[3])

    main(img_path, out_path=out, conf=conf_val)