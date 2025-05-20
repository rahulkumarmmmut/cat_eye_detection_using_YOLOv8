from ultralytics import YOLO

model   = YOLO('runs/detect/train/weights/best.pt')
results = model.predict(source='img_523.jpg', conf=0.3)

# Grab the first (and only) result
res = results[0]
if res.boxes:
    # Get the first detected box
    box = res.boxes[0]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print("Box corners:", x1, y1, x2, y2)
    cx = (x1 + x2)/2
    cy = (y1 + y2)/2
    print("Center:", cx, cy)