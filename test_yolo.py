'''from ultralytics import YOLO

model = YOLO("weights/best.pt")

import os

test_dir = "test_images"
for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    results = model.predict(source=img_path, conf=0.5, save=False, show=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"Image: {img_file}")
            print(f"Predicted: {model.names[cls_id]} with confidence {conf:.2f}")
            print(f"Bounding box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            print()'''

'''from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("weights/best.pt")

# Load one of your test images
img = cv2.imread("rock_paper_scissor/test_images/25_jpg.rf.121bbf3873487349ba7e5e2cde55dcef.jpg")
  # replace with an actual image path you have

if img is None:
    print("Image not found!")
else:
    print(f"Image shape: {img.shape}")
    results = model.predict(source=img, show=True, conf=0.5)'''


import cv2
from ultralytics import YOLO

model = YOLO("weights/best.pt")

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(source=frame, conf=0.5)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Webcam", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
