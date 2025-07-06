from ultralytics import YOLO

# Load a YOLOv8 model (could be 'yolov8n', 'yolov8s', etc.)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="dataset.yaml",   # your dataset config (YOLO format)
    epochs=100,            # number of training epochs
    imgsz=640,             # image size
    batch=16,              # batch size
    project="runs",        # where to save results
    name="yolo_train",     # experiment name
)
