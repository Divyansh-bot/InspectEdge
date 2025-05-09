import sys
import pathlib
import torch
import cv2
import numpy as np
import os

# -------------------- PATH FIXES --------------------

# Fix PosixPath error on Windows
pathlib.PosixPath = pathlib.WindowsPath

# Add yolov5 to sys.path
sys.path.append('D:/project/PackTrack_AI/ultralytics/yolov5')  # Update this if folder is moved

# -------------------- DEVICE SETUP --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------- MODEL LOADING --------------------
models = {
    "banana": torch.hub.load('ultralytics/yolov5', 'custom', path='D:/project/PackTrack_AI/ultralytics/yolov5/runs/train/banana_model2/weights/best.pt', source='local'),
    "fresh_oranges": torch.hub.load('ultralytics/yolov5', 'custom', path='D:/project/PackTrack_AI/ultralytics/yolov5/runs/train/fresh_oranges_model2/weights/best.pt', source='local'),
    "freshapple": torch.hub.load('ultralytics/yolov5', 'custom', path='D:/project/PackTrack_AI/ultralytics/yolov5/runs/train/freshapple_model/weights/best.pt', source='local'),
    "grapes": torch.hub.load('ultralytics/yolov5', 'custom', path='D:/project/PackTrack_AI/ultralytics/yolov5/runs/train/grapes_model/weights/best.pt', source='local'),
    "guava": torch.hub.load('ultralytics/yolov5', 'custom', path='D:/project/PackTrack_AI/ultralytics/yolov5/runs/train/guava_model/weights/best.pt', source='local'),
    "rotten_oranges": torch.hub.load('ultralytics/yolov5', 'custom', path='D:/project/PackTrack_AI/ultralytics/yolov5/runs/train/rotten_oranges_model/weights/best.pt', source='local'),
    "rottenapple": torch.hub.load('ultralytics/yolov5', 'custom', path='D:/project/PackTrack_AI/ultralytics/yolov5/runs/train/rottenapple_model/weights/best.pt', source='local')
}



# Move models to GPU
for name in models:
    models[name] = models[name].to(device)
    print(f"✅ {name} model loaded on {device}")

# -------------------- VIDEO STREAM --------------------
cap = cv2.VideoCapture(0)  # 0 for webcam or use ESP32 URL like "http://192.168.x.x/capture"
CONF_THRESHOLD = 0.5

def get_best_model(frame):
    best_model_name = None
    best_results = None
    best_conf = 0.0

    for name, model in models.items():
        result = model(frame)
        preds = result.xyxy[0].cpu().numpy()
        if len(preds) > 0:
            conf = preds[:, 4].max()
            if conf > CONF_THRESHOLD and conf > best_conf:
                best_model_name = name
                best_results = result
                best_conf = conf

    return best_model_name, best_results

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Error reading frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    best_model_name, results = get_best_model(rgb_frame)

    if results:
        print(f"✅ Detected using: {best_model_name}")
        for *xyxy, conf, cls in results.xyxy[0]:
            label = f'{results.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        print("❗ No confident detection")

    cv2.imshow("InspectEdge - Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
