# InspectEdge - AI-Based Fruit Inspection System

## Overview
InspectEdge is an AI-powered system designed to detect and classify rotten and fresh fruits using YOLOv5. It utilizes an ESP32-CAM module for real-time image capture and sends frames to a detection script running on a local machine. The system supports multiple fruit categories, including:
- **Banana**
- **Apple**
- **Orange**
- **Guava**
- **Grapes**

## Features
- **Real-time Detection:** Uses a webcam or ESP32-CAM stream for live inference.
- **Multi-Model Support:** Automatically selects the best YOLOv5 model for each fruit category.
- **ESP32-CAM Integration:** Captures and streams images from an ESP32-based web server.
- **Torch and OpenCV-Based Inference:** Uses PyTorch and OpenCV for efficient processing.
- **Automated Model Selection:** Detects and loads the appropriate model dynamically.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- OpenCV (`pip install opencv-python`)
- PyTorch (`pip install torch torchvision torchaudio`)
- Ultralytics YOLOv5 (`pip install ultralytics`)
- ESP32 Setup for image streaming

### Clone the Repository
```bash
git clone https://github.com/yourusername/InspectEdge.git
cd InspectEdge
```

### Setup Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Detection Script
To start live detection:
```bash
python esp32_detect.py
```
If using an ESP32-CAM for image input, modify the script to use the ESP32 stream URL:
```python
cap = cv2.VideoCapture("http://<ESP32-IP>/capture")
```

## ESP32-CAM Setup
1. Flash your ESP32 with the appropriate firmware to start a web server.
2. Access the ESP32 stream URL in a browser to verify the camera feed.
3. Update `esp32_detect.py` with the ESP32 URL for real-time processing.

## Model Weights
Store YOLOv5 weights in `weights/` and specify paths in `esp32_detect.py`:
```python
models = {
    "banana": "weights/banana.pt",
    "apple": "weights/apple.pt",
    "orange": "weights/orange.pt",
    "guava": "weights/guava.pt",
    "grapes": "weights/grapes.pt"
}
```

## Uploading Large Files to GitHub
For large files (>100MB), use Git LFS:
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "Added model weights"
git push origin main
```

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m "Added new feature"`)
4. Push to GitHub (`git push origin feature-name`)
5. Create a pull request

## License
This project is open-source and available under the MIT License.

