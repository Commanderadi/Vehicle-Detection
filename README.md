# Vehicle Detection (Haar Cascades + YOLOv3)

Two approaches to detecting vehicles with OpenCV, side by side: a fast classical **Haar cascade** detector for video, and a deep-learning **YOLOv3** detector (COCO classes) for images.

## Approaches

| Script | Method | Input | Notes |
|---|---|---|---|
| `Vehicle Detection through Haar cascading.py` | Haar cascade (`haarcascade_car.xml`) | Video / webcam | Very fast and CPU-friendly; size filtering to cut false positives |
| `Vehicle Detection through yolo and coco cfg.py` | YOLOv3 via `cv2.dnn` | Image | Far more accurate; detects all 80 COCO classes (car, bus, truck, motorbike…) |

**Takeaway:** Haar cascades run in real time on a CPU but miss angled or partly hidden vehicles and produce more false positives. YOLOv3 is much more robust but heavier.

## Setup

```bash
git clone https://github.com/Commanderadi/Vehicle-Detection.git
cd Vehicle-Detection
pip install opencv-python numpy
```

YOLOv3 weights (~240 MB) are not stored in the repo. Download `yolov3.weights` from the [official YOLO site](https://pjreddie.com/media/files/yolov3.weights) into the project folder.

## Run

```bash
# Haar cascade on the sample video (press q to quit)
python "Vehicle Detection through Haar cascading.py"

# YOLOv3 on the sample image
python "Vehicle Detection through yolo and coco cfg.py"
```

For a webcam, change `cv2.VideoCapture('test_video.mp4')` to `cv2.VideoCapture(0)`.

## Files

- `haarcascade_car.xml`: pre-trained car cascade
- `yolov3.cfg`, `coco.names`: YOLOv3 network config and class labels
- `test_image.jpg`, `test_video.mp4`: sample inputs

## Stack

Python · OpenCV (`cv2.dnn`, CascadeClassifier) · NumPy
