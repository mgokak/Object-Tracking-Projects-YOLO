# Car and Bus Detection

## Overview

This repository contains a computer vision project that detects **cars and buses** from video or image input using deep learning–based object detection. The system processes frames sequentially, identifies vehicles, and draws bounding boxes with class labels.

The objective of this project is to demonstrate how object detection models can be applied to **traffic monitoring and intelligent transportation systems**. The project showcases a real-time inference pipeline suitable for surveillance, smart city applications, and road analytics.

This project demonstrates:
- Video and image processing using OpenCV  
- Deep learning–based multi-class object detection  
- Real-time inference workflow  
- Practical transportation and surveillance use cases  

Applications include traffic monitoring, vehicle counting, congestion analysis, and smart city infrastructure.

---

## Project Flow

The system follows a structured pipeline:

1. Import required libraries  
2. Load trained detection model  
3. Read video stream or image input  
4. Extract frames sequentially  
5. Run object detection on each frame  
6. Draw bounding boxes and class labels  
7. Display or save the processed output  

```
Input → Frame Extraction → Model Inference → Vehicle Detection → Visualization
```

This frame-by-frame processing enables real-time analysis for traffic surveillance systems.

---

## Installation

Install the required dependencies:

```
pip install opencv-python numpy ultralytics matplotlib
```

---

## Project Structure

```
Car_Bus_Detection.ipynb
best.pt
input_video.mp4
output_video.mp4
README.md
```

---

## Import Libraries

```python
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
```

### Explanation

- **cv2** – Handles video capture and frame processing  
- **numpy** – Supports numerical and array operations  
- **YOLO** – Loads and runs the object detection model  
- **matplotlib** – Used for optional visualization  

These libraries form the foundation of the real-time detection pipeline.

---

## Load the Detection Model

```python
model = YOLO("best.pt")
```

### Explanation

This step loads the trained weights into memory. The model has learned features that allow it to recognize different vehicle types such as cars and buses. Once loaded, it is ready to perform inference on new frames.

---

## Video Input Processing

```python
cap = cv2.VideoCapture("input_video.mp4")
```

Frame extraction loop:

```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
```

### Explanation

The video is processed frame-by-frame instead of loading the entire file into memory. This method:
- Reduces memory usage  
- Supports real-time processing  
- Enables deployment with live camera feeds  

This approach is commonly used in traffic surveillance and monitoring systems.

---

## Object Detection on Frames

```python
results = model(frame, conf=0.5)
```

### Explanation

Each frame is passed through the neural network, which returns:
- Bounding box coordinates  
- Confidence scores  
- Predicted class labels (car or bus)  

The confidence threshold filters out weak or unreliable detections.

---

## Drawing Bounding Boxes and Labels

```python
for result in results:
    boxes = result.boxes.xyxy
    classes = result.boxes.cls

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
```

### Explanation

Bounding boxes highlight detected vehicles, and class labels indicate whether the object is a car or a bus. This visual output helps verify model performance and interpret results easily.

---

## Display Output

```python
cv2.imshow("Car and Bus Detection", frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

Cleanup:

```python
cap.release()
cv2.destroyAllWindows()
```

### Explanation

Frames are displayed continuously to create a live monitoring interface. Proper resource cleanup ensures stable execution during long-running operations.

---

## How YOLO Works

YOLO (You Only Look Once) is a single-stage object detection algorithm that analyzes the entire image in one pass. Instead of scanning multiple regions separately, the model predicts object locations and class probabilities simultaneously.

This makes YOLO:
- Fast  
- Efficient  
- Suitable for real-time traffic monitoring  

The model divides the image into grid regions and predicts bounding boxes along with confidence scores for each region.

---

## Real-Time Traffic Monitoring Concept

Video-based vehicle detection introduces challenges such as:
- Motion blur  
- Varying lighting conditions  
- Occlusions between vehicles  
- Different vehicle sizes and speeds  

Frame-by-frame processing allows continuous analysis and immediate response, which is essential for intelligent transportation systems.

---

## Practical Applications

This system can be used for:

- Traffic volume monitoring  
- Vehicle counting  
- Congestion detection  
- Smart signal control systems  
- Highway surveillance  
- Urban mobility analytics  

With additional logic, the system can generate traffic statistics and detect abnormal traffic patterns.

---

## Detection Pipeline (Detailed)

Each frame goes through the following stages:

1. Frame Capture  
2. Preprocessing (resize and normalization)  
3. Model Inference  
4. Post-processing (confidence filtering and NMS)  
5. Visualization  

This loop runs continuously for real-time operation.

---

## Limitations

Current implementation focuses only on detection and does not include:

- Vehicle tracking across frames  
- Automatic vehicle counting  
- Speed estimation  
- Traffic analytics dashboard  

These features can be added in future enhancements.

---

## Future Improvements

Possible extensions include:

- Multi-object tracking (SORT / DeepSORT)  
- Vehicle counting and traffic flow analysis  
- Speed estimation using frame timestamps  
- Edge deployment for roadside cameras  
- Cloud-based traffic monitoring systems  

---

## Learning Outcomes

This project demonstrates:

- Real-time computer vision system design  
- Multi-class object detection implementation  
- Video analytics using OpenCV  
- Integration of deep learning models into practical applications  

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science 
