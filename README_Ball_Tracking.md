# Real-Time Basket Ball Tracking using YOLO and BoT-SORT

## Overview

This repository implements a real-time **ball detection and tracking system** using deep learning and multi-object tracking. The system detects a sports ball in each frame, assigns a persistent track ID, and visualizes its movement using a short fading motion trail.

Unlike simple frame-by-frame detection, this project maintains **temporal consistency** across frames using YOLO’s tracking mode with **BoT-SORT**. The system continues tracking even during short detection gaps and provides smooth motion visualization.

This project demonstrates:
- Deep learning–based object detection (YOLO)
- Multi-object tracking using BoT-SORT
- Real-time video processing
- Temporal motion analysis
- Deque-based fading trajectory visualization

Applications include sports analytics, motion analysis, surveillance, robotics, and intelligent video systems.

---

## Project Flow

The system follows a real-time tracking pipeline:

1. Import required libraries  
2. Load YOLO detection model  
3. Configure BoT-SORT tracker  
4. Read video frames sequentially  
5. Detect the ball in each frame  
6. Maintain persistent track IDs  
7. Store recent position history  
8. Render fading motion trail  
9. Display and save output  

```
Video → YOLO Detection → BoT-SORT Tracking → Position History → Fading Trail Visualization
```

---

## Installation

Install required dependencies:

```
pip install ultralytics opencv-python numpy matplotlib
```

---

## Project Structure

```
ball_tracking.ipynb
yolo26s.pt
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
from collections import defaultdict, deque
```

### Explanation

- **cv2** – Video processing and drawing operations  
- **numpy** – Numerical and coordinate handling  
- **YOLO** – Deep learning object detection and tracking  
- **deque** – Stores recent object positions for trajectory rendering  

---

## Load Detection Model

```python
model = YOLO("yolo26s.pt")
BALL_CLASS_ID = 32   # COCO class for sports ball
```

### Explanation

The YOLO model is loaded with pretrained weights. The system filters detections to track only the sports ball class.

---

## Tracking with BoT-SORT

```python
results = model.track(
    frame,
    conf=0.15,
    classes=[BALL_CLASS_ID],
    tracker=CUSTOM_TRACKER,
    persist=True
)
```

### Explanation

- **BoT-SORT** maintains object identity across frames  
- **persist=True** keeps track IDs consistent  
- Custom tracker settings improve robustness during short detection gaps  

This enables true object tracking rather than independent frame detection.

---

## Position History Buffer

Each tracked object maintains a short motion history.

```python
track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
track_history[track_id].append((cx, cy))
```

### Explanation

- Stores recent center positions of the ball  
- Limits history length to keep the trail short and efficient  
- Enables smooth temporal visualization  

---

## Motion Trail Rendering (Exact Method Used)

The notebook does **not draw a single trajectory line**.  
Instead, it renders multiple short line segments between consecutive positions with a fading effect.

```python
trail = track_history[track_id]

for i in range(1, len(trail)):
    alpha = i / len(trail)
    faded_color = (
        int(TRAIL_COLOR[0] * alpha),
        int(TRAIL_COLOR[1] * alpha),
        int(TRAIL_COLOR[2] * alpha),
    )

    cv2.line(
        frame,
        trail[i - 1],
        trail[i],
        faded_color,
        1,
        cv2.LINE_AA
    )
```

### Explanation

- Consecutive points from the deque are connected  
- Older segments gradually fade  
- Newer segments appear brighter  
- Thin anti-aliased lines ensure smooth visualization  

This produces a **dynamic fading motion trail**, not a single static trajectory.

---

## Handling Detection Gaps

When the ball is temporarily not detected:

- The last known track ID is retained  
- A fading “ghost trail” is rendered for a few frames  
- A ghost circle marks the last known position  

This improves robustness during occlusion or fast motion.

---

## Bounding Box and Center Visualization

```python
cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)
cv2.circle(frame, (cx, cy), 4, BALL_COLOR, -1)
```

### Explanation

- Bounding box shows the detected object  
- Center point represents the tracked position  
- Used to update motion history  

---

## Display and Save Output

```python
cv2.imshow("Ball Tracking", frame)
```

Cleanup:

```python
cap.release()
writer.release()
cv2.destroyAllWindows()
```

### Explanation

Frames are displayed in real time and saved as a processed video. Proper resource cleanup ensures stable execution for long-duration processing.

---

## Detection and Tracking Pipeline (Detailed)

Each frame follows this lifecycle:

1. Frame Capture  
2. YOLO Detection  
3. Track ID Assignment (BoT-SORT)  
4. Position Extraction  
5. History Update (deque)  
6. Fading Trail Rendering  
7. Visualization  

This loop runs continuously for real-time performance.

---

## Practical Applications

- Sports analytics (trajectory and performance analysis)
- Ball movement pattern analysis
- Surveillance and motion tracking
- Robotics object following
- Intelligent video analytics

---

## Limitations

- Tracks a single object class  
- Performance depends on video quality  
- May lose tracking during long occlusions  
- No speed or physics-based trajectory analysis  

---

## Future Improvements

- Kalman Filter for smoother motion prediction  
- Speed and acceleration estimation  
- Multi-object tracking  
- Integration with sports analytics dashboard  
- Edge deployment for real-time systems  

---

## Learning Outcomes

This project demonstrates:

- Real-time tracking using YOLO + BoT-SORT  
- Temporal data association  
- Deque-based motion history management  
- Fading trajectory visualization  
- End-to-end video analytics pipeline design  

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science 
