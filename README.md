# Football Player Tracking with YOLO and OSNet

This project tracks football players, referees, and the ball in a video, assigning consistent IDs to players even when they leave and re-enter the frame. It uses YOLO for detection, ByteTrack for tracking, and OSNet for player re-identification (ReID).

---

## Table of Contents
- [Features](#features)
- [YOLOv11l Model Summary](#yolov11l-model-summary)
- [Setup](#setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## Features
- Detects players, referees, and ball in football videos
- Assigns unique, consistent IDs to each player (even after occlusion or re-entry)
- Enhanced ReID: Combines color histogram (jersey color) and OSNet deep features for more robust player identity matching
- Draws annotations on output video
- Fast and robust, works on CPU or GPU

---

## YOLOv11l Model Summary

- **Model:** YOLOv11l (fused)
- **Layers:** 190
- **Parameters:** 25,282,396
- **GFLOPs:** 86.6

### Detection Performance (Validation Set)
| Class        | Images | Instances | Box(P) | Box(R) | mAP50 | mAP50-95 |
|--------------|--------|-----------|--------|--------|-------|----------|
| **all**      | 43     | 1025      | 0.885  | 0.756  | 0.821 | 0.566    |
| **ball**     | 39     | 39        | 0.808  | 0.333  | 0.432 | 0.176    |
| **goalkeeper** | 32   | 32        | 0.898  | 0.812  | 0.924 | 0.678    |
| **player**   | 43     | 853       | 0.936  | 0.955  | 0.982 | 0.781    |
| **referee**  | 43     | 101       | 0.896  | 0.921  | 0.948 | 0.629    |

- **Box(P):** Precision for bounding box
- **Box(R):** Recall for bounding box
- **mAP50:** Mean Average Precision at IoU 0.5
- **mAP50-95:** Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95

---

## Setup

### 1. Clone the Repository
```bash
git clone <https://github.com/Zubair-10-eng/Football-Detection-with-ReIDentification.git>
cd Football Assignment
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r require.txt
```

#### Main dependencies:
- `ultralytics` (YOLO)
- `torch` and `torchvision`
- `opencv-python`
- `supervision`
- `torchreid` (for OSNet ReID)

If you need to install torchreid manually:
```bash
pip install torchreid
```

### 4. Download/Prepare Models
- Place your YOLO model (e.g., `last.pt`) in the `custom_model/` directory.
- OSNet weights are downloaded automatically by torchreid on first use.

### 5. Prepare Input Video
- Place your input video (e.g., `15sec_input_720p.mp4`) in the `Input videos/` directory.

---

## Usage

### 1. Run the Main Script
```