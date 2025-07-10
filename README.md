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
```bash
python main_file.py
```

- The script will process the input video, track all players, and save the annotated output to `output_videos/output.avi`.

### 2. Output
- The output video will show bounding boxes and unique IDs for each player, referee, and the ball.

---

## How It Works

This project tracks football players, referees, and the ball in a video, assigning consistent IDs to players even when they leave and re-enter the frame. It uses YOLO for detection, ByteTrack for tracking, and OSNet for player re-identification (ReID).

### Detection (YOLO)
- YOLO detects all players, referees, and the ball in each frame.

### Tracking (ByteTrack)
- ByteTrack links detections across frames, assigning temporary IDs.

### Re-Identification (OSNet + Color Histogram)
- OSNet extracts appearance features for each player.
- Color histograms (based on jersey color) are computed for each player crop.
- When a player leaves and re-enters, their color histogram is compared to those of previously tracked players as a fast pre-filter.
- If the color histogram is similar, OSNet features are then compared for fine-grained matching.
- If both are similar, the same ID is assigned; otherwise, a new ID is given.
- This approach improves ID consistency, especially for players on different teams or with similar builds.

### Annotation
- The system draws ellipses for players, rectangles for IDs, and triangles for the ball.

---

## Project Structure
```
Football Assignment/
├── main_file.py           # Main script to run tracking
├── require.txt            # Python dependencies
├── custom_model/          # YOLO model weights
├── Input videos/          # Input football videos
├── output_videos/         # Output annotated videos
├── trackers/
│   └── trackers.py        # Tracking and ReID logic
├── utils/
│   └── ...                # Utility functions (video I/O, etc.)
└── README.md              # This file
```

---

## Troubleshooting
- **No detections?**
  - Check your YOLO model path and input video format.
- **OSNet not working?**
  - Make sure `torchreid` is installed and you have a supported version of PyTorch.
- **Slow processing?**
  - Try running on a machine with a GPU, or use a smaller YOLO model.
- **Output video not saving?**
  - Check the `output_videos/` directory exists and you have write permissions.

---

## Credits
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OSNet / torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- [Supervision](https://github.com/roboflow/supervision) 