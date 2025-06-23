# Player Re-Identification in Sports Footage

This project performs **cross-camera player re-identification** using deep learning-based object detection and visual similarity matching. The goal is to ensure that the same player retains the same ID across two different video feeds: `broadcast.mp4` and `tacticam.mp4`.

## Project Structure

reid_project/
├── best.pt                  # Fine-tuned YOLOv11 model for player/ball detection
├── broadcast.mp4            # Broadcast angle video
├── tacticam.mp4             # Tactical camera angle video
├── main.py                  # Main pipeline to run player re-identification
├── utils/
│   ├── __init__.py
│   ├── detect.py            # Player detection using YOLO
│   ├── features.py          # Appearance feature extraction using ResNet50
│   ├── matcher.py           # Player matching logic (Hungarian algorithm)
│   └── tracker.py           # (Optional) For further tracking enhancements
└── README.md

## Setup Instructions

### 1. Clone the Repo or Download the Files
Ensure the folder structure matches the layout above.

### 2. Create and Activate a Virtual Environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # On Linux/macOS
.venv\Scripts\activate         # On Windows
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt

If you don’t have a `requirements.txt`, install manually:

```bash
pip install ultralytics opencv-python torch torchvision scipy tqdm


## How to Run
1. Make sure you have placed `broadcast.mp4`, `tacticam.mp4`, and `best.pt` in the root project folder.
2. Then run:
```bash
python main.py

You will see logs like:
[INFO] Detecting players in broadcast video...
[INFO] Detecting players in tacticam video...
[INFO] Extracting visual features...
[INFO] Matching players across views...
Tacticam player 0 -> Broadcast player 2
Tacticam player 1 -> Broadcast player 0

## Output
* Player IDs are matched across camera angles.

## Requirements
* Python 3.8–3.11
* PyTorch
* Ultralytics (for YOLO)
* OpenCV
* Torchvision
* SciPy
* tqdm

## Optional Extensions
* Use ByteTrack or DeepSORT for multi-object tracking
* Add ball detection and possession estimation
* Train on your own dataset using Ultralytics format

## Author
Built by Bhavya Agarwal for computer vision-based sports analytics.

## License
MIT or as per institution/company guidelines.
