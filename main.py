import os
import cv2
import torch
import numpy as np
from utils.detect import detect_players
from utils.features import extract_features
from utils.matcher import match_players
from tqdm import tqdm

BROADCAST_VIDEO = "broadcast.mp4"
TACTICAM_VIDEO = "tacticam.mp4"
MODEL_PATH = "best.pt"

print("\n[INFO] Detecting players in broadcast video...")
broadcast_dets = detect_players(BROADCAST_VIDEO, MODEL_PATH)

print("[INFO] Detecting players in tacticam video...")
tacticam_dets = detect_players(TACTICAM_VIDEO, MODEL_PATH)

def crop_and_extract(video_path, detections):
    cap = cv2.VideoCapture(video_path)
    frame_cache = {}
    features = []

    for det in tqdm(detections):
        frame_idx = det["frame"]
        bbox = det["bbox"]

        if frame_idx not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_cache[frame_idx] = frame
        else:
            frame = frame_cache[frame_idx]

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        feat = extract_features(crop)
        features.append(feat)

    cap.release()
    return np.array(features)

print("\n[INFO] Extracting features from broadcast video...")
broadcast_feats = crop_and_extract(BROADCAST_VIDEO, broadcast_dets)

print("[INFO] Extracting features from tacticam video...")
tacticam_feats = crop_and_extract(TACTICAM_VIDEO, tacticam_dets)

print("\n[INFO] Matching players...")
matches = match_players(broadcast_feats, tacticam_feats)

print("\n[INFO] Matched Player IDs:")
for tid, bid in matches:
    print(f"Tacticam player {tid} -> Broadcast player {bid}")


print("[INFO] Extracting features from tacticam video...")
tacticam_feats = crop_and_extract(TACTICAM_VIDEO, tacticam_dets)

print("\n[INFO] Matching players...")
matches = match_players(broadcast_feats, tacticam_feats)

print("\n[INFO] Matched Player IDs:")
for tid, bid in matches:
    print(f"Tacticam player {tid} -> Broadcast player {bid}")
