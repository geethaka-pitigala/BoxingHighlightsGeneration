import cv2
import json
import numpy as np
from ultralytics import YOLO
from collections import deque

# Global configs
POSE_NOSE_IDX = 0
POSE_ANKLE_IDX = [15, 16]
KNOCKDOWN_FRAME_THRESHOLD = 3
KNOCKDOWN_PRE_SECONDS = 3
KNOCKDOWN_POST_SECONDS = 3
MAX_HISTORY = 5

# Load models
pose_model = YOLO("yolov8n-pose.pt")
player_model = YOLO("playerIdentificationModel.pt")

def is_knockdown_pose(keypoints):
    try:
        nose_y = keypoints[POSE_NOSE_IDX][1]
        ankle_y = max(keypoints[i][1] for i in POSE_ANKLE_IDX)
        return abs(nose_y - ankle_y) < 50
    except:
        return False

def is_knockdown_fallback(bbox, history):
    x, y, w, h = bbox
    aspect_ratio = h / (w + 1e-5)
    if aspect_ratio < 0.5 and history:
        prev_y = history[-1]
        if y - prev_y > 20:
            return True
    return False

def frame_to_timestamp(frame_idx, fps):
    seconds = int(frame_idx / fps)
    return f"{seconds//3600:02}:{(seconds%3600)//60:02}:{seconds%60:02}"

def merge_ranges(ranges):
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for current in ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # overlapping or touching
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def extract_knockdown_durations(video_path, output_json="knockdown_ranges.json"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    knockdown_buffer = {}
    player_history = {}
    knockdown_frames = set()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = player_model(frame)
        players = results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(players):
            x1, y1, x2, y2 = box[:4]
            w = int(x2 - x1)
            h = int(y2 - y1)
            player_id = f"player_{i}"

            # Track player Y position history
            player_history.setdefault(player_id, deque(maxlen=MAX_HISTORY)).append(y1)

            # Pose detection
            player_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            pose_result = pose_model(player_crop)
            keypoints = pose_result[0].keypoints.xy.cpu().numpy()[0] if pose_result[0].keypoints else []

            knockdown = False
            if len(keypoints) > 0:
                knockdown = is_knockdown_pose(keypoints)

            if not knockdown:
                knockdown = is_knockdown_fallback((x1, y1, w, h), list(player_history[player_id]))

            # Temporal logic
            if knockdown:
                knockdown_buffer[player_id] = knockdown_buffer.get(player_id, 0) + 1
            else:
                knockdown_buffer[player_id] = 0

            # Confirm knockdown
            if knockdown_buffer[player_id] >= KNOCKDOWN_FRAME_THRESHOLD:
                knockdown_frames.add(frame_idx)

        frame_idx += 1

    cap.release()

    # Convert knockdown frames to time ranges
    time_ranges = []
    frame_margin = int(fps * KNOCKDOWN_PRE_SECONDS)
    post_margin = int(fps * KNOCKDOWN_POST_SECONDS)

    for f in sorted(knockdown_frames):
        start = max(0, f - frame_margin)
        end = min(total_frames - 1, f + post_margin)
        time_ranges.append((start, end))

    # Merge overlapping ranges
    merged_ranges = merge_ranges(time_ranges)

    # Convert to readable format
    knockdown_json = []
    for start_f, end_f in merged_ranges:
        knockdown_json.append({
            "start": frame_to_timestamp(start_f, fps),
            "end": frame_to_timestamp(end_f, fps)
        })

    # Save as JSON
    with open(output_json, "w") as f:
        json.dump(knockdown_json, f, indent=4)

    print("Knockdown time ranges saved to", output_json)
    return knockdown_json

video_path = input("enter video path here :")
knockdown_ranges = extract_knockdown_durations(video_path)
print(knockdown_ranges)