import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load models
player_model = YOLO("playerIdentificationModel.pt")
shot_model = load_model("final_camera_model_finetuned.keras")
label_map = ['closeup', 'long', 'medium']  # match your model output
scaling_factors = {
    "closeup":  0.595,
    "medium": 1.1,
    "long": 1.8
}

# Predict shot type
def predict_shot_type(frame, img_size=224):
    img = cv2.resize(frame, (img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = shot_model.predict(img, verbose=0)[0]
    return label_map[np.argmax(pred)]

# Draw boxes and estimate distance with revised logic
def annotate_frame(frame):
    results = player_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) != 2:
        return frame  # Skip frame if not exactly 2 players

    centroids = []
    heights = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centroids.append((cx, cy))
        heights.append(y2 - y1)

        # Draw player bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Step 1: Calculate pixel distance between centroids
    d_pixels = np.linalg.norm(np.array(centroids[0]) - np.array(centroids[1]))

    # Step 2: Calculate average bounding box height
    avg_height = np.mean(heights)

    # Step 3: Predict camera shot type
    shot_type = predict_shot_type(frame)

    # Step 4: Get scaling factor for the shot type
    scale = scaling_factors.get(shot_type, 0.15)  # default scale if missing

    # Step 5: Convert pixel distance to real-world distance (feet)
    KNOWN_PLAYER_HEIGHT_FEET = 5.83   # Adjust if needed

    real_distance = (d_pixels / avg_height) * KNOWN_PLAYER_HEIGHT_FEET * scale
    real_distance = round(real_distance, 2)

    # Draw line and distance text on the frame
    cv2.line(frame, centroids[0], centroids[1], (255, 0, 0), 2)
    mid_point = ((centroids[0][0] + centroids[1][0]) // 2,
                 (centroids[0][1] + centroids[1][1]) // 2)
    cv2.putText(frame, f"{real_distance} ft ({shot_type})", mid_point,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame

# 🎞️ Analyze a full video
def analyze_video_with_distance(input_path, output_path="output_with_distance.mp4"):
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Processing {frame_count} frames...")

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        annotated_frame = annotate_frame(frame)
        out.write(annotated_frame)

        if frame_num % 20 == 0:
            print(f"✅ Processed {frame_num}/{frame_count} frames")

    cap.release()
    out.release()
    print(f"🎉 Finished processing. Saved to {output_path}")

# Example usage:
analyze_video_with_distance("test_video.mp4", "op3333.mp4")
