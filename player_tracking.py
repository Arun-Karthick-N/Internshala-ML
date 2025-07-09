import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT
from ultralytics.utils import IterableSimpleNamespace
import os

# Paths to input video and model
VIDEO_PATH = "15sec_input_720p.mp4"
MODEL_PATH = "player_model.pt"
OUTPUT_PATH = "output_tracked_video.mp4"

# Specify the class ID for players (based on debug output)
PLAYER_CLASS_ID = 2  # Players are class 2

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# Configure BOTSORT arguments for improved accuracy
tracker_args = IterableSimpleNamespace(
    track_high_thresh=0.6,  # Higher confidence for reliable detections
    track_low_thresh=0.1,   # Minimum confidence threshold
    new_track_thresh=0.7,   # Stricter threshold for new tracks
    match_thresh=0.7,       # Balanced for occlusion handling
    track_buffer=90,        # Increased to retain tracks longer
    frame_rate=30,          # Frame rate of the video
    fuse_score=True,        # Enable score fusion
    gmc_method="sparseOptFlow",  # Global Motion Compensation
    with_reid=False,        # Disabled to avoid ReID feature error
    proximity_thresh=0.5,   # Minimum IoU for matching
    appearance_thresh=0.25  # Minimum appearance similarity (unused without ReID)
)

# Initialize BOTSORT tracker
tracker = BOTSORT(tracker_args)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise Exception("Error: Could not open video file.")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# Counter to limit debug output
frame_count = 0
max_debug_frames = 10  # Extended for better monitoring

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection with confidence threshold
    results = model(frame, conf=0.6)

    # Debug output for detections
    if len(results[0].boxes) > 0 and frame_count < max_debug_frames:
        print(f"Frame {frame_count}:")
        print("Detected classes:", results[0].boxes.cls.cpu().numpy())
        print("Confidences:", results[0].boxes.conf.cpu().numpy())

    # Filter for players with high confidence
    detections = results[0].boxes[(results[0].boxes.cls == PLAYER_CLASS_ID) & 
                                 (results[0].boxes.conf > 0.6)]

    # Debug output for detections
    if frame_count < max_debug_frames:
        print(f"Number of player detections: {len(detections)}")
        frame_count += 1

    # Skip tracking if no detections
    if len(detections) == 0:
        out.write(frame)
        continue

    # Update tracker
    tracks = tracker.update(detections, frame)

    # Debug output for tracks
    if frame_count <= max_debug_frames:
        print(f"Number of tracks: {len(tracks)}")

    # Draw bounding boxes and IDs for all tracks
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track[:5])
        # Draw bounding box in green
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw ID above head in red
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(frame, str(track_id), (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {OUTPUT_PATH}")