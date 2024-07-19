import sys
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Add the path to the 'sort' directory to sys.path
sort_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'realtime-tracking', 'Downloads', 'build', 'sort')
sys.path.append(sort_path)

print(f"Adding {sort_path} to sys.path")
print("sys.path:", sys.path)

# Print contents of the sort directory
if os.path.exists(sort_path):
    print("Directory contents:", os.listdir(sort_path))
else:
    print("Path does not exist.")

try:
    from sort import Sort
    print("Successfully imported Sort module.")
except ImportError as e:
    print(f"Error importing Sort module: {e}")

# Load the YOLO model
try:
    model = YOLO('yolov8n.pt')  # Load model with ultralytics
    print("Successfully loaded YOLOv8 model.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    sys.exit()  # Exit if model loading fails

# Initialize SORT tracker
tracker = Sort()

# Open video capture (0 for webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(rgb_frame)
    
    # Extract bounding boxes and confidences
    detections = []
    for result in results[0]:  # Iterate over results (each result should be a tensor)
        # Assuming result format is [x1, y1, x2, y2, conf, cls]
        for *box, conf, cls in result:  # Unpack bounding box coordinates and other attributes
            x1, y1, x2, y2 = box
            detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])

    # Update tracker with detections
    tracked_objects = tracker.update(np.array(detections))

    # Draw bounding boxes and object IDs
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, conf = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Tracking', frame)

    # Exit if ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
