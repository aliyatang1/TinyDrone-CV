import sys
import os
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
from sort import Sort
from scipy.spatial.distance import euclidean

# Define the correct path to the 'sort' directory
sort_path = os.path.join('os.path.expanduser('~'), 'Downloads', 'realtime-tracking', 'Downloads', 'build', 'sort'')
sys.path.append(sort_path)

print(f"Adding {sort_path} to sys.path")
print("sys.path:", sys.path)

# Print contents of the sort directory
if os.path.exists(sort_path):
    print("Directory contents:", os.listdir(sort_path))
else:
    print("Path does not exist.")
    sys.exit()

try:
    from sort import Sort
    print("Successfully imported Sort module.")
except ImportError as e:
    print(f"Error importing Sort module: {e}")
    sys.exit()

# Load the YOLO model
try:
    model = YOLO('yolov5su.pt')  # or YOLO('yolov8n.pt')
    print("Successfully loaded YOLO model.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit()

# Initialize SORT tracker
tracker = Sort()

# Open video capture (0 for webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access webcam.")
    sys.exit()

# Setup Tkinter window
root = tk.Tk()
root.title("Real-Time Object Detection")

# Create a label to display the video feed
label = tk.Label(root)
label.pack()

# Variables to store the last frame and its detected objects
last_frame = None
last_detections = None

# depth_estimation.py

def update_frame():
    global last_frame, last_detections

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from webcam.")
        root.quit()
        return

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(rgb_frame)

    # Extract bounding boxes and confidences
    detections = []
    for result in results:
        for detection in result.boxes.data.tolist():
            if len(detection) >= 5:
                x1, y1, x2, y2, conf = detection[:5]
                detections.append([x1, y1, x2, y2, conf])
            else:
                print(f"Unexpected detection format: {detection}")

    # Update tracker with detections
    tracked_objects = tracker.update(np.array(detections))

    print(f"Tracked objects: {tracked_objects}")

    if last_frame is not None and last_detections is not None:
        # Compute distance between same objects in consecutive frames
        for obj in tracked_objects:
            if len(obj) == 6:
                x1, y1, x2, y2, obj_id, conf = obj
            elif len(obj) == 5:
                x1, y1, x2, y2, obj_id = obj
            else:
                print(f"Unexpected tracked object format: {obj}")
                continue

            # Find the corresponding object in the last frame by ID
            for last_obj in last_detections:
                if len(last_obj) == 6:
                    last_x1, last_y1, last_x2, last_y2, last_id, _ = last_obj
                elif len(last_obj) == 5:
                    last_x1, last_y1, last_x2, last_y2, last_id = last_obj
                else:
                    print(f"Unexpected last tracked object format: {last_obj}")
                    continue

                if int(last_id) == int(obj_id):
                    # Compute centroids of the bounding boxes
                    current_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                    last_centroid = ((last_x1 + last_x2) / 2, (last_y1 + last_y2) / 2)

                    # Compute Euclidean distance between centroids
                    distance = euclidean(current_centroid, last_centroid)
                    print(f"Distance between object ID {int(obj_id)}: {distance:.2f} pixels")

    # Store the current frame and detections
    last_frame = frame.copy()
    last_detections = tracked_objects

    # Draw bounding boxes, object IDs, and coordinates
    for obj in tracked_objects:
        if len(obj) == 6:
            x1, y1, x2, y2, obj_id, conf = obj
        elif len(obj) == 5:
            x1, y1, x2, y2, obj_id = obj
        else:
            continue

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # Draw object ID
        cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # Draw coordinates
        cv2.putText(frame, f'({int(x1)}, {int(y1)})', (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'({int(x2)}, {int(y2)})', (int(x2) - 150, int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert OpenCV frame to ImageTk format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update label with new image
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Call this function again after 10 ms
    label.after(10, update_frame)

# Start the update loop
update_frame()

# Run the Tkinter event loop
root.mainloop()

# Release webcam when done
cap.release()