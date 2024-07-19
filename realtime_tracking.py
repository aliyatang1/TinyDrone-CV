import sys
import os
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
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

def update_frame():
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
    for result in results[0]:
        for *box, conf, cls in result:
            x1, y1, x2, y2 = box
            detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])

    # Update tracker with detections
    tracked_objects = tracker.update(np.array(detections))

    # Draw bounding boxes and object IDs
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, conf = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
