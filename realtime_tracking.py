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
try:
    tracker = Sort()
    print("Successfully initialized SORT tracker.")
except Exception as e:
    print(f"Error initializing SORT tracker: {e}")
    sys.exit()

# Open video capture (0 for webcam)
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Unable to access webcam.")
    print("Webcam opened successfully.")
except Exception as e:
    print(f"Error opening video capture: {e}")
    sys.exit()

# Setup Tkinter window
root = tk.Tk()
root.title("Real-Time Object Detection")

# Create a label to display the video feed
label = tk.Label(root)
label.pack()

def update_frame():
    try:
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Unable to read frame from webcam.")
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model(rgb_frame)

        # Extract bounding boxes and confidences
        detections = []
        if hasattr(results, 'boxes'):
            boxes = results.boxes
            if boxes is not None:
                for box in boxes:
                    # The format of box: [x1, y1, x2, y2, conf, cls]
                    try:
                        x1, y1, x2, y2, conf, cls = box.tolist()
                        detections.append([x1, y1, x2, y2, conf])
                    except ValueError as e:
                        print(f"Error unpacking detection result: {e}")
                        continue
        else:
            print("No boxes found in the detection results.")

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

    except Exception as e:
        print(f"Error processing frame: {e}")
        root.quit()
        return

    # Call this function again after 10 ms
    label.after(10, update_frame)

def on_closing():
    print("Closing application...")
    cap.release()  # Release the webcam
    print("Webcam released.")
    root.destroy()  # Close the Tkinter window

# Set the function to be called when the Tkinter window is closed
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the update loop
update_frame()

# Run the Tkinter event loop
try:
    root.mainloop()
except Exception as e:
    print(f"Error running Tkinter event loop: {e}")
finally:
    # Ensure the webcam is released
    cap.release()
    print("Webcam released.")
