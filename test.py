import cv2

# Open video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    cv2.imshow('Webcam Feed', frame)

    # Exit if ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
