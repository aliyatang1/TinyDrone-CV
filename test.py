import cv2

cap = cv2.VideoCapture(0)  # Open the webcam
if not cap.isOpened():
    print("Error: Unable to access webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break
        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break
    cap.release()
    cv2.destroyAllWindows()

