import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam initialization failed.")
else:
    print("Webcam works!")
    cap.release()