import cv2
import numpy as np
import pygame
import time

# ---------- CONFIG ----------
AUDIO_DURATION = 16  # seconds
CONFIDENCE_THRESHOLD = 0.6

# ---------- AUDIO ----------
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# ---------- YOLO ----------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names") as f:
    classes = f.read().splitlines()

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)

alarm_active = False
alarm_start = 0

print("Camera started. Press 'q' to quit.")

# ---------- LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(
        frame, 1/255, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    phone_detected = False

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD and classes[class_id] == "cell phone":
                phone_detected = True
                break

    now = time.time()

    if phone_detected and not alarm_active:
        print("PHONE DETECTED → ALARM PLAY")
        pygame.mixer.music.play()
        alarm_active = True
        alarm_start = now

    if alarm_active and (now - alarm_start) >= AUDIO_DURATION:
        alarm_active = False

    cv2.imshow("Phone Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
