import cv2
import numpy as np
import os
import time

# ---------- CONFIG ----------
AUDIO_DURATION = 16  # seconds (tera audio length)
CONFIDENCE_THRESHOLD = 0.6

# ---------- LOAD YOLO ----------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names") as f:
    classes = f.read().splitlines()

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)

alarm_active = False
alarm_start_time = 0

print("Camera started. Press 'q' to quit.")

# ---------- MAIN LOOP ----------
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

    # ---------- ALARM LOGIC (LATCHED) ----------
    if phone_detected and not alarm_active:
        print("PHONE DETECTED → PLAY FULL ALARM")
        os.system("paplay alarm.wav &")
        alarm_active = True
        alarm_start_time = now

    # allow alarm to finish fully
    if alarm_active and (now - alarm_start_time) >= AUDIO_DURATION:
        print("ALARM FINISHED → READY FOR NEXT TRIGGER")
        alarm_active = False

    cv2.imshow("Phone Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------- CLEANUP ----------
cap.release()
cv2.destroyAllWindows()
