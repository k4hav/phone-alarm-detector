import cv2
import numpy as np
import os
import time

# ---------- CONFIG ----------

AUDIO_DURATION = 16  # seconds (audio length)
CONFIDENCE_THRESHOLD = 0.6
COOLDOWN = 2            # seconds
DETECT_EVERY_N_FRAMES = 1

# -------- YOLO LOAD --------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names") as f:
    classes = f.read().splitlines()

# -------- CAMERA --------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

alarm_on = False
last_time = 0
frame_count = 0

print("Camera started (FAST MODE). Press 'q' to quit.")

# -------- MAIN LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    phone_detected = False

    # 🔥 RUN YOLO ONLY EVERY N FRAMES
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        blob = cv2.dnn.blobFromImage(
            frame, 1/255, (320, 320), swapRB=True, crop=False
        )
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if classes[class_id] == "cell phone" and confidence > CONFIDENCE_THRESHOLD:
                    phone_detected = True
                    break
            if phone_detected:
                break

    now = time.time()

    # ---------- ALARM LOGIC ----------
    if phone_detected and not alarm_on:
        print("PHONE DETECTED → ALARM ON")
        os.system("paplay alarm.wav &")
        alarm_on = True
        last_time = now

    if alarm_on and not phone_detected and (now - last_time) > COOLDOWN:
        print("PHONE GONE → ALARM OFF")
        os.system("pkill paplay")
        alarm_on = False

    cv2.imshow("Phone Detector (Fast)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
os.system("pkill paplay")
