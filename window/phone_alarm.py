import cv2
import numpy as np
import time
import pygame

# ================= CONFIG =================
AUDIO_DURATION = 16  # seconds (audio length)
CONFIDENCE_THRESHOLD = 0
COOLDOWN = 2
DETECT_EVERY_N_FRAMES = 1

# ================= AUDIO =================
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# ================= YOLO LOAD =================
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
use_cuda = False

# ---- SAFE CUDA TEST ----
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 🔥 dummy forward to test CUDA REALLY works
    dummy = np.zeros((1, 3, 416, 416), dtype=np.float32)
    net.setInput(dummy)
    _ = net.forward(net.getUnconnectedOutLayersNames())

    use_cuda = True
    print("✅ CUDA enabled (GPU mode)")

except Exception as e:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("CUDA acceleration is not enabled in this OpenCV build.")
    print("Falling back to CPU execution.✅")


with open("coco.names") as f:
    classes = f.read().splitlines()

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

alarm_on = False
last_time = 0
frame_count = 0

print("Camera started (Windows). Press 'q' to quit.")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    phone_detected = False

    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (320, 320), swapRB=True, crop=False
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

    # ================= ALARM =================
    if phone_detected and not alarm_on:
        print("PHONE DETECTED → ALARM ON")
        pygame.mixer.music.play(-1)
        alarm_on = True
        last_time = now

    if alarm_on and not phone_detected and (now - last_time) > COOLDOWN:
        print("PHONE GONE → ALARM OFF")
        pygame.mixer.music.stop()
        alarm_on = False

    cv2.imshow("Phone Detector (Windows)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
