import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from collections import deque

# ---------- CONFIG ----------
LOG_FILE = "emotion_log.csv"
SMOOTHING_WINDOW = 12

COLORS = {
    "Happy": (0, 255, 180),
    "Surprised": (255, 255, 0),
    "Sad": (255, 100, 100),
    "Neutral": (180, 180, 180)
}

# ---------- MEDIAPIPE ----------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

emotion_history = deque(maxlen=SMOOTHING_WINDOW)
emotion = "Neutral"
persona = "Calm"
confidence = 0
last_infer_time = 0

fullscreen = False
prev_time = time.time()

# ---------- CSV LOGGER ----------
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "emotion", "persona", "confidence"])


def infer_emotion(landmarks):
    left_eye = abs(landmarks[159].y - landmarks[145].y)
    right_eye = abs(landmarks[386].y - landmarks[374].y)
    eye_open = (left_eye + right_eye) / 2
    mouth_open = abs(landmarks[13].y - landmarks[14].y)

    if mouth_open > 0.04 and eye_open > 0.03:
        return "Surprised", "Curious", 92
    elif mouth_open > 0.03:
        return "Happy", "Confident", 88
    elif eye_open < 0.015:
        return "Sad", "Reflective", 75
    else:
        return "Neutral", "Calm", 60


def smooth_emotion(new_emotion):
    emotion_history.append(new_emotion)
    return max(set(emotion_history), key=emotion_history.count)


def draw_glow_box(frame, x, y, w, h, color):
    for i in range(8, 0, -2):
        cv2.rectangle(frame, (x-i, y-i), (x+w+i, y+h+i),
                      color, 1)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            xs = [int(lm.x * w) for lm in face.landmark]
            ys = [int(lm.y * h) for lm in face.landmark]
            x, y = min(xs), min(ys)
            bw, bh = max(xs) - x, max(ys) - y

            if time.time() - last_infer_time > 0.4:
                emo, pers, conf = infer_emotion(face.landmark)
                emotion = smooth_emotion(emo)
                persona = pers
                confidence = conf
                last_infer_time = time.time()

                with open(LOG_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.time(), emotion, persona, confidence])

            draw_glow_box(frame, x, y, bw, bh, COLORS[emotion])
            cv2.rectangle(frame, (x, y), (x+bw, y+bh),
                          COLORS[emotion], 2)

    # ---------- FPS ----------
    current_time = time.time()
    fps = int(1 / (current_time - prev_time))
    prev_time = current_time

    # ---------- HUD ----------
    cv2.rectangle(frame, (0, 0), (w, 70), (10, 10, 10), -1)
    cv2.putText(frame, "AI FACE EMOTION SYSTEM",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                COLORS[emotion], 3)

    cv2.putText(frame, f"Emotion: {emotion}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                COLORS[emotion], 2)

    cv2.putText(frame, f"Persona: {persona}",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                COLORS[emotion], 2)

    # Confidence bar
    bar_x, bar_y = 20, 185
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + 320, bar_y + 22), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(3.2 * confidence), bar_y + 22),
                  COLORS[emotion], -1)

    cv2.putText(frame, f"Confidence: {confidence}%",
                (bar_x, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1)

    cv2.putText(frame, f"FPS: {fps}",
                (w - 120, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 200), 2)

    cv2.putText(frame, "Q: Quit | F: Fullscreen",
                (w - 320, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1)

    cv2.imshow("Emotion HUD", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("f"):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("Emotion HUD",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Emotion HUD",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
