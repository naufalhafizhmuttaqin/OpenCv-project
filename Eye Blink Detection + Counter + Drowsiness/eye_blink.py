import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam (DirectShow for Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye Aspect Ratio (EAR) parameters
EAR_THRESHOLD = 0.27          # Eye closed threshold
MIN_CLOSED_FRAMES = 2         # Minimum consecutive frames to count as blink

blink_count = 0
closed_frames = 0
eye_state = "OPEN"

# Eye landmark indices (MediaPipe Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(landmarks, eye_indices, w, h):
    """Compute Eye Aspect Ratio (EAR) from eye landmarks"""
    pts = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append((x, y))

    p1, p2, p3, p4, p5, p6 = pts
    v1 = np.linalg.norm(np.array(p2) - np.array(p6))
    v2 = np.linalg.norm(np.array(p3) - np.array(p5))
    h_dist = np.linalg.norm(np.array(p1) - np.array(p4))

    return (v1 + v2) / (2.0 * h_dist)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the camera feed
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_text = "OPEN"
    status_color = (0, 255, 0)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Calculate EAR for both eyes
        left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection logic
        if ear < EAR_THRESHOLD:
            closed_frames += 1
            status_text = "CLOSED"
            status_color = (0, 0, 255)

            if closed_frames >= MIN_CLOSED_FRAMES:
                eye_state = "CLOSED"
        else:
            if eye_state == "CLOSED":
                blink_count += 1
            closed_frames = 0
            eye_state = "OPEN"

        # Display EAR value
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Display blink counter
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)

        # Status overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 95), (260, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # Display eye status
        cv2.putText(frame, f"Status: {status_text}", (25, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)

    cv2.imshow("Eye Blink Detection and Counter", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
