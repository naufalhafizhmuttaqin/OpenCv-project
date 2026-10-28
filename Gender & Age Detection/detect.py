"""
detect.py
=========
Gender and Age Detection menggunakan OpenCV DNN.
Jalankan: python detect.py
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from model_loader import load_models

# ── Konfigurasi ───────────────────────────────────────────────
FACE_CONFIDENCE = 0.7
PADDING         = 20

GENDER_LIST = ["Male", "Female"]
AGE_LIST    = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
]

COLOR_MALE   = (255, 150,  50)
COLOR_FEMALE = (200,  80, 220)
COLOR_FPS    = (0,   255, 255)

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.7
THICKNESS  = 2


# ── Preprocessing ─────────────────────────────────────────────
def preprocess_face(face_img):
    return cv2.dnn.blobFromImage(
        face_img, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )

def preprocess_frame(frame):
    return cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False
    )


# ── Deteksi ───────────────────────────────────────────────────
def detect_faces(net, frame):
    h, w = frame.shape[:2]
    net.setInput(preprocess_frame(frame))
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > FACE_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2, y2, float(conf)))
    return faces


def predict_gender_age(gender_net, age_net, face_img):
    blob = preprocess_face(face_img)

    gender_net.setInput(blob)
    g_preds = gender_net.forward()
    g_idx   = g_preds[0].argmax()
    gender  = GENDER_LIST[g_idx]
    g_conf  = float(g_preds[0][g_idx])

    age_net.setInput(blob)
    a_preds = age_net.forward()
    a_idx   = a_preds[0].argmax()
    age     = AGE_LIST[a_idx]
    a_conf  = float(a_preds[0][a_idx])

    return gender, g_conf, age, a_conf


# ── Tampilan ──────────────────────────────────────────────────
def draw_results(frame, faces, predictions):
    for (x1, y1, x2, y2, _), (gender, g_conf, age, a_conf) in zip(faces, predictions):
        color = COLOR_MALE if gender == "Male" else COLOR_FEMALE
        label = f"{gender}  {age}"
        conf  = f"G:{g_conf:.0%}  A:{a_conf:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)

        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, conf, (x1, y2 + 20),
                    FONT, 0.5, color, 1, cv2.LINE_AA)
    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                FONT, 0.8, COLOR_FPS, 2, cv2.LINE_AA)
    return frame


def draw_no_face(frame):
    h, w = frame.shape[:2]
    msg = "No face detected"
    (tw, _), _ = cv2.getTextSize(msg, FONT, 0.7, 1)
    cv2.putText(frame, msg, ((w - tw) // 2, h - 20),
                FONT, 0.7, (80, 80, 200), 1, cv2.LINE_AA)
    return frame


# ── Pipeline ──────────────────────────────────────────────────
def process_frame(nets, frame):
    face_net, gender_net, age_net = nets
    faces = detect_faces(face_net, frame)
    predictions = []

    for (x1, y1, x2, y2, _) in faces:
        y1p = max(0, y1 - PADDING)
        y2p = min(frame.shape[0], y2 + PADDING)
        x1p = max(0, x1 - PADDING)
        x2p = min(frame.shape[1], x2 + PADDING)
        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue
        predictions.append(predict_gender_age(gender_net, age_net, crop))

    output = draw_results(frame.copy(), faces[:len(predictions)], predictions)
    if not faces:
        output = draw_no_face(output)
    return output, len(faces)


# ── Mode Webcam ───────────────────────────────────────────────
def run_camera(nets, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka kamera id={camera_id}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[INFO] Webcam aktif.")
    print("[INFO] Tekan  q  untuk keluar,  s  untuk screenshot.\n")

    prev_time = time.time()
    sc_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Gagal membaca frame.")
            break

        frame = cv2.flip(frame, 1)

        output, _ = process_frame(nets, frame)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        output = draw_fps(output, fps)

        cv2.imshow("Gender & Age Detection  |  q=quit  s=save", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{sc_count:03d}.jpg"
            cv2.imwrite(fname, output)
            print(f"[INFO] Screenshot disimpan: {fname}")
            sc_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program selesai.")


# ── Mode Gambar ───────────────────────────────────────────────
def run_image(nets, image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    output, n = process_frame(nets, frame)
    print(f"[INFO] Wajah terdeteksi: {n}")

    cv2.imshow(f"Hasil - {Path(image_path).name}  |  tekan sembarang tombol", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out_path = f"result_{Path(image_path).stem}.jpg"
    cv2.imwrite(out_path, output)
    print(f"[INFO] Hasil disimpan: {out_path}")


# ── Mode Video ────────────────────────────────────────────────
def run_video(nets, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = f"result_{Path(video_path).stem}.mp4"
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    print(f"[INFO] Memproses: {video_path}")
    print("[INFO] Tekan  q  untuk berhenti.\n")

    prev_time   = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, _ = process_frame(nets, frame)
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        output = draw_fps(output, fps)

        writer.write(output)
        cv2.imshow("Video Detection  |  q=quit", output)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {frame_count} frame diproses. Disimpan: {out_path}")


# ── Main ──────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Gender and Age Detection")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image",  "-i", type=str, help="Path ke file gambar")
    group.add_argument("--video",  "-v", type=str, help="Path ke file video")
    group.add_argument("--camera", "-c", type=int, default=0, help="ID kamera (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 55)
    print("  Gender and Age Detection  —  OpenCV DNN")
    print("=" * 55)

    nets = load_models()

    if args.image:
        run_image(nets, args.image)
    elif args.video:
        run_video(nets, args.video)
    else:
        run_camera(nets, args.camera)


if __name__ == "__main__":
    main()