"""
model_loader.py
===============
Mengunduh dan memuat model DNN untuk:
  - Face Detection  (OpenCV Caffe)
  - Gender Detection
  - Age Detection
"""

import urllib.request
from pathlib import Path
import cv2

MODEL_DIR = Path(__file__).parent / "models"

MODELS = {
    "face_proto": {
        "path": MODEL_DIR / "deploy.prototxt",
        "url":  "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    },
    "face_model": {
        "path": MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel",
        "url":  "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    },
    "gender_proto": {
        "path": MODEL_DIR / "gender_deploy.prototxt",
        "url":  "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt"
    },
    "gender_model": {
        "path": MODEL_DIR / "gender_net.caffemodel",
        "url":  "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel"
    },
    "age_proto": {
        "path": MODEL_DIR / "age_deploy.prototxt",
        "url":  "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt"
    },
    "age_model": {
        "path": MODEL_DIR / "age_net.caffemodel",
        "url":  "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel"
    },
}

MIN_FILE_SIZE = {
    "face_proto":   1_000,
    "face_model":   10_000_000,
    "gender_proto": 1_000,
    "gender_model": 20_000_000,
    "age_proto":    1_000,
    "age_model":    20_000_000,
}


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct // 2)
        print(f"\r  [{'█' * bar}{'░' * (50 - bar)}] {pct:5.1f}%", end="", flush=True)


def _download(name, url, dest):
    print(f"\n[DL] {name}")
    print(f"     {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"\n[ERROR] Gagal mengunduh {name}.\n"
            f"  URL  : {url}\n"
            f"  Error: {exc}\n"
        ) from exc


def ensure_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    any_downloaded = False

    for name, info in MODELS.items():
        path = info["path"]
        min_size = MIN_FILE_SIZE.get(name, 0)
        needs_download = (not path.exists() or path.stat().st_size < min_size)

        if needs_download:
            if not any_downloaded:
                print("\n[INFO] Mengunduh model yang diperlukan ...")
            _download(name, info["url"], path)
            any_downloaded = True

    if any_downloaded:
        print("\n[INFO] Semua model berhasil diunduh.\n")
    else:
        print("[INFO] Semua model sudah tersedia.\n")


def load_models():
    ensure_models()
    print("[INFO] Memuat model ke memori ...")

    face_net = cv2.dnn.readNet(
        str(MODELS["face_model"]["path"]),
        str(MODELS["face_proto"]["path"])
    )
    gender_net = cv2.dnn.readNet(
        str(MODELS["gender_model"]["path"]),
        str(MODELS["gender_proto"]["path"])
    )
    age_net = cv2.dnn.readNet(
        str(MODELS["age_model"]["path"]),
        str(MODELS["age_proto"]["path"])
    )

    for net in [face_net, gender_net, age_net]:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("[INFO] Model berhasil dimuat.\n")
    return face_net, gender_net, age_net