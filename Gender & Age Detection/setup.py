"""
setup.py
========
Jalankan SEKALI sebelum memakai detect.py:
    python setup.py
"""

import sys
import subprocess

REQUIRED_PACKAGES = [
    ("opencv-python", "cv2"),
    ("numpy",         "numpy"),
]

PYTHON_MIN = (3, 8)


def check_python():
    v = sys.version_info
    if v < PYTHON_MIN:
        print(f"[ERROR] Python {PYTHON_MIN[0]}.{PYTHON_MIN[1]}+ dibutuhkan.")
        sys.exit(1)
    print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")


def install_packages():
    print("\n[INFO] Memeriksa dependensi ...")
    for pkg_install, pkg_import in REQUIRED_PACKAGES:
        try:
            __import__(pkg_import)
            print(f"[OK] {pkg_install} sudah terinstall")
        except ImportError:
            print(f"[INSTALL] Menginstall {pkg_install} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_install])
            print(f"[OK] {pkg_install} berhasil diinstall")


def download_models():
    print("\n[INFO] Memeriksa model ...")
    from model_loader import ensure_models
    ensure_models()


def print_usage():
    print("\n" + "=" * 55)
    print("  Setup selesai! Cara penggunaan:")
    print("=" * 55)
    print()
    print("  Webcam          : python detect.py")
    print("  Kamera lain     : python detect.py --camera 1")
    print("  Dari gambar     : python detect.py --image foto.jpg")
    print("  Dari video      : python detect.py --video video.mp4")
    print()
    print("  Kontrol webcam  :  q = keluar  |  s = screenshot")
    print()


if __name__ == "__main__":
    print("=" * 55)
    print("  Gender & Age Detection  —  Setup")
    print("=" * 55)

    check_python()
    install_packages()
    download_models()
    print_usage()