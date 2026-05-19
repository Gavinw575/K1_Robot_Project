#!/usr/bin/env python3
"""
K1 Robot — YOLOv8 Live Detection
Polls the robot's HTTP camera bridge and runs real-time object detection on GPU.
Run: python3 yolo_detect.py
Press Q in the window to quit.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import shutil
import subprocess

# ── Dependency bootstrap ──────────────────────────────────────────────────────
def _can_import(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False

def _ensure_pip_deps():
    deps = [
        ("cv2",          "opencv-python"),
        ("numpy",        "numpy"),
        ("ultralytics",  "ultralytics"),
    ]
    missing = [pkg for mod, pkg in deps if not _can_import(mod)]
    if missing:
        print(f"[yolo] Installing: {', '.join(missing)} ...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing
            )
            print("[yolo] Packages ready.")
        except subprocess.CalledProcessError as e:
            print(f"[yolo] pip install failed: {e}")

def _warn_system_deps():
    missing = [t for t in ("sshpass",) if not shutil.which(t)]
    if missing:
        print(f"[yolo] Missing system tools: sudo apt install {' '.join(missing)}")

_ensure_pip_deps()
_warn_system_deps()
# ─────────────────────────────────────────────────────────────────────────────

import time
import threading
import urllib.request
from collections import Counter
import numpy as np
import cv2
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
ROBOT_IP      = "192.168.10.102"
ROBOT_USER    = "booster"
ROBOT_PASS    = "123456"
CAMERA_URL    = f"http://{ROBOT_IP}:8080/frame.jpg"
YOLO_MODEL    = "yolov8n.pt"   # auto-downloaded on first run
USE_GPU       = True            # flip to False to force CPU
CONF_THRESH   = 0.4
FRAME_TIMEOUT = 3
# ─────────────────────────────────────────────────────────────────────────────


def _start_camera_bridge():
    """Start the HTTP camera bridge on the robot if port 8080 isn't already open."""
    print("[yolo] Starting camera bridge on robot...")
    subprocess.run([
        "sshpass", f"-p{ROBOT_PASS}",
        "ssh", "-o", "StrictHostKeyChecking=no",
        f"{ROBOT_USER}@{ROBOT_IP}",
        (
            "ss -tlnp 2>/dev/null | grep -q ':8080' || "
            "setsid bash -c 'source /opt/ros/humble/setup.bash && "
            "python3 ~/robot_video_bridge.py "
            "--topic /booster_video_stream --port 8080 "
            ">/tmp/video_bridge.log 2>&1' </dev/null &"
        ),
    ], capture_output=True)
    print("[yolo] Camera bridge ready.")


def _fetch_frame() -> "np.ndarray | None":
    try:
        with urllib.request.urlopen(CAMERA_URL, timeout=FRAME_TIMEOUT) as resp:
            data = resp.read()
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _draw_overlay(frame: np.ndarray, fps: float, results) -> np.ndarray:
    """Draw FPS + detection summary bar at the bottom of the frame."""
    annotated = results[0].plot()
    h, w = annotated.shape[:2]

    boxes = results[0].boxes
    if boxes is not None and len(boxes):
        names = [results[0].names[int(c)] for c in boxes.cls.cpu().numpy()]
        counts = Counter(names)
        summary = "  ".join(f"{v}× {k}" for k, v in counts.most_common(5))
    else:
        summary = "no detections"

    # Semi-transparent bar
    bar = annotated.copy()
    cv2.rectangle(bar, (0, h - 34), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(bar, 0.55, annotated, 0.45, 0, annotated)
    cv2.putText(
        annotated,
        f"FPS {fps:4.1f}   {summary}",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return annotated


def main():
    threading.Thread(target=_start_camera_bridge, daemon=True).start()

    print(f"[yolo] Loading {YOLO_MODEL}...")
    model = YOLO(YOLO_MODEL)
    device = "cuda" if USE_GPU else "cpu"
    try:
        model.to(device)
        print(f"[yolo] Running on {device.upper()}.")
    except Exception:
        model.to("cpu")
        print("[yolo] CUDA unavailable — falling back to CPU.")

    print("[yolo] Connecting to camera... (press Q to quit)")

    fps_timer  = time.time()
    fps        = 0.0
    tick       = 0

    while True:
        frame = _fetch_frame()

        if frame is None:
            blank = np.full((360, 640, 3), 40, dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera bridge...",
                        (130, 185), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (180, 180, 180), 2, cv2.LINE_AA)
            cv2.putText(blank, CAMERA_URL,
                        (175, 215), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.imshow("K1  YOLO Detection", blank)
            if cv2.waitKey(250) & 0xFF == ord('q'):
                break
            continue

        results = model(frame, conf=CONF_THRESH, verbose=False)

        # FPS — averaged over 1-second windows
        tick += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = tick / (now - fps_timer)
            tick = 0
            fps_timer = now

        annotated = _draw_overlay(frame, fps, results)
        cv2.imshow("K1  YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("[yolo] Exited.")


if __name__ == "__main__":
    main()
