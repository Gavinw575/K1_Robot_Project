#!/usr/bin/env python3
"""
K1 Robot — Voice + Vision Control
Run: python3 k1_voice_vision.py
Hold the button to speak, release to send. Say "what do you see?" for vision.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Dependency bootstrap (runs before third-party imports) ───────────────────
import sys
import shutil
import subprocess

def _can_import(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False

def _ensure_pip_deps():
    deps = [
        ("groq",        "groq"),
        ("whisper",     "openai-whisper"),
        ("sounddevice", "sounddevice"),
        ("numpy",       "numpy"),
        ("PIL",         "Pillow"),
    ]
    missing = [pkg for mod, pkg in deps if not _can_import(mod)]
    if missing:
        print(f"[k1] Installing missing packages: {', '.join(missing)} ...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing
            )
            print("[k1] Packages ready.")
        except subprocess.CalledProcessError as e:
            print(f"[k1] pip install failed: {e}")

def _warn_system_deps():
    missing = [t for t in ("sshpass", "ffmpeg", "espeak-ng") if not shutil.which(t)]
    if missing:
        print(f"\n[k1] Missing system tools — install with:")
        print(f"     sudo apt install {' '.join(missing)}\n")

_ensure_pip_deps()
_warn_system_deps()
# ─────────────────────────────────────────────────────────────────────────────

import base64
import re
import tempfile
import threading
import time
import io
import urllib.request
import tkinter as tk
from tkinter import scrolledtext
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
import numpy as np
import sounddevice as sd
import wave
import whisper
from groq import Groq

# ── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY  = "your_groq_api_key_here"  # get one free at console.groq.com
ROBOT_IP      = "192.168.10.102"
ROBOT_USER    = "booster"
ROBOT_PASS    = "123456"
ROBOT_SPEAKER = "alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo"
MIC_DEVICE    = None        # None = system default (laptop mic)
SAMPLE_RATE   = 16000
WHISPER_MODEL = "base"
USE_ROBOT_MIC = True        # True = robot mic via SSH, False = laptop mic
CAMERA_URL    = f"http://{ROBOT_IP}:8080/frame.jpg"
CAMERA_TIMEOUT = 3          # seconds before giving up on frame fetch
TEXT_MODEL    = "llama-3.1-8b-instant"
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the command interpreter for a Booster K1 humanoid robot.
The user will speak a natural language instruction. Your job is to decide:

1. Is this a robot movement command? Convert it to one of these formats:
   - "walk forward"
   - "walk backward"
   - "turn left N deg"
   - "turn right N deg"
   - "stop"
   - chained: "walk forward | turn left 90 deg | walk forward"
   Respond with: COMMAND: <instruction>

2. Is this a question or conversation? Respond as the robot, short and friendly.
   Respond with: REPLY: <response>

3. Unclear? Ask for clarification.
   Respond with: REPLY: <clarification question>

Keep all responses under 2 sentences."""

# Phrases that should trigger the vision model instead of the text model
_VISION_PATTERNS = re.compile(
    r"what (do you|can you|are you) see"
    r"|what('?s| is) (that|this|in front|there|around)"
    r"|what am i (holding|showing|pointing at)"
    r"|what do i have"
    r"|look at (this|that)"
    r"|can you see"
    r"|describe (what|this|that|what you see)"
    r"|tell me what you see"
    r"|what's? (in front|behind|around|to (the )?(left|right))",
    re.IGNORECASE,
)


def is_vision_query(transcript: str) -> bool:
    return bool(_VISION_PATTERNS.search(transcript))


def fetch_camera_frame() -> "str | None":
    """Fetch a JPEG from the camera bridge and return it as a base64 string."""
    try:
        with urllib.request.urlopen(CAMERA_URL, timeout=CAMERA_TIMEOUT) as resp:
            return base64.b64encode(resp.read()).decode("utf-8")
    except Exception:
        return None


def ask_vision_model(groq_client: Groq, transcript: str, b64_image: str) -> str:
    return groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a robot's vision system. Answer in 1-2 sentences, clearly and concisely.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                    {"type": "text", "text": transcript},
                ],
            },
        ],
        max_tokens=80,
    ).choices[0].message.content.strip()


def robot_speak(text):
    clean = text.replace('"', "'")
    ssh_cmd = (
        f'sshpass -p "{ROBOT_PASS}" ssh -o StrictHostKeyChecking=no '
        f'{ROBOT_USER}@{ROBOT_IP} '
        f'"espeak-ng \\"{clean}\\" --stdout | aplay -D plughw:0,0 -q 2>/dev/null '
        f'|| espeak-ng \\"{clean}\\" 2>/dev/null"'
    )
    subprocess.Popen(ssh_cmd, shell=True)


def record_robot_mic(stop_event):
    # Safety net: free the mic in case PulseAudio or booster-audio restarted
    subprocess.run(
        f'sshpass -p "{ROBOT_PASS}" ssh -o StrictHostKeyChecking=no '
        f'{ROBOT_USER}@{ROBOT_IP} '
        f'"systemctl --user stop pulseaudio 2>/dev/null; '
        f'pkill -x pulseaudio 2>/dev/null; '
        f'pkill -x booster-audio 2>/dev/null; true"',
        shell=True, capture_output=True
    )
    proc = subprocess.Popen(
        f'sshpass -p "{ROBOT_PASS}" ssh -o StrictHostKeyChecking=no '
        f'{ROBOT_USER}@{ROBOT_IP} '
        f'"arecord -D plughw:1,0 -f S16_LE -r 16000 -c 6 --quiet /tmp/rec.wav"',
        shell=True
    )
    stop_event.wait()
    subprocess.run(
        f'sshpass -p "{ROBOT_PASS}" ssh -o StrictHostKeyChecking=no '
        f'{ROBOT_USER}@{ROBOT_IP} "pkill -INT arecord"',
        shell=True
    )
    proc.wait()
    time.sleep(0.3)
    result = subprocess.run(
        f'sshpass -p "{ROBOT_PASS}" ssh -o StrictHostKeyChecking=no '
        f'{ROBOT_USER}@{ROBOT_IP} "cat /tmp/rec.wav"',
        shell=True, capture_output=True
    )
    return result.stdout


def record_laptop_mic(stop_event):
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        device=MIC_DEVICE, callback=callback):
        stop_event.wait()

    audio = np.concatenate(chunks, axis=0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        fname = f.name
    with wave.open(fname, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return open(fname, 'rb').read()


class K1VoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("K1 Robot Voice Control")
        self.root.geometry("700x680")
        self.root.configure(bg="#f8f5ee")
        self.root.resizable(False, False)

        self.recording = False
        self.stop_event = None
        self.whisper_model = None
        self.groq_client = None
        self.ready = False

        self._build_ui()
        threading.Thread(target=self._load_models, daemon=True).start()
        threading.Thread(target=self._prepare_robot_audio, daemon=True).start()
        threading.Thread(target=self._start_camera_bridge, daemon=True).start()

    def _build_ui(self):
        title_frame = tk.Frame(self.root, bg="#f8f5ee")
        title_frame.pack(pady=(20, 0))

        tk.Label(title_frame, text="K1", font=("Courier", 42, "bold"),
                 fg="#0077aa", bg="#f8f5ee").pack(side=tk.LEFT)
        tk.Label(title_frame, text=" VOICE CONTROL",
                 font=("Courier", 20, "bold"),
                 fg="#1a1a2e", bg="#f8f5ee").pack(side=tk.LEFT, pady=(14, 0))

        self.status_var = tk.StringVar(value="Loading models...")
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var,
            font=("Courier", 11), fg="#666666", bg="#f8f5ee"
        )
        self.status_label.pack(pady=(4, 0))

        btn_frame = tk.Frame(self.root, bg="#f8f5ee")
        btn_frame.pack(pady=20)

        # Record + Stop side by side
        row = tk.Frame(btn_frame, bg="#f8f5ee")
        row.pack()

        self.record_btn = tk.Button(
            row,
            text="⏺  HOLD TO SPEAK",
            font=("Courier", 14, "bold"),
            fg="#ffffff", bg="#0077aa",
            activebackground="#005f8a",
            relief=tk.FLAT,
            padx=30, pady=14,
            cursor="hand2",
            state=tk.DISABLED,
        )
        self.record_btn.bind("<ButtonPress-1>", self._on_press)
        self.record_btn.bind("<ButtonRelease-1>", self._on_release)
        self.record_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_btn = tk.Button(
            row,
            text="■  STOP",
            font=("Courier", 14, "bold"),
            fg="#ffffff", bg="#cc2244",
            activebackground="#aa1133",
            relief=tk.FLAT,
            padx=20, pady=14,
            cursor="hand2",
            command=self._on_stop,
        )
        self.stop_btn.pack(side=tk.LEFT)

        # Mic source toggle
        self.mic_var = tk.StringVar(value="🤖 Robot Mic" if USE_ROBOT_MIC else "💻 Laptop Mic")
        self.mic_btn = tk.Button(
            btn_frame,
            textvariable=self.mic_var,
            font=("Courier", 10),
            fg="#555555", bg="#e0ddd0",
            activebackground="#d0cdc0",
            relief=tk.FLAT,
            padx=10, pady=6,
            cursor="hand2",
            command=self._toggle_mic
        )
        self.mic_btn.pack(pady=(8, 0))

        # Volume slider
        vol_frame = tk.Frame(btn_frame, bg="#f8f5ee")
        vol_frame.pack(pady=(10, 0), fill=tk.X, padx=20)

        tk.Label(vol_frame, text="SPEAKER VOLUME", font=("Courier", 8),
                 fg="#999999", bg="#f8f5ee").pack(anchor=tk.W)

        self.volume_var = tk.IntVar(value=80)
        vol_scale = tk.Scale(
            vol_frame,
            from_=0, to=100,
            orient=tk.HORIZONTAL,
            variable=self.volume_var,
            font=("Courier", 8),
            fg="#555555", bg="#e0ddd0",
            troughcolor="#c8c5bc",
            highlightthickness=0,
            bd=0,
            length=300,
        )
        vol_scale.bind("<ButtonRelease-1>", self._on_volume_change)
        vol_scale.pack()

        self.use_robot_mic = USE_ROBOT_MIC

        tk.Label(self.root, text="TRANSCRIPT", font=("Courier", 9),
                 fg="#999999", bg="#f8f5ee").pack(anchor=tk.W, padx=30)

        self.transcript_var = tk.StringVar(value="—")
        tk.Label(
            self.root, textvariable=self.transcript_var,
            font=("Courier", 12), fg="#333366", bg="#ece9e0",
            wraplength=620, justify=tk.LEFT,
            anchor=tk.W, padx=14, pady=10
        ).pack(fill=tk.X, padx=30, pady=(2, 10))

        tk.Label(self.root, text="ROBOT RESPONSE", font=("Courier", 9),
                 fg="#999999", bg="#f8f5ee").pack(anchor=tk.W, padx=30)

        self.response_var = tk.StringVar(value="—")
        self.response_label = tk.Label(
            self.root, textvariable=self.response_var,
            font=("Courier", 12), fg="#006633", bg="#ece9e0",
            wraplength=620, justify=tk.LEFT,
            anchor=tk.W, padx=14, pady=10
        )
        self.response_label.pack(fill=tk.X, padx=30, pady=(2, 10))

        tk.Label(self.root, text="LOG", font=("Courier", 9),
                 fg="#999999", bg="#f8f5ee").pack(anchor=tk.W, padx=30)

        self.log = scrolledtext.ScrolledText(
            self.root, height=8,
            font=("Courier", 9),
            fg="#445566", bg="#e8e5dc",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.log.pack(fill=tk.X, padx=30, pady=(2, 20))

    def _toggle_mic(self):
        self.use_robot_mic = not self.use_robot_mic
        self.mic_var.set("🤖 Robot Mic" if self.use_robot_mic else "💻 Laptop Mic")

    def _log(self, msg):
        def _do():
            self.log.config(state=tk.NORMAL)
            self.log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            self.log.see(tk.END)
            self.log.config(state=tk.DISABLED)
        self.root.after(0, _do)

    def _set_status(self, msg, color="#666666"):
        self.status_var.set(msg)
        self.status_label.config(fg=color)

    def _prepare_robot_audio(self):
        self._log("Freeing robot mic (stopping PulseAudio + booster-audio)...")
        subprocess.run(
            f'sshpass -p "{ROBOT_PASS}" ssh -o StrictHostKeyChecking=no '
            f'{ROBOT_USER}@{ROBOT_IP} '
            f'"systemctl --user stop pulseaudio 2>/dev/null; '
            f'pkill -x pulseaudio 2>/dev/null; '
            f'pkill -x booster-audio 2>/dev/null; '
            f'true"',
            shell=True, capture_output=True
        )
        self._log("Robot mic ready.")

    def _start_camera_bridge(self):
        self._log("Starting camera bridge on robot...")
        # Port check avoids pgrep matching its own SSH session command string.
        # setsid creates a new session so the process survives SSH disconnect.
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
        self._log("Camera bridge ready.")

    def _load_models(self):
        self._log(f"Loading Whisper ({WHISPER_MODEL})...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        self._log("Whisper ready.")
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self._log("Groq client ready.")
        self.ready = True
        self.root.after(0, self._set_ready)

    def _set_ready(self):
        self.record_btn.config(state=tk.NORMAL)
        self._set_status("Ready — press the button to speak", "#0077aa")

    def _on_press(self, event):
        if not self.ready or self.recording:
            return
        self.stop_event = threading.Event()
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _on_release(self, event):
        if self.stop_event:
            self.stop_event.set()

    def _run_pipeline(self):
        self.recording = True
        self.root.after(0, lambda: self.record_btn.config(
            state=tk.DISABLED, text="⏺  RECORDING...", bg="#cc2244"))
        self.root.after(0, lambda: self._set_status(
            "Recording... (release to stop)", "#cc2244"))

        if self.use_robot_mic:
            self._log("Recording from robot mic...")
            audio_bytes = record_robot_mic(self.stop_event)
        else:
            self._log("Recording from laptop mic...")
            audio_bytes = record_laptop_mic(self.stop_event)

        if len(audio_bytes) < 1000:
            self._log("⚠ No audio captured.")
            self.root.after(0, lambda: self._set_status("No audio — try again", "#cc6600"))
            self._reset_btn()
            return

        self.root.after(0, lambda: self._set_status("Transcribing...", "#aa6600"))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            fname = f.name

        mono = fname.replace(".wav", "_mono.wav")
        subprocess.run(
            f'ffmpeg -i {fname} '
            f'-af "pan=mono|c0=0.333*c0+0.333*c1+0.333*c2,highpass=f=200" '
            f'-ar 16000 {mono} -y -loglevel quiet',
            shell=True
        )

        result = self.whisper_model.transcribe(mono, fp16=False, language="en")
        transcript = result["text"].strip()

        if not transcript:
            self._log("⚠ Nothing heard.")
            self.root.after(0, lambda: self._set_status("Nothing heard — try again", "#cc6600"))
            self._reset_btn()
            return

        self._log(f'You said: "{transcript}"')
        self.root.after(0, lambda: self.transcript_var.set(f'"{transcript}"'))

        # ── Fast-path stop — skip Groq roundtrip ─────────────────────────────
        if re.fullmatch(r'(please\s+)?(robot[,\s]+)?stop[\s!.]*', transcript.strip(), re.IGNORECASE):
            self._on_stop()
            self._reset_btn()
            return

        # ── Vision path ───────────────────────────────────────────────────────
        if is_vision_query(transcript):
            self.root.after(0, lambda: self._set_status("Looking...", "#773399"))
            self.root.after(0, lambda: self.response_label.config(fg="#773399"))
            self._log("Vision query detected — fetching camera frame...")
            b64 = fetch_camera_frame()
            if b64 is None:
                reply = "I can't see right now, is the camera bridge running?"
                self._log("⚠ Camera fetch failed — bridge may not be running.")
            else:
                self._log(f"Frame captured ({CAMERA_URL}), asking vision model...")
                self._show_camera_popup(b64)
                self.root.after(0, lambda: self._set_status("Asking vision model...", "#773399"))
                reply = ask_vision_model(self.groq_client, transcript, b64)
            self._log(f"→ Vision reply: {reply}")
            self.root.after(0, lambda: self.response_var.set(f"👁 {reply}"))
            robot_speak(reply)
            self.root.after(0, lambda: self._set_status("Ready", "#0077aa"))
            self._reset_btn()
            return

        # ── Text / command path ───────────────────────────────────────────────
        self.root.after(0, lambda: self._set_status("Thinking...", "#5544bb"))
        response = self.groq_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": transcript},
            ],
            max_tokens=100,
        ).choices[0].message.content.strip()

        self._log(f"Groq: {response}")

        if response.startswith("COMMAND:"):
            cmd = response.split("COMMAND:", 1)[1].strip()
            self.root.after(0, lambda: self.response_var.set(f"🚀 COMMAND: {cmd}"))
            self.root.after(0, lambda: self.response_label.config(fg="#006633"))
            self._log(f"→ Robot command: {cmd}")
            robot_speak(f"Executing: {cmd}")
            self.root.after(0, lambda: self._set_status("Command sent!", "#006633"))
        elif response.startswith("REPLY:"):
            reply = response.split("REPLY:", 1)[1].strip()
            self.root.after(0, lambda: self.response_var.set(f"💬 {reply}"))
            self.root.after(0, lambda: self.response_label.config(fg="#0077aa"))
            self._log(f"→ Reply: {reply}")
            robot_speak(reply)
            self.root.after(0, lambda: self._set_status("Ready", "#0077aa"))
        else:
            self.root.after(0, lambda: self.response_var.set(response))
            self.root.after(0, lambda: self._set_status("Ready", "#0077aa"))

        self._reset_btn()

    def _on_stop(self):
        self._log("STOP command triggered.")
        self.root.after(0, lambda: self._set_status("Stopping...", "#cc2244"))
        self.root.after(0, lambda: self.response_var.set("■ STOP"))
        self.root.after(0, lambda: self.response_label.config(fg="#cc2244"))
        threading.Thread(target=self._send_stop, daemon=True).start()

    def _send_stop(self):
        robot_speak("Stopping")
        # Wire B1LocoClient.Stop() here when motion pipeline is connected
        self.root.after(0, lambda: self._set_status("Ready", "#0077aa"))

    def _on_volume_change(self, event=None):
        vol = self.volume_var.get()
        threading.Thread(target=self._set_robot_volume, args=(vol,), daemon=True).start()

    def _set_robot_volume(self, percent):
        subprocess.run(
            f'sshpass -p "{ROBOT_PASS}" ssh -o StrictHostKeyChecking=no '
            f'{ROBOT_USER}@{ROBOT_IP} '
            f'"amixer -c 0 set Speaker {percent}% 2>/dev/null || '
            f'amixer -c 0 set PCM {percent}% 2>/dev/null || '
            f'amixer -c 0 set Master {percent}% 2>/dev/null"',
            shell=True, capture_output=True
        )
        self._log(f"Volume → {percent}%")

    def _show_camera_popup(self, b64_image: str):
        if not PIL_AVAILABLE:
            self._log("⚠ Pillow not installed — skipping camera popup (pip install Pillow)")
            return

        def _do():
            img = Image.open(io.BytesIO(base64.b64decode(b64_image)))
            img.thumbnail((480, 320))

            popup = tk.Toplevel(self.root)
            popup.title("What the robot sees")
            popup.configure(bg="#f8f5ee")
            popup.resizable(False, False)

            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(popup, image=photo, bg="#f8f5ee")
            lbl.image = photo  # prevent garbage collection
            lbl.pack(padx=10, pady=10)

            tk.Button(
                popup, text="Close", command=popup.destroy,
                font=("Courier", 10, "bold"),
                fg="#ffffff", bg="#0077aa",
                activebackground="#005f8a",
                relief=tk.FLAT, padx=20, pady=8
            ).pack(pady=(0, 10))

        self.root.after(0, _do)

    def _reset_btn(self):
        self.recording = False
        self.root.after(0, lambda: self.record_btn.config(
            state=tk.NORMAL, text="⏺  HOLD TO SPEAK", bg="#0077aa"))


def main():
    root = tk.Tk()
    app = K1VoiceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
