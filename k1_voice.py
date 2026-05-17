

#!/usr/bin/env python3
"""K1 Voice Command System — GUI version"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import tempfile
import subprocess
import threading
import time
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
from groq import Groq

# ── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY  = "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"
ROBOT_IP      = "192.168.10.102"
ROBOT_USER    = "booster"
ROBOT_PASS    = "123456"
ROBOT_SPEAKER = "alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo"
MIC_DEVICE    = None       # None = system default (laptop mic)
SAMPLE_RATE   = 16000
WHISPER_MODEL = "base"
USE_ROBOT_MIC = True       # True = robot mic via SSH, False = laptop mic
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
        wav.write(f.name, SAMPLE_RATE, audio)
        return open(f.name, 'rb').read()


class K1VoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("K1 Robot Voice Control")
        self.root.geometry("700x600")
        self.root.configure(bg="#0a0f1e")
        self.root.resizable(False, False)

        self.recording = False
        self.stop_event = None
        self.whisper_model = None
        self.groq_client = None
        self.ready = False

        self._build_ui()
        threading.Thread(target=self._load_models, daemon=True).start()
        threading.Thread(target=self._prepare_robot_audio, daemon=True).start()

    def _build_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#0a0f1e")
        title_frame.pack(pady=(20, 0))

        tk.Label(title_frame, text="K1", font=("Courier", 42, "bold"),
                 fg="#00d4ff", bg="#0a0f1e").pack(side=tk.LEFT)
        tk.Label(title_frame, text=" VOICE CONTROL",
                 font=("Courier", 20, "bold"),
                 fg="#ffffff", bg="#0a0f1e").pack(side=tk.LEFT, pady=(14, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Loading models...")
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var,
            font=("Courier", 11), fg="#888888", bg="#0a0f1e"
        )
        self.status_label.pack(pady=(4, 0))

        # Record button
        btn_frame = tk.Frame(self.root, bg="#0a0f1e")
        btn_frame.pack(pady=20)

        self.record_btn = tk.Button(
            btn_frame,
            text="⏺  HOLD TO SPEAK",
            font=("Courier", 14, "bold"),
            fg="#0a0f1e", bg="#00d4ff",
            activebackground="#00a8cc",
            relief=tk.FLAT,
            padx=30, pady=14,
            cursor="hand2",
            state=tk.DISABLED,
        )
        self.record_btn.bind("<ButtonPress-1>", self._on_press)
        self.record_btn.bind("<ButtonRelease-1>", self._on_release)
        self.record_btn.pack()

        # Mic source toggle
        self.mic_var = tk.StringVar(value="🤖 Robot Mic" if USE_ROBOT_MIC else "💻 Laptop Mic")
        self.mic_btn = tk.Button(
            btn_frame,
            textvariable=self.mic_var,
            font=("Courier", 10),
            fg="#888888", bg="#131929",
            activebackground="#1e2d45",
            relief=tk.FLAT,
            padx=10, pady=6,
            cursor="hand2",
            command=self._toggle_mic
        )
        self.mic_btn.pack(pady=(8, 0))

        self.use_robot_mic = USE_ROBOT_MIC

        # Transcript display
        tk.Label(self.root, text="TRANSCRIPT", font=("Courier", 9),
                 fg="#444466", bg="#0a0f1e").pack(anchor=tk.W, padx=30)

        self.transcript_var = tk.StringVar(value="—")
        tk.Label(
            self.root, textvariable=self.transcript_var,
            font=("Courier", 12), fg="#ccccff", bg="#0d1426",
            wraplength=620, justify=tk.LEFT,
            anchor=tk.W, padx=14, pady=10
        ).pack(fill=tk.X, padx=30, pady=(2, 10))

        # Response display
        tk.Label(self.root, text="ROBOT RESPONSE", font=("Courier", 9),
                 fg="#444466", bg="#0a0f1e").pack(anchor=tk.W, padx=30)

        self.response_var = tk.StringVar(value="—")
        self.response_label = tk.Label(
            self.root, textvariable=self.response_var,
            font=("Courier", 12), fg="#00ff99", bg="#0d1426",
            wraplength=620, justify=tk.LEFT,
            anchor=tk.W, padx=14, pady=10
        )
        self.response_label.pack(fill=tk.X, padx=30, pady=(2, 10))

        # Log box
        tk.Label(self.root, text="LOG", font=("Courier", 9),
                 fg="#444466", bg="#0a0f1e").pack(anchor=tk.W, padx=30)

        self.log = scrolledtext.ScrolledText(
            self.root, height=8,
            font=("Courier", 9),
            fg="#556677", bg="#080c18",
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

    def _set_status(self, msg, color="#888888"):
        self.status_var.set(msg)
        self.status_label.config(fg=color)

    def _prepare_robot_audio(self):
        """Runs once at startup: permanently mask PulseAudio and free the mic."""
        self._log("Freeing robot mic (stopping PulseAudio + booster-audio)...")
        subprocess.run(
            f'sshpass -p "{ROBOT_PASS}" ssh -o StrictHostKeyChecking=no '
            f'{ROBOT_USER}@{ROBOT_IP} '
            f'"systemctl --user mask pulseaudio 2>/dev/null; '
            f'systemctl --user stop pulseaudio 2>/dev/null; '
            f'pkill -x pulseaudio 2>/dev/null; '
            f'pkill -x booster-audio 2>/dev/null; '
            f'true"',
            shell=True, capture_output=True
        )
        self._log("Robot mic ready.")

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
        self._set_status("Ready — press the button to speak", "#00d4ff")

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
            state=tk.DISABLED, text="⏺  RECORDING...", bg="#ff4466"))
        self.root.after(0, lambda: self._set_status(
            "Recording... (release to stop)", "#ff4466"))

        # Record
        if self.use_robot_mic:
            self._log("Recording from robot mic...")
            audio_bytes = record_robot_mic(self.stop_event)
        else:
            self._log("Recording from laptop mic...")
            audio_bytes = record_laptop_mic(self.stop_event)

        if len(audio_bytes) < 1000:
            self._log("⚠ No audio captured.")
            self.root.after(0, lambda: self._set_status("No audio — try again", "#ff8800"))
            self._reset_btn()
            return

        # Save and convert to mono
        self.root.after(0, lambda: self._set_status("Transcribing...", "#ffaa00"))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            fname = f.name

        mono = fname.replace(".wav", "_mono.wav")
        # Mix only front 3 channels (FL/FR/FC), skip LFE/SL/SR which carry motor noise.
        # High-pass at 200Hz removes low-frequency rumble that Whisper reads as Japanese.
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
            self.root.after(0, lambda: self._set_status("Nothing heard — try again", "#ff8800"))
            self._reset_btn()
            return

        self._log(f'You said: "{transcript}"')
        self.root.after(0, lambda: self.transcript_var.set(f'"{transcript}"'))

        # Ask Groq
        self.root.after(0, lambda: self._set_status("Thinking...", "#aa88ff"))
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
            self.root.after(0, lambda: self.response_label.config(fg="#00ff99"))
            self._log(f"→ Robot command: {cmd}")
            robot_speak(f"Executing: {cmd}")
            self.root.after(0, lambda: self._set_status("Command sent!", "#00ff99"))
        elif response.startswith("REPLY:"):
            reply = response.split("REPLY:", 1)[1].strip()
            self.root.after(0, lambda: self.response_var.set(f"💬 {reply}"))
            self.root.after(0, lambda: self.response_label.config(fg="#00d4ff"))
            self._log(f"→ Reply: {reply}")
            robot_speak(reply)
            self.root.after(0, lambda: self._set_status("Ready", "#00d4ff"))
        else:
            self.root.after(0, lambda: self.response_var.set(response))
            self.root.after(0, lambda: self._set_status("Ready", "#00d4ff"))

        self._reset_btn()

    def _reset_btn(self):
        self.recording = False
        self.root.after(0, lambda: self.record_btn.config(
            state=tk.NORMAL, text="⏺  HOLD TO SPEAK", bg="#00d4ff"))


def main():
    root = tk.Tk()
    app = K1VoiceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
