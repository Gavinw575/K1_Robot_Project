# K1 Robot — Custom Voice + Vision Control

**Hardware:** Booster K1 humanoid | Firmware V1.6+ |  
**Laptop:** Ubuntu 22.04, i7 9th gen, GTX 1650 Mobile  
**Network:** Laptop `192.168.10.100` ↔ Robot `192.168.10.102` (ethernet)

---

## What it does

Replaces Booster's built-in cloud voice assistant with a fully local + Groq pipeline:

```
Hold button → speak
      ↓
Whisper (local STT, laptop GPU)
      ↓
      ├─ Vision phrase? ("what do you see?", "what is that?", ...)
      │        ↓
      │   Grab frame from robot camera (HTTP :8080/frame.jpg)
      │        ↓
      │   Groq llama-4-scout-17b (vision model) → popup + spoken reply
      │
      └─ Movement / conversation
               ↓
          Groq llama-3.1-8b-instant → COMMAND or REPLY
               ↓
          espeak-ng through robot speaker via SSH
```

---

## Quick Start

**1. Install dependencies (laptop)**
```bash
sudo apt install portaudio19-dev ffmpeg sshpass espeak-ng
pip install sounddevice numpy openai-whisper groq Pillow
```

For YOLO detection only:
```bash
pip install opencv-python numpy ultralytics
```

**2. Add your Groq API key**

Edit the `GROQ_API_KEY` line at the top of `k1_voice_vision.py` — get a free key at [console.groq.com](https://console.groq.com).

**3. Run the voice + vision app (laptop)**
```bash
python3 ~/k1-voice-control/k1_voice_vision.py
```

On launch the app automatically frees the robot mic (stops PulseAudio and booster-audio via SSH) and starts the camera bridge on the robot. Hold the button to speak, release to send.

**Run YOLO live detection (laptop, separate terminal)**
```bash
python3 ~/k1-voice-control/yolo_detect.py
```

The camera bridge is started automatically. Press **Q** in the window to quit. Model (`yolov8n.pt`) is downloaded automatically on first run.

---

## Files

| File | Purpose |
|------|---------|
| `k1_voice_vision.py` | **Main app** — voice + vision + nav-object routing, hold-to-record GUI |
| `yolo_detect.py` | **Live YOLO detection** — real-time object detection on the robot camera feed |
| `k1_voice_vision_backup.py` | Backup of the main app before nav-object changes |
| `k1_voice.py` | Original voice-only version (no vision) |
| `K1_VOICE_SETUP.txt` | Full robot setup notes and architecture reference |

---

## Configuration

Edit the block at the top of `k1_voice_vision.py`:

```python
GROQ_API_KEY  = "your_key_here"      # console.groq.com
ROBOT_IP      = "192.168.10.102"
ROBOT_USER    = "booster"
ROBOT_PASS    = "123456"
MIC_DEVICE    = None                  # None = system default mic
WHISPER_MODEL = "base"                # tiny / base / small
USE_ROBOT_MIC = True                  # flip to False for laptop mic
CAMERA_URL    = "http://192.168.10.102:8080/frame.jpg"
```

---

## Voice Commands

The LLM routes each transcription to one of two outputs:

| Output | Example | What happens |
|--------|---------|--------------|
| `COMMAND: walk forward` | "walk forward", "turn left 90 degrees" | Robot acknowledges and executes |
| `REPLY: <text>` | "what's your name?", "stop" | Robot speaks the reply |

Supported movement commands: `walk forward`, `walk backward`, `turn left N deg`, `turn right N deg`, `stop`, and chains like `walk forward \| turn left 90 deg`.

---

## Vision

Say anything matching these patterns and the app grabs a camera frame automatically:

> "what do you see", "what is that", "what am I holding", "what's in front of you", "look at this", "can you see", "describe what you see", "what's this", ...

The frame pops up in a window and the robot speaks a 1–2 sentence description. If the camera bridge isn't running it says so out loud.

**Vision model:** `meta-llama/llama-4-scout-17b-16e-instruct` (Groq)  
**Text model:** `llama-3.1-8b-instant` (Groq) — used for all non-vision queries

---

## Robot Audio

The app handles mic access automatically on every launch — no manual SSH steps needed.

**What it does:** SSHes to the robot and stops `PulseAudio` and `booster-audio`, both of which lock the 6-channel mic array on boot. They come back on robot reboot.

**To restore manually** (if you need Booster's built-in voice agent back):
```bash
ssh booster@192.168.10.102
systemctl --user start pulseaudio
/opt/booster/BoosterAudio/bin/booster-audio &
```

### Mic / speaker reference

| Device | ALSA | Channels | Rate |
|--------|------|----------|------|
| NationalChip mic array | `plughw:1,0` | 6 (FL FR FC LFE SL SR) | 16000 Hz |
| USB speaker (C-Media) | `plughw:0,0` | 2 | 44100 Hz |

The app mixes only the front 3 channels (FL/FR/FC) and applies a 200 Hz high-pass filter to cut motor noise before transcription.

**Manual mic test:**
```bash
# On robot:
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 6 -d 5 /tmp/test.wav && aplay /tmp/test.wav
```

---

## Robot Internals (for reference)

| Component | Binary | What it does |
|-----------|--------|--------------|
| `booster-audio` | `/opt/booster/BoosterAudio/bin/booster-audio` | Locks mic at boot, GPU beamforming |
| `booster_lui` | `/opt/booster/BoosterLui/bin/booster_lui` | Built-in ASR/TTS via ByteDance Volcengine |
| `booster_rtc_cli` | `/opt/booster/RTCCli/bin/booster_rtc_cli` | Volcengine RTC client |

Built-in LLM config lives at `/opt/booster/RTCCli/custom_settings.toml` — the `system_prompt` field is editable and survives firmware updates.

---

## Known Issues

| Issue | Status |
|-------|--------|
| Mic locked by booster-audio / PulseAudio on boot | **Auto-fixed at app launch** |
| Motor noise causing Whisper to transcribe as Japanese | **Fixed** — front-3-channel mix + 200 Hz high-pass |
| Mic lock returns after firmware update | Expected — app re-fixes it on next launch |
| Camera popup requires Pillow | Install with `pip install Pillow` |

---

## Next Steps

- [ ] Wire `COMMAND:` output → HTTP → `navila_k1_realrobot.py` → `B1LocoClient.Move()`
- [ ] Intercept NaVILA action strings and speak them via robot speaker
- [ ] Continuous listen mode (wake word instead of hold button)
- [ ] Set up RTAB-Map for autonomous mapping
