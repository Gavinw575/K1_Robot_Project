# Booster K1 — Custom Voice Command System
**Last updated:** May 2026  
**Robot SN:** 01022026030030 | **Firmware:** V1.6+  
**Laptop:** Ubuntu 22.04, i7 9th gen, GTX 1650 Mobile  
**Ethernet:** Laptop `192.168.10.100` ↔ Robot `192.168.10.102`

---

## Overview

This project replaces Booster's built-in cloud voice assistant with a fully custom pipeline:

```
Mic (robot or laptop)
    ↓
Whisper (local STT on laptop GPU)
    ↓
Groq API — llama-3.1-8b-instant (command parsing)
    ↓
Printed command + robot speaker response (espeak-ng via SSH)
    ↓
(future) Motion commands → friend's laptop → navila_k1_realrobot.py → B1LocoClient.Move()
```

---

## Files

| File | Location | Purpose |
|------|----------|---------|
| `k1_voice.py` | `~/k1_voice.py` (laptop) | Main voice command GUI |
| `robot_video_bridge.py` | `~/robot_video_bridge.py` (robot) | HTTP MJPEG camera bridge |
| `view_camera.py` | `~/view_camera.py` (laptop) | ROS2 camera viewer |
| `yolo_detect.py` | `~/yolo_detect.py` (laptop) | YOLO object detection |

---

## Robot Audio Setup

### Problem
`booster-audio` (PID varies) and PulseAudio both lock the NationalChip 6-channel mic array on startup, preventing any other process from recording.

### Fix — Mask PulseAudio (survives reboots, wiped by firmware updates)
```bash
# SSH into robot first
ssh booster@192.168.10.102  # password: 123456

# Stop and mask PulseAudio so it doesn't auto-restart
systemctl --user stop pulseaudio
systemctl --user mask pulseaudio
```

### Restore normal robot behavior
```bash
systemctl --user unmask pulseaudio
systemctl --user start pulseaudio
```

> **Note:** Masking PulseAudio disables Booster's built-in Hi Chat / LUI voice agent since it routes through PulseAudio. Our pipeline uses ALSA directly so it still works.

### Mic device
```
Card: NationalChip Uac Speaker (card 1, device 0)
ALSA device: plughw:1,0
Channels: 6
Sample rate: 16000 Hz
Format: S16_LE
```

### Speaker device
```
Card: USB-C Media Electronics (card 0)
ALSA device: plughw:0,0
Sample rate: 44100 Hz
```

### Record from robot mic (manual test)
```bash
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 6 -d 5 /tmp/test.wav
aplay /tmp/test.wav
```

### Speak through robot speaker (manual test)
```bash
espeak-ng "hello" --stdout | aplay -D plughw:0,0 -q
```

---

## Voice Pipeline (k1_voice.py)

### Dependencies (laptop)
```bash
sudo apt install portaudio19-dev ffmpeg sshpass espeak-ng
pip install sounddevice scipy numpy openai-whisper groq
```

### Config variables at top of k1_voice.py
```python
GROQ_API_KEY  = "your_key_here"       # from console.groq.com
ROBOT_IP      = "192.168.10.102"
ROBOT_USER    = "booster"
ROBOT_PASS    = "123456"
ROBOT_SPEAKER = "alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo"
MIC_DEVICE    = None                   # None = system default (laptop mic)
SAMPLE_RATE   = 16000
RECORD_SECS   = 5
WHISPER_MODEL = "base"                 # tiny/base/small
USE_ROBOT_MIC = True                   # toggle in GUI or set here
```

### Running
```bash
python3 ~/k1_voice.py
```

Opens a dark-themed GUI. Press the record button, speak, and the robot will respond via speaker and print the parsed command.

### How it works
1. Records audio (robot mic via SSH or laptop mic via sounddevice)
2. ffmpeg converts 6-channel audio to mono 16kHz WAV
3. Whisper transcribes with `language="en"` forced
4. Groq llama-3.1-8b-instant parses the transcript
5. If `COMMAND:` → prints command + robot says "Executing: ..."
6. If `REPLY:` → robot speaks the reply

### Groq system prompt
The LLM is instructed to output either:
- `COMMAND: walk forward | turn left 90 deg` — robot motion instruction
- `REPLY: <conversational response>` — robot speaks back

---

## Camera

### HTTP bridge (better latency than ROS2 subscriber)
```bash
# On robot:
source /opt/ros/humble/setup.bash
python3 ~/robot_video_bridge.py --topic /booster_video_stream --port 8080

# On laptop — view stream:
python3 -c "
import cv2, urllib.request, numpy as np, time
while True:
    try:
        with urllib.request.urlopen('http://192.168.10.102:8080/frame.jpg', timeout=2) as r:
            data = r.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow('K1', img)
    except Exception as e:
        time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"
```

### ROS2 viewer (original)
```bash
# Requires FastDDS env vars set (see main K1 setup doc)
python3 ~/view_camera.py
```

---

## Robot Audio Architecture (discovered via reverse engineering)

| Component | Binary | Purpose |
|-----------|--------|---------|
| `booster_lui` | `/opt/booster/BoosterLui/bin/booster_lui` | LUI service — handles ASR via ByteDance Volcengine WebSocket WSS, TTS, RPC |
| `booster-audio` | `/opt/booster/BoosterAudio/bin/booster-audio` | ALSA capture manager — locks mic at startup |
| `lui_rpc_service_bridge` | ROS2 node | Bridges JSON RPC commands to booster_lui |
| `booster_rtc_cli` | `/opt/booster/RTCCli/bin/booster_rtc_cli` | ByteDance Volcengine RTC client |

### Built-in LLM config
- Config file: `/opt/booster/RTCCli/custom_settings.toml` (survives firmware updates)
- Default config: `/opt/booster/RTCCli/default_settings.toml` (overwritten on update)
- LLM endpoint: hardcoded in `multi_voice_agent` binary (ByteDance/Volcengine, not configurable)
- System prompt: configurable via `system_prompt = ""` in custom_settings.toml
- ASR: streams to `wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async`
- TTS: streams to `wss://openspeech.bytedance.com/api/v3/tts/bidirection`

---

## Quick Start Every Session

### Laptop
```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_k1.xml
ros2 daemon stop && ros2 daemon start && sleep 5
```

### Robot (SSH)
```bash
ssh booster@192.168.10.102
# If mic needed, mask PulseAudio:
systemctl --user stop pulseaudio && systemctl --user mask pulseaudio
```

### Start voice system
```bash
# Laptop:
python3 ~/k1_voice.py
```

---

## Next Steps

- [ ] Wire `COMMAND:` output to friend's laptop HTTP server → `navila_k1_realrobot.py` → `B1LocoClient.Move()`
- [ ] Fix robot mic audio quality (sox normalization, gain boost)
- [ ] Add vision commands — grab camera frame + send to Groq vision model on "what do you see?"
- [ ] Intercept NaVILA action strings and speak them aloud via robot speaker
- [ ] Set up RTAB-Map for autonomous mapping

---

## Known Issues

| Issue | Cause | Workaround |
|-------|-------|------------|
| Mic locked on boot | `booster-audio` locks ALSA device | Mask PulseAudio before recording |
| Mask wiped on firmware update | Firmware restores systemd config | Re-run mask command after updates |
| Robot mic hears motor noise | Mic array picks up internal vibration | Speak loudly and close to robot head |
| Camera stream laggy | 10fps JPEG over ethernet | Use `robot_video_bridge.py` HTTP endpoint |
| `/lui_asr_chunk` topic invalid | `booster_interface` not in laptop SDK | Use our own Whisper STT instead |
