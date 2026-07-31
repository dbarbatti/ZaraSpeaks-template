# modules/audio_input.py
# -----------------------------------------------------------
#  Audio input: microphone recording + Whisper transcription
#  Unchanged from V2 except cleaner encapsulation.
# -----------------------------------------------------------

import os
import sys
import time
import wave
import tempfile
from openai import OpenAI
from config import (
    OPENAI_API_KEY, AUDIO_CHANNELS, AUDIO_RATE, AUDIO_CHUNK,
    DEFAULT_RECORD_SECONDS
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def record_audio(duration=DEFAULT_RECORD_SECONDS):
    """
    Record audio from the microphone.

    Args:
        duration: Recording length in seconds

    Returns:
        str: Path to temporary WAV file, or None on failure
    """
    try:
        import pyaudio
    except ImportError:
        print("[Audio] ERROR: pyaudio not installed. Run: pip install pyaudio")
        return None

    p = pyaudio.PyAudio()

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK
        )

        print(">> Speak now...")
        frames = []
        total_chunks = int(AUDIO_RATE / AUDIO_CHUNK * duration)

        for _ in range(total_chunks):
            data = stream.read(AUDIO_CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        # Write to temp file
        tmp = tempfile.mktemp(suffix=".wav")
        with wave.open(tmp, 'wb') as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(AUDIO_RATE)
            wf.writeframes(b"".join(frames))

        return tmp

    except Exception as e:
        print(f"[Audio] Recording error: {e}")
        return None
    finally:
        p.terminate()


def transcribe_audio(audio_path):
    """
    Transcribe audio using OpenAI Whisper API.

    Args:
        audio_path: Path to WAV file

    Returns:
        str: Transcribed text, or None on failure
    """
    try:
        with open(audio_path, "rb") as f:
            transcript = openai_client.audio.transcriptions.create(
                file=f,
                model="whisper-1"
            )
        return transcript.text
    except Exception as e:
        print(f"[Audio] Transcription error: {e}")
        return None


def record_and_transcribe(duration=DEFAULT_RECORD_SECONDS):
    """
    Record from microphone and transcribe in one step.

    Returns:
        str: Transcribed text
    """
    audio_path = record_audio(duration)
    if not audio_path:
        return "[Recording failed]"

    try:
        text = transcribe_audio(audio_path)
        return text if text else "[Transcription failed]"
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


def get_text_input():
    """
    Get text input from user with clipboard paste support.

    Returns:
        str: User's input text
    """
    print("\n>> Enter text (type your message and press Enter):")
    print("   Type 'paste' to use clipboard content")
    user_text = input("Text: ").strip()

    if user_text.lower() == "paste":
        try:
            import pyperclip
            user_text = pyperclip.paste()
            print(f"Pasted: {user_text}")
        except ImportError:
            print("[pyperclip not installed, manual entry only]")
            user_text = input("Text (manual entry): ").strip()
        except Exception as e:
            print(f"[Clipboard error: {e}]")
            user_text = input("Text (manual entry): ").strip()

    return user_text


def get_multiline_input():
    """
    Get multi-line text input (song lyrics, poems, passages) with line
    breaks preserved. Collects lines until a line containing only 'END'.

    Returns:
        str: The full multi-line text, or "" if nothing entered
    """
    print("\n>> Paste or type multi-line text (lyrics, poems, passages).")
    print("   Line breaks are preserved. Type END on its own line when done.")
    print("   (Type QUIT on its own line to cancel.)\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            # Ctrl-D / Ctrl-Z also ends input gracefully
            break

        stripped = line.strip()
        if stripped == "END":
            break
        if stripped == "QUIT":
            print("[Cancelled]")
            return ""
        lines.append(line)

    text = "\n".join(lines).strip()

    if not text:
        print("[No text entered]")
        return ""

    line_count = len(text.splitlines())
    print(f"\n[Received {line_count} line(s), {len(text)} characters]")
    return text


def get_image_input():
    """
    Get an image file path + text/voice prompt from the user.

    Returns:
        tuple: (user_text, image_b64, image_mime) or (user_text, None, None) on failure
    """
    import base64

    # Get image path
    print("\n>> Enter image path (drag & drop or paste full path):")
    print("   Supported: .png, .jpg, .jpeg, .gif, .webp")
    raw_path = input("Image: ").strip().strip('"').strip("'")

    if not raw_path:
        print("[No path provided — falling back to text-only]")
        text = input("Text: ").strip()
        return text, None, None

    # Resolve path
    image_path = os.path.abspath(raw_path)

    if not os.path.exists(image_path):
        print(f"[File not found: {image_path}]")
        print("[Falling back to text-only]")
        text = input("Text: ").strip()
        return text, None, None

    # Determine MIME type
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    image_mime = mime_map.get(ext)
    if not image_mime:
        print(f"[Unsupported image format: {ext}]")
        print("[Falling back to text-only]")
        text = input("Text: ").strip()
        return text, None, None

    # Read and encode
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        size_mb = os.path.getsize(image_path) / (1024 * 1024)
        print(f"[Image loaded: {os.path.basename(image_path)} ({size_mb:.1f} MB)]")
    except Exception as e:
        print(f"[Error reading image: {e}]")
        print("[Falling back to text-only]")
        text = input("Text: ").strip()
        return text, None, None

    # Size check (OpenAI limit is 20MB)
    if size_mb > 20:
        print("[Image too large — OpenAI limit is 20MB]")
        print("[Falling back to text-only]")
        text = input("Text: ").strip()
        return text, None, None

    # Get prompt — voice or text
    print("\nHow do you want to describe this image to Aria?")
    print("  1. Type a message")
    print("  2. Record voice")
    prompt_method = input("Choice [1-2, default=1]: ").strip()

    if prompt_method == "2":
        dur_in = input("Recording length (sec) [Enter=5]: ").strip()
        dur = float(dur_in) if dur_in else DEFAULT_RECORD_SECONDS

        print("\n>>  Speak when you see 3-2-1")
        import time
        for n in ("3", "2", "1"):
            print(n, end=" ", flush=True)
            time.sleep(1)
        print()

        user_text = record_and_transcribe(dur)
    else:
        user_text = input("Text: ").strip()

    if not user_text:
        user_text = "What do you think of this image?"

    return user_text, image_b64, image_mime


def get_video_input():
    """
    Get a video clip file path + text/voice prompt from the user.
    Extracts sampled frames and a Whisper transcript so Aria can
    "watch" the clip (sight + sound).

    Returns:
        tuple: (user_text, frames_b64, transcript) where frames_b64 is a
               list of base64 PNG strings, or (user_text, None, None) on failure
    """
    import base64
    import subprocess
    import json as _json
    import tempfile
    import shutil

    # Get video path
    print("\n>> Enter video clip path (drag & drop or paste full path):")
    print("   Supported: .mp4, .mov, .avi, .mkv, .webm")
    raw_path = input("Video: ").strip().strip('"').strip("'")

    if not raw_path:
        print("[No path provided — falling back to text-only]")
        text = input("Text: ").strip()
        return text, None, None

    video_path = os.path.abspath(raw_path)

    if not os.path.exists(video_path):
        print(f"[File not found: {video_path}]")
        text = input("Text: ").strip()
        return text, None, None

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        print(f"[Unsupported video format: {ext}]")
        text = input("Text: ").strip()
        return text, None, None

    # Get video duration
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True
        )
        info = _json.loads(result.stdout)
        duration = float(info["format"]["duration"])
    except Exception as e:
        print(f"[Could not read video: {e}]")
        text = input("Text: ").strip()
        return text, None, None

    print(f"[Video: {os.path.basename(video_path)} ({duration:.1f}s)]")

    # Determine frame sampling — 1 frame / 2s, capped at 30 frames
    MAX_FRAMES = 40
    target_interval = 1.0
    n_frames = int(duration / target_interval) + 1
    if n_frames > MAX_FRAMES:
        n_frames = MAX_FRAMES
        target_interval = duration / n_frames
    if n_frames < 1:
        n_frames = 1

    print(f"[Extracting {n_frames} frames (1 every {target_interval:.1f}s)...]")

    tmp_dir = tempfile.mkdtemp(prefix="zara_video_")
    frames_b64 = []

    try:
        # Extract evenly spaced frames
        for i in range(n_frames):
            timestamp = (i * target_interval) + (target_interval / 2)
            if timestamp >= duration:
                timestamp = max(0, duration - 0.1)
            frame_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{timestamp:.2f}", "-i", video_path,
                 "-frames:v", "1", "-q:v", "3",
                 "-vf", "scale=768:-1",  # downscale to keep tokens reasonable
                 frame_path],
                capture_output=True
            )
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))

        if not frames_b64:
            print("[Frame extraction failed — falling back to text-only]")
            text = input("Text: ").strip()
            return text, None, None

        print(f"[Extracted {len(frames_b64)} frames]")

        # Extract + transcribe audio (best effort — clip may have no speech)
        transcript = ""
        try:
            audio_path = os.path.join(tmp_dir, "audio.wav")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vn",
                 "-ar", "16000", "-ac", "1", audio_path],
                capture_output=True
            )
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                print("[Transcribing audio...]")
                transcript = transcribe_audio(audio_path) or ""
                if transcript.strip():
                    print(f"[Transcript: {transcript[:80]}...]" if len(transcript) > 80
                          else f"[Transcript: {transcript}]")
                else:
                    print("[No speech detected in clip]")
        except Exception as e:
            print(f"[Audio transcription skipped: {e}]")

    finally:
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass

    # Get prompt — voice or text
    print("\nHow do you want to describe this clip to Aria?")
    print("  1. Type a message")
    print("  2. Record voice")
    prompt_method = input("Choice [1-2, default=1]: ").strip()

    if prompt_method == "2":
        dur_in = input("Recording length (sec) [Enter=5]: ").strip()
        dur = float(dur_in) if dur_in else DEFAULT_RECORD_SECONDS
        print("\n>>  Speak when you see 3-2-1")
        import time
        for n in ("3", "2", "1"):
            print(n, end=" ", flush=True)
            time.sleep(1)
        print()
        user_text = record_and_transcribe(dur)
    else:
        user_text = input("Text: ").strip()

    if not user_text:
        user_text = "What do you think of this video?"

    return user_text, frames_b64, transcript
