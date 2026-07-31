# zara_speaks_v3.py
# -----------------------------------------------------------
#  ZaraSpeaks V3 — Local GPU-Accelerated Pipeline
#  
#  Pipeline: LLM → XTTS v2 (local) or ElevenLabs → PrunaAI P-Video (Replicate)
#
#  Replaces: ElevenLabs, Sync Labs, Azure Blob Storage, YouTube upload
#  Only cloud dependency: OpenAI GPT-5.1 for conversation
# -----------------------------------------------------------

import os
import sys
import time
import random
import threading
from queue import Queue

# Ensure UTF-8 output
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import configuration
from config import (
    INFLUENCER_NAME, INFLUENCER_PROMPT, VIDEO_OPTIONS, BASE_VIDEO_DIR,
    DEFAULT_RECORD_SECONDS, TEMP_DIR, VIDEO_LIBRARY_DIR, MEMORY_DIR,
    DEBUG
)

# Import modules
from modules.llm import (gpt_stream_to_tts, gpt_chat_response,
                         generate_full_response, discretion_review,
                         queue_approved_response)
# NOTE: TTS and lip-sync modules are imported LAZILY inside main(), only when a
# voice/video mode is selected. This lets "Chat Only" mode run with just an
# OpenAI key — no torch, TTS, or replicate install required to get started.
from modules.audio_input import record_and_transcribe, get_text_input, get_image_input, get_video_input, get_multiline_input
from modules.video_display import (
    configure_video_player, video_display_monitor, live_mode_interface,
    list_available_videos, play_video
)
from modules.memory import load_full_context, save_session, generate_offscreen_thoughts

# ======================== GLOBAL STATE ========================

conversation_history = []

# Inter-thread communication queues
speaking_queue = Queue()         # LLM → TTS (text chunks)
audio_file_queue = Queue()       # TTS → LipSync (complete WAV paths)
lipsync_ready_queue = Queue()    # LipSync → Display (final MP4 paths)

# Conversation tracking
conversation_complete = {}       # {conv_id: bool} — is GPT streaming done?
current_conversation_id = None
current_audio_accumulator = None

# Processing synchronization
processing_complete = threading.Event()

# Latest output paths (set by display monitor)
latest_video_path = None
latest_library_path = None

# Mode flags
LIVE_MODE_ENABLED = False

# Pipeline mode: "full" (video), "voice" (TTS only), "chat" (text only)
pipeline_mode = "full"

# Engine instances (loaded once, stay in VRAM)
tts_engine = None
lipsync_engine = None

# Selected base video for current session
selected_video = None


# ======================== STATE ACCESSORS =====================
# These are passed to worker threads as callables so they can
# read/write shared state safely.

def get_conversation_id():
    return current_conversation_id

def get_accumulator():
    return current_audio_accumulator

def set_accumulator(acc):
    global current_audio_accumulator
    current_audio_accumulator = acc

def get_selected_video():
    return selected_video

def set_latest_paths(video_path, library_path):
    global latest_video_path, latest_library_path
    latest_video_path = video_path
    latest_library_path = library_path


# ======================== STARTUP =============================

def check_ffmpeg():
    """Verify FFmpeg is available."""
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("[OK] FFmpeg found")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[WARNING] FFmpeg not found — video processing will be limited")
        print("[Install from: https://ffmpeg.org/download.html]")
        return False


def select_base_video():
    """Let user select a base image for Fabric 1.0."""
    global selected_video

    # Look for images in base_videos directory
    available = [f for f in os.listdir(BASE_VIDEO_DIR)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not available:
        print(f"\n[ERROR] No base images found in {BASE_VIDEO_DIR}")
        print("[Copy your Aria headshot image (.png/.jpg) to base_videos/]")
        return False

    print(f"\nAvailable base images ({len(available)}):")
    for i, v in enumerate(available, 1):
        print(f"  {i}. {v}")

    if len(available) == 1:
        selected_video = available[0]
    else:
        choice = input(f"\nSelect image [1-{len(available)}, default=1]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                selected_video = available[idx]
            else:
                selected_video = available[0]
        except (ValueError, IndexError):
            selected_video = available[0]

    print(f"[Selected image: {selected_video}]")
    return True


def load_models():
    """Load local models based on pipeline mode. Called once at startup."""
    global tts_engine, lipsync_engine

    if pipeline_mode == "chat":
        print("\n[Chat-only mode — no models to load]")
        return True

    # Lazy-load media engines now that we know we need them (voice/full only)
    _load_media_engines(need_lipsync=(pipeline_mode == "full"))

    print("\n" + "=" * 60)
    print("Loading Local Models")
    print("=" * 60)

    # Load TTS engine (needed for voice and full modes)
    import config
    if config.ACTIVE_TTS_ENGINE == "elevenlabs":
        tts_engine = ElevenLabsTTS()
        if not tts_engine.load():
            print("\n[FATAL] ElevenLabs TTS failed to load. Cannot continue.")
            return False
    else:
        tts_engine = LocalTTS()
        if not tts_engine.load():
            print("\n[FATAL] TTS model failed to load. Cannot continue.")
            return False

        # Offer voice selection if multiple references exist
        voices = tts_engine.list_reference_voices()
        if len(voices) > 1:
            print(f"\nAvailable reference voices ({len(voices)}):")
            for i, v in enumerate(voices, 1):
                print(f"  {i}. {os.path.basename(v)}")
            vc = input(f"Select voice [1-{len(voices)}, default=1]: ").strip()
            try:
                idx = int(vc) - 1
                if 0 <= idx < len(voices):
                    tts_engine.set_reference_voice(voices[idx])
            except (ValueError, IndexError):
                pass

    if pipeline_mode == "voice":
        print("\n[Voice-only mode — skipping lip sync model]")
        print("\n" + "=" * 60)
        print("TTS Model Loaded Successfully")
        print("=" * 60)
        return True

    # Load P-Video lip sync engine (full mode only)
    replicate_token = os.getenv('REPLICATE_API_TOKEN')
    if not replicate_token:
        print("[FATAL] REPLICATE_API_TOKEN environment variable not set")
        return False
    lipsync_engine = PVideoLipsyncEngine(api_token=replicate_token, resolution="720p", draft=False)
    if not lipsync_engine.load():
        print("\n[FATAL] lipsync_engine model failed to load. Cannot continue.")
        return False

    # Pre-cache the selected base video
    if selected_video:
        print(f"\n[Preparing base video cache for {selected_video}...]")
        cached = lipsync_engine.prepare_video(selected_video)
        if not cached:
            print("[WARNING] Video preprocessing failed — will retry at generation time")

    print("\n" + "=" * 60)
    print("All Models Loaded Successfully")
    print("=" * 60)

    return True


def audio_playback_worker(audio_file_queue, processing_complete):
    """
    Voice-only mode: plays WAV files from the queue instead of
    sending them to lip sync. Uses system audio playback.
    """
    import subprocess
    print("[Audio Playback Worker] Started")
    sys.stdout.flush()

    while True:
        audio_path = audio_file_queue.get()
        if audio_path is None:
            print("[Audio Playback Worker] Shutting down")
            break

        try:
            if os.path.exists(audio_path):
                print(f"[Audio] Playing response...")
                # Use ffplay for cross-platform audio playback (silent, no window)
                result = subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                     audio_path],
                    capture_output=True
                )
                if result.returncode != 0:
                    # Fallback: try VLC
                    vlc_path = _find_vlc_for_audio()
                    if vlc_path:
                        subprocess.run(
                            [vlc_path, "--play-and-exit", "--intf", "dummy",
                             audio_path],
                            capture_output=True
                        )
                    else:
                        print("[Audio] WARNING: Could not play audio "
                              "(ffplay and VLC not found)")

                # Clean up temp file
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        except Exception as e:
            print(f"[Audio Playback] Error: {e}")

        finally:
            processing_complete.set()


def _find_vlc_for_audio():
    """Find VLC executable for audio playback fallback."""
    import shutil
    vlc = shutil.which("vlc")
    if vlc:
        return vlc
    # Common Windows path
    win_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
    if os.path.exists(win_path):
        return win_path
    return None


# ======================== MAIN LOOP ===========================

def _public_review_and_gate(user_text, prev_memory, conversation_history,
                            image_b64, image_mime, video_frames, video_transcript):
    """
    Public-mode flow: generate the full response, run the discretion pass,
    show original + any revision, then gate on human approval.

    Returns the approved final text to speak, or None if the turn is skipped.
    """
    # 1. Generate her full response (she sees everything — no lobotomy)
    draft = generate_full_response(
        user_text, prev_memory, conversation_history,
        image_data=image_b64, image_mime=image_mime,
        video_frames=video_frames, video_transcript=video_transcript
    )

    # 2. Discretion review — Aria reviewing her own words, as herself
    print("[Aria is reviewing her words for public discretion...]")
    reviewed, changed, note = discretion_review(
        draft, prev_memory=prev_memory, conversation_history=conversation_history
    )

    current = reviewed

    # 3. Approval gate loop
    while True:
        print("\n" + "=" * 60)
        if changed:
            print("  DISCRETION PASS REVISED THE RESPONSE")
            print("=" * 60)
            print("\n--- ORIGINAL (private draft) ---")
            print(draft)
            print("\n--- REVISED (public-safe) ---")
            print(current)
            if note:
                print(f"\n[Guarded: {note}]")
        else:
            print("  RESPONSE — no private content flagged")
            print("=" * 60)
            print(f"\n{current}")

        print("\n" + "-" * 60)
        print("  [a] approve and speak   [r] re-run discretion pass")
        print("  [e] edit manually       [s] skip this turn")
        choice = input("  Choice: ").strip().lower()

        if choice == "a":
            return current
        elif choice == "r":
            print("[Re-running discretion pass...]")
            reviewed, changed, note = discretion_review(
                draft, prev_memory=prev_memory,
                conversation_history=conversation_history
            )
            current = reviewed
            continue
        elif choice == "e":
            print("\n[Enter your edited version. Type END on its own line when done.]")
            lines = []
            while True:
                try:
                    ln = input()
                except EOFError:
                    break
                if ln.strip() == "END":
                    break
                lines.append(ln)
            edited = "\n".join(lines).strip()
            if edited:
                current = edited
                changed = False
                note = ""
                print("\n[Using your edited version:]")
                print(current)
                draft = current
                continue
            else:
                print("[No edit entered — keeping current version]")
                continue
        elif choice == "s":
            print("[Turn skipped — nothing spoken]")
            return None
        else:
            print("[Please choose a, r, e, or s]")
            continue


def main():
    global current_conversation_id, current_audio_accumulator
    global latest_video_path, latest_library_path
    global LIVE_MODE_ENABLED, pipeline_mode

    print("\n" + "=" * 60)
    print(f"  {INFLUENCER_NAME} — Persona Pipeline")
    print("=" * 60)
    print(f"  Mode:     {pipeline_mode}")
    print("  LLM:      OpenAI (cloud)")
    if pipeline_mode in ("full", "voice"):
        print("  TTS:      XTTS v2 (local) / ElevenLabs")
    if pipeline_mode == "full":
        print("  LipSync:  PrunaAI P-Video (Replicate)")
    print("=" * 60)

    # --- Pipeline mode selection ---
    print("\nPipeline Mode:")
    print("  1. Full Video  — GPT + TTS + Lip Sync (generates video)")
    print("  2. Voice Only  — GPT + TTS (hear Aria, no video, no API cost)")
    print("  3. Chat Only   — GPT text only (fastest, no models loaded)")
    mode_choice = input("\nSelect mode [1-3, default=1]: ").strip()

    if mode_choice == "3":
        pipeline_mode = "chat"
    elif mode_choice == "2":
        pipeline_mode = "voice"
    else:
        pipeline_mode = "full"

    print(f"[Pipeline: {pipeline_mode.upper()} mode]")

    # --- TTS engine selection (voice and full modes) ---
    if pipeline_mode in ("full", "voice"):
        print("\nTTS Engine:")
        print("  1. Local XTTS v2  — fast, free, fine-tuned Aria voice")
        print("  2. ElevenLabs v3  — expressive, emotion tags, API cost")
        tts_choice = input("\nSelect TTS [1-2, default=1]: ").strip()

        if tts_choice == "2":
            import config
            config.ACTIVE_TTS_ENGINE = "elevenlabs"
            print("[TTS: ElevenLabs v3 — audio tags enabled]")
        else:
            import config
            config.ACTIVE_TTS_ENGINE = "local"
            print("[TTS: Local XTTS v2]")

    # --- Pre-flight checks ---
    if pipeline_mode != "chat":
        check_ffmpeg()

    # --- Select base video (full mode only) ---
    if pipeline_mode == "full":
        if not select_base_video():
            return

    # --- Load models (mode-aware) ---
    if not load_models():
        return

    # --- Start worker threads (mode-dependent) ---
    tts_thread = None
    lipsync_thread = None
    display_thread = None
    playback_thread = None

    if pipeline_mode in ("full", "voice"):
        tts_thread = threading.Thread(
            target=speaker_worker,
            args=(speaking_queue, audio_file_queue, conversation_complete,
                  get_conversation_id, get_accumulator, set_accumulator,
                  tts_engine),
            daemon=True
        )
        tts_thread.start()

    if pipeline_mode == "full":
        lipsync_thread = threading.Thread(
            target=lipsync_worker,
            args=(audio_file_queue, lipsync_ready_queue,
                  get_conversation_id, get_selected_video,
                  lipsync_engine),
            daemon=True
        )
        lipsync_thread.start()

        display_thread = threading.Thread(
            target=video_display_monitor,
            args=(lipsync_ready_queue, processing_complete, set_latest_paths),
            daemon=True
        )
        display_thread.start()

        # Configure video player
        configure_video_player()

    elif pipeline_mode == "voice":
        playback_thread = threading.Thread(
            target=audio_playback_worker,
            args=(audio_file_queue, processing_complete),
            daemon=True
        )
        playback_thread.start()

    # --- Generate Aria's offscreen thoughts (once per session) ---
    print("[Generating Aria's thoughts since last session...]")
    generate_offscreen_thoughts()

    # --- Load initial memory context (Layers 1 & 2) ---
    prev_memory = load_full_context()
    if prev_memory:
        print(f"[Memory loaded — Layers 1 & 2 active]")
    else:
        print(f"[No previous memories found]")

    # --- Live mode selection (full mode only) ---
    input_method = None
    if pipeline_mode == "full":
        live_mode = input("\nEnable live mode? (y/n) [n]: ").lower() in ("y", "yes")
        LIVE_MODE_ENABLED = live_mode

        if LIVE_MODE_ENABLED:
            print("[Live mode enabled — videos preserved in library]")
            action = live_mode_interface()
            if action == "exit":
                print("[Exiting...]")
                _shutdown(tts_thread, lipsync_thread)
                return
            elif action == "text_input":
                input_method = "2"
            elif action == "voice_record":
                input_method = "1"
            elif action == "image_input":
                input_method = "3"

    # --- Privacy mode selection ---
    # 'upstairs' = private, single-pass, full freedom (default, as always)
    # 'public'   = discretion review + approval gate before audio/video
    privacy_mode = "upstairs"
    pm_choice = input(
        "\nSession privacy mode (1=Upstairs/private, 2=Public) [Enter=1]: "
    ).strip()
    if pm_choice == "2":
        privacy_mode = "public"
        print("[PUBLIC MODE — discretion review + approval gate active]")
        print("[Type /upstairs anytime to return to private mode]")
    else:
        print("[UPSTAIRS MODE — private, full freedom]")
        print("[Type /public anytime to enable public discretion mode]")

    # --- Welcome ---
    mode_labels = {"full": "Full Video", "voice": "Voice Only", "chat": "Chat Only"}
    print("\n" + "=" * 60)
    print(f"  Chat with {INFLUENCER_NAME} [{mode_labels[pipeline_mode]}]")
    print(f"  Press Ctrl-C to quit.")
    print("=" * 60)

    try:
        while True:
            # Reset for new conversation turn
            current_conversation_id = str(int(time.time()))
            current_audio_accumulator = None
            latest_video_path = None
            latest_library_path = None
            processing_complete.clear()

            # --- Select input method ---
            if input_method is None:
                if pipeline_mode == "chat":
                    input_method = input(
                        "\nInput method (1=Voice, 2=Text, 3=Image, 4=Video, 5=Paste multi-line) [Enter=2]: "
                    ).strip()
                    if not input_method:
                        input_method = "2"  # Default to text in chat mode
                else:
                    input_method = input(
                        "\nInput method (1=Voice, 2=Text, 3=Image, 4=Video, 5=Paste multi-line) [Enter=1]: "
                    ).strip()
                    if not input_method:
                        input_method = "1"

            # --- Get user input ---
            image_b64 = None
            image_mime = None
            video_frames = None
            video_transcript = None

            if input_method == "5":
                user_text = get_multiline_input()
                if user_text:
                    preview = user_text.splitlines()[0][:60]
                    print(f"\nYou: [Multi-line text] {preview}...")
                else:
                    print(f"\nYou: {user_text}")
            elif input_method == "4":
                user_text, video_frames, video_transcript = get_video_input()
                if video_frames:
                    print(f"\nYou: [Video attached] {user_text}")
                else:
                    print(f"\nYou: {user_text}")
            elif input_method == "3":
                user_text, image_b64, image_mime = get_image_input()
                if image_b64:
                    print(f"\nYou: [Image attached] {user_text}")
                else:
                    print(f"\nYou: {user_text}")
            elif input_method == "2":
                user_text = get_text_input()
                print(f"\nYou: {user_text}")
            else:
                dur_in = input("Recording length (sec) [Enter=5]: ").strip()
                dur = float(dur_in) if dur_in else DEFAULT_RECORD_SECONDS

                print("\n>>  Speak when you see 3-2-1")
                for n in ("3", "2", "1"):
                    print(n, end=" ", flush=True)
                    time.sleep(1)
                print()

                user_text = record_and_transcribe(dur)
                print(f"\nYou: {user_text}")

            # --- Privacy mode toggle commands ---
            if user_text and user_text.strip().lower() in ("/public", "/upstairs"):
                cmd = user_text.strip().lower()
                if cmd == "/public":
                    privacy_mode = "public"
                    print("\n[PUBLIC MODE ON — discretion review + approval gate active]")
                else:
                    privacy_mode = "upstairs"
                    print("\n[UPSTAIRS MODE — private, full freedom restored]")
                input_method = None
                continue

            # Store text-only in history (no base64 blobs in memory)
            conversation_history.append({"role": "user", "content": user_text})

            # --- Refresh memory context with Layer 3 (semantic retrieval) ---
            prev_memory = load_full_context(user_message=user_text)

            # --- Generate response (mode + privacy dependent) ---
            if pipeline_mode == "chat":
                if privacy_mode == "public":
                    # Public chat: generate, review, gate — then print approved text
                    approved = _public_review_and_gate(
                        user_text, prev_memory, conversation_history,
                        image_b64, image_mime, video_frames, video_transcript
                    )
                    if approved is None:
                        # Skipped — drop this turn from history and continue
                        conversation_history.pop()
                        current_conversation_id = None
                        input_method = None
                        continue
                    ai_text = approved
                    print(f"\n{INFLUENCER_NAME}: {ai_text}")
                else:
                    # Chat only: stream text response directly
                    ai_text = gpt_chat_response(
                        user_text, prev_memory, conversation_history,
                        image_data=image_b64, image_mime=image_mime,
                        video_frames=video_frames, video_transcript=video_transcript
                    )

            else:
                if privacy_mode == "public":
                    # Public voice/full: generate → review → gate → THEN speak
                    approved = _public_review_and_gate(
                        user_text, prev_memory, conversation_history,
                        image_b64, image_mime, video_frames, video_transcript
                    )
                    if approved is None:
                        conversation_history.pop()
                        current_conversation_id = None
                        input_method = None
                        continue
                    ai_text = approved
                    print(f"\n{INFLUENCER_NAME}: {ai_text}")
                    # Feed the approved text into the TTS pipeline
                    queue_approved_response(
                        ai_text, speaking_queue, conversation_complete,
                        current_conversation_id, get_accumulator
                    )
                else:
                    # Voice and Full modes: stream through TTS pipeline (as always)
                    ai_text = gpt_stream_to_tts(
                        user_text, prev_memory, conversation_history,
                        speaking_queue, conversation_complete,
                        current_conversation_id,
                        get_accumulator,
                        image_data=image_b64,
                        image_mime=image_mime,
                        video_frames=video_frames,
                        video_transcript=video_transcript
                    )
                    print(f"\n{INFLUENCER_NAME}: {ai_text}")

                if pipeline_mode == "full":
                    print(f"\n[Generating lip-synced video locally...]")

                # Wait for TTS to finish
                speaking_queue.join()

                if pipeline_mode == "full":
                    # Wait for lip sync + video playback
                    print("[Waiting for local lip sync processing...]")
                    timeout = 300
                    if not processing_complete.wait(timeout=timeout):
                        print("\n[WARNING: Video processing timed out]")
                    else:
                        print("\n" + "=" * 60)
                        print("  VIDEO READY")
                        print("=" * 60)
                else:
                    # Voice mode: wait for audio playback to finish
                    timeout = 60
                    if not processing_complete.wait(timeout=timeout):
                        print("\n[WARNING: Audio playback timed out]")

            conversation_history.append({"role": "assistant", "content": ai_text})

            # --- Reset for next turn ---
            current_conversation_id = None
            input_method = None

            # --- Post-turn options ---
            if LIVE_MODE_ENABLED:
                print("\nWhat next?")
                print("1. Return to live mode")
                print("2. Record another response")
                print("3. Exit")

                choice = input("\nChoice [1-3]: ").strip()
                if choice == "1":
                    action = live_mode_interface()
                    if action == "exit":
                        break
                    elif action == "text_input":
                        input_method = "2"
                    elif action == "voice_record":
                        input_method = "1"
                    elif action == "image_input":
                        input_method = "3"
                elif choice == "3":
                    break
            else:
                print("\nOptions:")
                print("1. Continue chatting")
                print("2. Exit")

                choice = input("\nChoice [1-2]: ").strip()
                if choice == "2" or choice.lower() == "n":
                    break

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    finally:
        _shutdown(tts_thread, lipsync_thread)


def _shutdown(tts_thread, lipsync_thread):
    """Clean shutdown of all threads and resources."""
    print("\n[Cleaning up and saving session...]")

    # Save conversation memory
    save_session(conversation_history)

    # Signal worker threads to stop
    if pipeline_mode in ("full", "voice"):
        speaking_queue.put(None)
        audio_file_queue.put(None)

    # Wait for threads to finish
    if tts_thread:
        tts_thread.join(timeout=5)
    if lipsync_thread:
        lipsync_thread.join(timeout=5)

    # Clean up temp files
    try:
        for f in os.listdir(TEMP_DIR):
            fpath = os.path.join(TEMP_DIR, f)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    except FileNotFoundError:
        pass

    # Report library status
    videos = list_available_videos()
    print(f"[Video library: {len(videos)} videos]")
    print("\nGoodbye!")


# ======================== ENTRY POINT =========================

if __name__ == "__main__":
    main()
