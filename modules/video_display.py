# modules/video_display.py
# -----------------------------------------------------------
#  Stage 4: Video display, library management, and live mode
#  Handles VLC playback, video preservation, and the live UI.
# -----------------------------------------------------------

import os
import sys
import time
import shutil
import platform
import subprocess
from queue import Empty
from datetime import datetime
from config import (
    VIDEO_LIBRARY_DIR, PRESERVE_VIDEOS, TEMP_DIR,
    USE_SYSTEM_DEFAULT_PLAYER, PREFERRED_PLAYER, FADE_DURATION,
    INFLUENCER_NAME, DEBUG
)


def generate_video_name(prefix="zara_response"):
    """Generate a unique timestamped name for a video."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.mp4"


def find_vlc():
    """Auto-detect VLC installation."""
    vlc_paths = [
        r"C:\Program Files\VideoLAN\VLC\vlc.exe",
        r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
        "/Applications/VLC.app/Contents/MacOS/VLC",
        "vlc",  # Linux PATH
    ]
    for path in vlc_paths:
        if os.path.exists(path):
            return path
    return None


def configure_video_player():
    """Configure the video player. Returns (player_path, use_system_default)."""
    global PREFERRED_PLAYER, USE_SYSTEM_DEFAULT_PLAYER

    print("\nVideo Player Configuration:")
    print("1. Use VLC media player (recommended)")
    print("2. Use system default player")
    print("3. Specify custom player path")

    choice = input("Choose player option [1-3, default=1]: ").strip()

    if choice == "2":
        USE_SYSTEM_DEFAULT_PLAYER = True
        print("[Will use system default player]")
    elif choice == "3":
        custom_path = input("Enter full path to video player: ").strip()
        PREFERRED_PLAYER = custom_path
        USE_SYSTEM_DEFAULT_PLAYER = False
        print(f"[Using custom player: {custom_path}]")
    else:
        vlc_path = find_vlc()
        if vlc_path:
            PREFERRED_PLAYER = vlc_path
            USE_SYSTEM_DEFAULT_PLAYER = False
            print(f"[Found VLC at: {vlc_path}]")
        else:
            print("[VLC not found, using system default]")
            USE_SYSTEM_DEFAULT_PLAYER = True


def play_video(video_path):
    """Play a video using the configured player."""
    try:
        video_path = os.path.abspath(video_path)
        print(f"[Opening video: {video_path}]")

        if USE_SYSTEM_DEFAULT_PLAYER:
            _play_system_default(video_path)
        else:
            _play_with_player(video_path)

        return True

    except Exception as e:
        print(f"[Error playing video: {e}]")
        # Last resort fallback
        try:
            _play_system_default(video_path)
            return True
        except Exception:
            print("[All player methods failed]")
            return False


def _play_system_default(video_path):
    """Open with system default player."""
    system = platform.system()
    if system == "Windows":
        os.startfile(video_path)
    elif system == "Darwin":
        subprocess.Popen(['open', video_path])
    else:
        subprocess.Popen(['xdg-open', video_path])


def _play_with_player(video_path):
    """Open with the configured player."""
    player = PREFERRED_PLAYER

    if player == "auto":
        vlc_path = find_vlc()
        if vlc_path:
            player = vlc_path
        else:
            _play_system_default(video_path)
            return

    if not os.path.exists(player) and player != "vlc":
        print(f"[Player not found: {player}, falling back to system default]")
        _play_system_default(video_path)
        return

    if "vlc" in player.lower():
        cmd = [player, "--no-video-title-show", "--no-loop", "--no-repeat", "--play-and-exit", video_path]
        if os.name == 'nt':
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        else:
            subprocess.Popen(cmd, start_new_session=True)
    else:
        subprocess.Popen([player, video_path])


def list_available_videos():
    """List all preserved videos in the library."""
    os.makedirs(VIDEO_LIBRARY_DIR, exist_ok=True)
    videos = []
    try:
        for filename in sorted(os.listdir(VIDEO_LIBRARY_DIR)):
            if filename.endswith(".mp4"):
                path = os.path.join(VIDEO_LIBRARY_DIR, filename)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                videos.append((filename, path, size_mb))
    except Exception as e:
        print(f"[Error listing videos: {e}]")
    return videos


def preserve_video(video_path):
    """
    Copy a video to the library directory.
    Returns the library path, or None on failure.
    """
    if not PRESERVE_VIDEOS:
        return None

    try:
        os.makedirs(VIDEO_LIBRARY_DIR, exist_ok=True)
        video_name = generate_video_name()
        library_path = os.path.join(VIDEO_LIBRARY_DIR, video_name)
        shutil.copy2(video_path, library_path)
        print(f"[Video preserved: {library_path}]")
        return library_path
    except Exception as e:
        print(f"[Error preserving video: {e}]")
        return None


def video_display_monitor(lipsync_ready_queue, processing_complete,
                          set_latest_paths):
    """
    Monitor thread: watches for completed lip-sync videos,
    plays them, and preserves to library.

    Args:
        lipsync_ready_queue: Queue of completed MP4 paths
        processing_complete: threading.Event to signal completion
        set_latest_paths: Callable(video_path, library_path) to store paths
    """
    print("[Display] Monitor started")
    sys.stdout.flush()

    while True:
        try:
            video_path = lipsync_ready_queue.get(timeout=1)

            print(f"\n[Display] Video ready: {video_path}")

            # Signal processing complete
            processing_complete.set()

            # Preserve to library
            library_path = preserve_video(video_path)

            # Store paths for main thread access
            set_latest_paths(video_path, library_path)

            # Play video
            play_video(video_path)

            lipsync_ready_queue.task_done()

        except Empty:
            # Normal timeout, keep polling
            pass
        except Exception as e:
            print(f"[Display] Monitor error: {e}")
            time.sleep(1)


def live_mode_interface():
    """
    Interactive interface for managing videos during live sessions.
    Returns an action string: 'voice_record', 'text_input', or 'exit'.
    """
    print("\n" + "=" * 60)
    print(f"{INFLUENCER_NAME} Live Mode")
    print("=" * 60)

    while True:
        print("\nOptions:")
        print("1. List available videos")
        print("2. Play a video")
        print("3. Record a new voice response")
        print("4. Enter a text question")
        print("5. Send an image")
        print("6. Exit live mode")

        choice = input("\nEnter choice [1-6]: ").strip()

        if choice == "1":
            videos = list_available_videos()
            if not videos:
                print("[No videos in library]")
            else:
                print("\nAvailable Videos:")
                for i, (name, path, size) in enumerate(videos, 1):
                    print(f"  {i}. {name} ({size:.1f} MB)")

        elif choice == "2":
            videos = list_available_videos()
            if not videos:
                print("[No videos to play]")
                continue
            print("\nSelect a video:")
            for i, (name, path, size) in enumerate(videos, 1):
                print(f"  {i}. {name} ({size:.1f} MB)")
            try:
                idx = int(input("\nVideo number: ").strip()) - 1
                if 0 <= idx < len(videos):
                    play_video(videos[idx][1])
                else:
                    print("[Invalid selection]")
            except ValueError:
                print("[Invalid input]")

        elif choice == "3":
            return "voice_record"

        elif choice == "4":
            return "text_input"

        elif choice == "5":
            return "image_input"

        elif choice == "6":
            return "exit"

        else:
            print("[Invalid choice]")
