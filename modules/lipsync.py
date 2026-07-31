# modules/lipsync.py
# -----------------------------------------------------------
#  Stage 3: Lip Sync via PrunaAI P-Video on Replicate
#  Image + audio → lip-synced talking video
#  Auto-chunks audio for clips over 10 seconds
#  Cost: ~$0.02/sec at 720p full quality
# -----------------------------------------------------------

import os
import sys
import time
import wave
import subprocess
import requests
from config import (TEMP_DIR, BASE_VIDEO_DIR, DEBUG, LIPSYNC_PROMPT,
                    LIPSYNC_MAX_RETRIES, LIPSYNC_RETRY_DELAY,
                    ENABLE_TEXT_DETECTION, TEXT_DETECT_MAX_RETRIES,
                    TEXT_DETECT_SAMPLES, TEXT_DETECT_CROP_TOP)


class PVideoLipsyncEngine:
    """
    Lip sync engine using PrunaAI P-Video on Replicate.
    Image + audio → talking video with lip sync.
    Auto-chunks audio > 10s and stitches results.
    """

    CHUNK_SECONDS = 20  # P-Video max duration per call

    def __init__(self, api_token, resolution="720p", draft=False):
        self.api_token = api_token
        self.resolution = resolution
        self.draft = draft
        self._loaded = False
        self._image_url = None  # Cached uploaded image URL
        self._image_name = None

    def load(self):
        if not self.api_token:
            print("[LipSync] ERROR: REPLICATE_API_TOKEN not set")
            return False

        # Verify replicate package
        try:
            import replicate
        except ImportError:
            print("[LipSync] ERROR: replicate not installed")
            print("[LipSync] Run: pip install replicate")
            return False

        mode = "DRAFT" if self.draft else "FULL"
        cost_per_sec = {
            ("720p", True): 0.005,
            ("720p", False): 0.02,
            ("1080p", True): 0.01,
            ("1080p", False): 0.04,
        }.get((self.resolution, self.draft), 0.02)

        print(f"[LipSync] PrunaAI P-Video via Replicate initialized")
        print(f"[LipSync] Mode: {mode} | Resolution: {self.resolution}")
        print(f"[LipSync] Cost: ${cost_per_sec}/sec")
        print(f"[LipSync] Auto-chunking for clips > {self.CHUNK_SECONDS}s")
        self._loaded = True
        return True

    def prepare_video(self, image_name):
        """Verify the base image exists."""
        image_path = os.path.join(BASE_VIDEO_DIR, image_name)
        if not os.path.exists(image_path):
            print(f"[LipSync] Image not found: {image_path}")
            return None
        self._image_name = image_name
        return {"image_path": image_path}

    def generate(self, audio_path, image_name, output_path):
        """
        Generate lip-synced video via P-Video.
        Auto-chunks audio > 10s and stitches results.
        """
        if not self._loaded:
            print("[LipSync] ERROR: Engine not loaded")
            return None

        try:
            import replicate
            os.environ["REPLICATE_API_TOKEN"] = self.api_token

            start = time.time()
            image_path = os.path.join(BASE_VIDEO_DIR, image_name)

            if not os.path.exists(image_path):
                print(f"[LipSync] Image not found: {image_path}")
                return None

            # Get audio duration
            audio_duration = self._get_audio_duration(audio_path)
            cost_per_sec = 0.005 if self.draft else 0.02
            if self.resolution == "1080p":
                cost_per_sec *= 2
            est_cost = audio_duration * cost_per_sec

            mode = "draft" if self.draft else "full"
            print(f"[LipSync] Processing {audio_duration:.1f}s audio "
                  f"({mode} {self.resolution}, est: ${est_cost:.3f})")

            # Split audio into chunks if needed
            if audio_duration > self.CHUNK_SECONDS:
                chunks = self._split_audio(audio_path, audio_duration)
            else:
                chunks = [{"path": audio_path, "start": 0, "end": audio_duration, "duration": audio_duration}]

            print(f"[LipSync] Generating {len(chunks)} chunk(s)...")

            # Generate each chunk
            # Strategy: use last frame from chunk 0 for chunk 1 only
            # (smooth transition at the most visible 20s seam).
            # All other chunks use the original image for maximum quality.
            video_parts = []
            last_frame_path = None
            self._text_retries = {}  # Reset text detection retry tracker

            for i, c in enumerate(chunks):
                chunk_path = c["path"]
                if i > 0:
                    print(f"[LipSync] Waiting 3s between chunks...")
                    time.sleep(1)

                # Chunk 1 uses last frame from chunk 0; all others use original
                if i == 1 and last_frame_path and os.path.exists(last_frame_path):
                    current_image = last_frame_path
                    source_label = "last frame"
                else:
                    current_image = image_path
                    source_label = "original"

                chunk_dur = self._get_audio_duration(chunk_path)
                print(f"[LipSync] Chunk {i+1}/{len(chunks)} "
                      f"({chunk_dur:.1f}s, {source_label})...", end=' ')
                sys.stdout.flush()

                chunk_start = time.time()
                chunk_success = False

                for attempt in range(1, LIPSYNC_MAX_RETRIES + 1):
                    try:
                        with open(current_image, "rb") as img, \
                             open(chunk_path, "rb") as aud:
                            output = replicate.run(
                                "prunaai/p-video",
                                input={
                                    "image": img,
                                    "audio": aud,
                                    "prompt": LIPSYNC_PROMPT,
                                    "resolution": self.resolution,
                                    "draft": self.draft,
                                    "prompt_upsampling": False,
                                    "fps": 24,
                                }
                            )

                        if output:
                            video_url = str(output)
                            part_path = os.path.join(
                                TEMP_DIR, f"pvideo_part_{i}_{int(time.time())}.mp4"
                            )

                            if self._download_file(video_url, part_path):
                                chunk_elapsed = time.time() - chunk_start
                                print(f"done ({chunk_elapsed:.1f}s)")

                                # Check for hallucinated text if enabled
                                if ENABLE_TEXT_DETECTION:
                                    text_result = self._check_for_text(part_path)
                                    if text_result and text_result.get("has_text"):
                                        texts_found = []
                                        for d in text_result.get("detections", []):
                                            texts_found.extend(d.get("texts", []))
                                        text_retries = getattr(self, '_text_retries', {})
                                        retry_count = text_retries.get(i, 0)
                                        if retry_count < TEXT_DETECT_MAX_RETRIES:
                                            text_retries[i] = retry_count + 1
                                            self._text_retries = text_retries
                                            print(f"[LipSync] ⚠️ Text detected: {texts_found}")
                                            print(f"[LipSync] Regenerating chunk "
                                                  f"(text retry {retry_count+1}/{TEXT_DETECT_MAX_RETRIES})...")
                                            try:
                                                os.remove(part_path)
                                            except OSError:
                                                pass
                                            time.sleep(LIPSYNC_RETRY_DELAY)
                                            continue  # retry this chunk
                                        else:
                                            print(f"[LipSync] ⚠️ Text still present after "
                                                  f"{TEXT_DETECT_MAX_RETRIES} retries — using anyway")

                                video_parts.append(part_path)

                                # Only need last frame from chunk 0
                                if i == 0 and len(chunks) > 1:
                                    last_frame_path = self._extract_last_frame(part_path)
                                chunk_success = True
                                break
                            else:
                                print(f"download failed (attempt {attempt}/{LIPSYNC_MAX_RETRIES})")
                        else:
                            print(f"no output (attempt {attempt}/{LIPSYNC_MAX_RETRIES})")

                    except Exception as e:
                        print(f"error: {e} (attempt {attempt}/{LIPSYNC_MAX_RETRIES})")

                    # Retry after delay if not the last attempt
                    if attempt < LIPSYNC_MAX_RETRIES:
                        print(f"[LipSync] Retrying in {LIPSYNC_RETRY_DELAY}s...")
                        time.sleep(LIPSYNC_RETRY_DELAY)

                if not chunk_success:
                    print(f"[LipSync] WARNING: Chunk {i+1} failed after "
                          f"{LIPSYNC_MAX_RETRIES} attempts — skipping")

            # Cleanup extracted frame
            if last_frame_path:
                try: os.remove(last_frame_path)
                except OSError: pass

            # Cleanup audio chunks (but not the original)
            for c in chunks:
                if c["path"] != audio_path:
                    try: os.remove(c["path"])
                    except OSError:
                        pass

            if not video_parts:
                print("[LipSync] ERROR: No video parts generated")
                return None

            # Stitch video parts if multiple
            if len(video_parts) > 1:
                print(f"[LipSync] Stitching {len(video_parts)} parts...")
                if not self._stitch_videos(video_parts, output_path):
                    print("[LipSync] ERROR: Stitching failed")
                    return None
            else:
                os.rename(video_parts[0], output_path)

            # Cleanup part files
            for part_path in video_parts:
                if os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass

            elapsed = time.time() - start
            print(f"[LipSync] Done! {audio_duration:.1f}s video "
                  f"in {elapsed:.1f}s (cost: ~${est_cost:.3f})")

            return output_path

        except Exception as e:
            print(f"[LipSync] Generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _check_for_text(self, video_path):
        """Run text detection on a video chunk. Returns detection result dict."""
        try:
            from modules.text_detect import detect_text_in_video
            return detect_text_in_video(
                video_path,
                sample_count=TEXT_DETECT_SAMPLES,
                crop_top_pct=TEXT_DETECT_CROP_TOP,
            )
        except ImportError:
            return None
        except Exception as e:
            print(f"[LipSync] Text detection error: {e}")
            return None

    def _extract_last_frame(self, video_path):
        """Extract the last frame of a video as a PNG image."""
        frame_path = os.path.join(TEMP_DIR, f"lastframe_{int(time.time())}.png")
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", frame_path],
                capture_output=True, text=True
            )
            if result.returncode == 0 and os.path.exists(frame_path):
                return frame_path
            else:
                print(f"[LipSync] Last frame extraction failed, "
                      f"falling back to original image")
                return None
        except Exception as e:
            print(f"[LipSync] Frame extraction error: {e}")
            return None

    def _split_audio(self, audio_path, total_duration):
        """Split audio into chunks. Ensures final chunk is >= 2 seconds."""
        max_chunk = self.CHUNK_SECONDS
        num_full = int(total_duration // max_chunk)
        remainder = total_duration - (num_full * max_chunk)

        # Build chunk durations list
        durations = [max_chunk] * num_full
        if remainder > 0:
            durations.append(remainder)

        # If last chunk is too short, steal from the one before it
        if len(durations) > 1 and durations[-1] < 2.0:
            steal = 2.0 - durations[-1]
            durations[-2] -= steal
            durations[-1] += steal

        chunks = []
        pos = 0.0
        for i, dur in enumerate(durations):
            cp = os.path.join(TEMP_DIR, f"achunk_{i}_{int(time.time())}.wav")
            subprocess.run(["ffmpeg", "-y", "-i", audio_path,
                "-ss", f"{pos:.3f}", "-t", f"{dur:.3f}",
                "-c:a", "pcm_s16le", "-ar", "24000", cp],
                capture_output=True)
            if os.path.exists(cp):
                ad = self._get_audio_duration(cp)
                if ad >= 1.0:
                    chunks.append({"path": cp, "start": pos, "end": pos + dur, "duration": ad})
            pos += dur

        return chunks

    def _stitch_videos(self, video_paths, output_path):
        """Concatenate multiple video files using FFmpeg."""
        list_path = os.path.join(TEMP_DIR, "concat_list.txt")

        with open(list_path, 'w') as f:
            for vp in video_paths:
                abs_path = os.path.abspath(vp).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")

        # Try concat without re-encoding first (fastest)
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Fallback: re-encode for compatibility
            print("[LipSync] Re-encoding for stitch compatibility...")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                "-movflags", "+faststart",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

        # Cleanup
        try:
            os.remove(list_path)
        except OSError:
            pass

        return os.path.exists(output_path)

    def _download_file(self, url, output_path):
        """Download result video."""
        try:
            response = requests.get(url, stream=True, timeout=180)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
                return True
            else:
                print(f"[LipSync] Download error: {response.status_code}")
                return False
        except Exception as e:
            print(f"[LipSync] Download error: {e}")
            return False

    def _get_audio_duration(self, audio_path):
        try:
            with wave.open(audio_path, 'rb') as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            try:
                cmd = ["ffprobe", "-v", "error", "-show_entries",
                       "format=duration", "-of", "csv=p=0", audio_path]
                result = subprocess.run(cmd, capture_output=True,
                                        text=True, check=True)
                return float(result.stdout.strip())
            except Exception:
                return 20.0


def lipsync_worker(audio_file_queue, lipsync_ready_queue,
                   get_conversation_id, selected_video_name,
                   lipsync_engine):
    """Lip sync worker thread."""
    print("[LipSync Worker] Started")
    processed_files = set()

    while True:
        audio_file = audio_file_queue.get()
        if audio_file is None:
            print("[LipSync Worker] Shutting down")
            break

        if audio_file in processed_files:
            audio_file_queue.task_done()
            continue

        processed_files.add(audio_file)

        try:
            image_name = selected_video_name()
            timestamp = int(time.time())
            output_path = os.path.join(TEMP_DIR, f"lipsync_{timestamp}.mp4")

            print(f"[LipSync Worker] Processing with {image_name}")

            result = lipsync_engine.generate(
                audio_file, image_name, output_path
            )

            if result and os.path.exists(result):
                lipsync_ready_queue.put(result)
                print(f"[LipSync Worker] Video ready: {result}")
            else:
                print("[LipSync Worker] ERROR: Generation failed")

        except Exception as e:
            print(f"[LipSync Worker] Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except OSError:
                pass
            audio_file_queue.task_done()
