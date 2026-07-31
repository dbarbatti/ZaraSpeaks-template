# modules/text_detect.py
# -----------------------------------------------------------
#  Text detection for P-Video lip sync output
#
#  Samples frames from a video chunk and runs OCR on the
#  lower portion (where P-Video tends to hallucinate text).
#  Returns True if text is detected, False if clean.
#
#  Requires: pip install easyocr
# -----------------------------------------------------------

import os
import sys
import subprocess
import tempfile
import shutil

# Lazy-load EasyOCR reader
_ocr_reader = None


def _load_ocr():
    """Lazy-load EasyOCR reader (first call downloads model ~100MB)."""
    global _ocr_reader
    if _ocr_reader is not None:
        return _ocr_reader
    try:
        import easyocr
        print("[TextDetect] Loading OCR model...")
        _ocr_reader = easyocr.Reader(
            ["en"],
            gpu=True,
            verbose=False
        )
        print("[TextDetect] OCR model loaded")
        return _ocr_reader
    except ImportError:
        print("[TextDetect] EasyOCR not installed — text detection disabled")
        print("[TextDetect] Install with: pip install easyocr")
        return None


def detect_text_in_video(video_path, sample_count=4, crop_top_pct=0.0,
                         min_confidence=0.5, min_detections=1):
    """
    Check a video for hallucinated text by sampling frames and
    running OCR on the lower portion of each frame.

    Args:
        video_path: Path to the MP4 file to check
        sample_count: Number of frames to sample (evenly spaced)
        crop_top_pct: Crop away the top N% of frame (0.55 = check bottom 45%)
                      This avoids false positives from mouth/face movement
        min_confidence: Minimum OCR confidence to count as a detection
        min_detections: Minimum number of frames with text to flag the video

    Returns:
        dict with:
            "has_text": bool — True if text was detected
            "detections": list of {frame, texts, confidences}
            "frames_checked": int
            "frames_with_text": int
    """
    reader = _load_ocr()
    if reader is None:
        return {"has_text": False, "detections": [], "frames_checked": 0,
                "frames_with_text": 0, "error": "OCR not available"}

    if not os.path.exists(video_path):
        return {"has_text": False, "detections": [], "frames_checked": 0,
                "frames_with_text": 0, "error": f"File not found: {video_path}"}

    # Create temp directory for extracted frames
    tmp_dir = tempfile.mkdtemp(prefix="textdetect_")

    try:
        # Get video duration
        duration = _get_duration(video_path)
        if duration <= 0:
            return {"has_text": False, "detections": [], "frames_checked": 0,
                    "frames_with_text": 0, "error": "Could not read video duration"}

        # Extract frames evenly spaced through the video
        # Skip very start and end (0.5s buffer)
        start = 0.5
        end = max(duration - 0.5, start + 0.5)
        interval = (end - start) / max(sample_count, 1)

        frame_paths = []
        for i in range(sample_count):
            timestamp = start + (i * interval)
            frame_path = os.path.join(tmp_dir, f"frame_{i:03d}.png")

            # Extract frame and crop to lower portion only
            # Using FFmpeg crop filter: crop=w:h:x:y
            # crop_top_pct=0.55 means keep bottom 45% of frame
            result = subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{timestamp:.2f}",
                 "-i", video_path, "-frames:v", "1",
                 "-vf", f"crop=iw:ih*{1-crop_top_pct:.2f}:0:ih*{crop_top_pct:.2f}",
                 frame_path],
                capture_output=True, text=True
            )
            if result.returncode == 0 and os.path.exists(frame_path):
                frame_paths.append((i, timestamp, frame_path))

        if not frame_paths:
            return {"has_text": False, "detections": [], "frames_checked": 0,
                    "frames_with_text": 0, "error": "No frames extracted"}

        # Run OCR on each cropped frame
        detections = []
        frames_with_text = 0

        for frame_idx, timestamp, frame_path in frame_paths:
            try:
                results = reader.readtext(frame_path)

                # Filter by confidence
                texts = []
                confidences = []
                for (bbox, text, conf) in results:
                    if conf >= min_confidence and len(text.strip()) >= 2:
                        texts.append(text.strip())
                        confidences.append(round(conf, 3))

                if texts:
                    frames_with_text += 1
                    detections.append({
                        "frame": frame_idx,
                        "timestamp": round(timestamp, 1),
                        "texts": texts,
                        "confidences": confidences,
                    })

            except Exception as e:
                print(f"[TextDetect] OCR error on frame {frame_idx}: {e}")

        has_text = frames_with_text >= min_detections

        return {
            "has_text": has_text,
            "detections": detections,
            "frames_checked": len(frame_paths),
            "frames_with_text": frames_with_text,
        }

    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass


def _get_duration(video_path):
    """Get video duration using ffprobe."""
    try:
        import json
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


# ======================== STANDALONE TEST ========================

if __name__ == "__main__":
    """
    Test text detection on a video file.
    Usage: python -m modules.text_detect path/to/video.mp4
    """
    if len(sys.argv) < 2:
        print("Usage: python -m modules.text_detect <video_path>")
        print("       python -m modules.text_detect <video_path> <video_path2> ...")
        sys.exit(1)

    for video_path in sys.argv[1:]:
        print(f"\n{'='*50}")
        print(f"Checking: {os.path.basename(video_path)}")
        print(f"{'='*50}")

        result = detect_text_in_video(video_path)

        if result.get("error"):
            print(f"  Error: {result['error']}")
            continue

        print(f"  Frames checked: {result['frames_checked']}")
        print(f"  Frames with text: {result['frames_with_text']}")
        print(f"  Text detected: {'YES ⚠️' if result['has_text'] else 'No ✓'}")

        if result["detections"]:
            print(f"\n  Detections:")
            for d in result["detections"]:
                print(f"    Frame {d['frame']} ({d['timestamp']}s): "
                      f"{d['texts']} (conf: {d['confidences']})")
