# test_text_detect.py
# Quick test of text detection against known good/bad videos
# Usage: python test_text_detect.py

import sys
sys.path.insert(0, ".")

from modules.text_detect import detect_text_in_video

# Test videos — update paths if needed
videos = [
    ("WITH text (should detect)", r"video_library\zara_response_20260406_193118.mp4"),
    ("NO text (should be clean)", r"video_library\zara_response_20260328_141821.mp4"),
]

for label, path in videos:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*50}")

    result = detect_text_in_video(path)

    if result.get("error"):
        print(f"  Error: {result['error']}")
        continue

    print(f"  Frames checked: {result['frames_checked']}")
    print(f"  Frames with text: {result['frames_with_text']}")

    if result["has_text"]:
        print(f"  Result: TEXT DETECTED ⚠️")
        for d in result["detections"]:
            print(f"    @ {d['timestamp']}s: {d['texts']} "
                  f"(conf: {d['confidences']})")
    else:
        print(f"  Result: CLEAN ✓")
