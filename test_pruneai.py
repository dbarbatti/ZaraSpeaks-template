# test_pruneai.py
# -----------------------------------------------------------
#  Quick test: PrunaAI P-Video on Replicate
#  Tests draft mode vs full quality with Aria image + audio
#  
#  Setup:
#    pip install replicate
#    set REPLICATE_API_TOKEN=your-token-here
# -----------------------------------------------------------

import os
import sys
import time
import requests

REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_TOKEN:
    print("ERROR: Set REPLICATE_API_TOKEN environment variable")
    print("  set REPLICATE_API_TOKEN=r8_your_token_here")
    print("  Get your token at: https://replicate.com/account/api-tokens")
    sys.exit(1)


def get_audio_duration(path):
    """Get audio duration in seconds."""
    try:
        import wave
        with wave.open(path, 'rb') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0


def split_audio(audio_path, chunk_seconds=10):
    """Split audio into chunks for P-Video's 10-second limit."""
    import subprocess
    
    duration = get_audio_duration(audio_path)
    if duration <= chunk_seconds:
        return [audio_path], duration
    
    chunks = []
    num_chunks = int(duration / chunk_seconds) + (1 if duration % chunk_seconds > 0 else 0)
    
    for i in range(num_chunks):
        start = i * chunk_seconds
        chunk_path = f"temp_chunk_{i}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-t", str(chunk_seconds),
            "-c:a", "pcm_s16le", "-ar", "24000",
            chunk_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        chunks.append(chunk_path)
    
    print(f"  Split {duration:.1f}s audio into {len(chunks)} chunks of ~{chunk_seconds}s")
    return chunks, duration


def stitch_videos(video_paths, output_path):
    """Concatenate multiple video files using FFmpeg."""
    import subprocess
    
    if len(video_paths) == 1:
        # Just copy the single file
        with open(video_paths[0], 'rb') as src:
            with open(output_path, 'wb') as dst:
                dst.write(src.read())
        return True
    
    # Create concat list file
    list_path = "temp_concat_list.txt"
    with open(list_path, 'w') as f:
        for vp in video_paths:
            # Use absolute paths and forward slashes for FFmpeg
            abs_path = os.path.abspath(vp).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")
    
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Cleanup
    try:
        os.remove(list_path)
    except OSError:
        pass
    
    if result.returncode != 0:
        print(f"  Stitch error: {result.stderr[:200]}")
        # Fallback: re-encode
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path if os.path.exists(list_path) else "",
            "-c:v", "libx264", "-c:a", "aac",
            output_path
        ]
        # Write list file again
        with open("temp_concat_list.txt", 'w') as f:
            for vp in video_paths:
                abs_path = os.path.abspath(vp).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
        subprocess.run(cmd, capture_output=True)
        try:
            os.remove("temp_concat_list.txt")
        except OSError:
            pass
    
    return os.path.exists(output_path)


def run_pvideo(image_path, audio_path, draft=False, resolution="720p",
               prompt="A woman speaking naturally, front facing, warm expression, soft lighting"):
    """Run a single P-Video generation."""
    import replicate
    
    with open(image_path, "rb") as img_file, open(audio_path, "rb") as aud_file:
        output = replicate.run(
            "prunaai/p-video",
            input={
                "image": img_file,
                "audio": aud_file,
                "prompt": prompt,
                "resolution": resolution,
                "draft": draft,
                "prompt_upsampling": False,
                "fps": 24,
            }
        )
    
    # Output is a FileOutput URL
    if output:
        return str(output)
    return None


def download(url, path):
    """Download file from URL."""
    resp = requests.get(url, stream=True, timeout=120)
    if resp.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"  Downloaded: {path} ({size:.1f} MB)")
        return True
    print(f"  Download failed: {resp.status_code}")
    return False


def test_single(image_path, audio_path, draft, resolution, label, output_name):
    """Run a single test — handles chunking if audio > 10s."""
    print(f"\n{'='*55}")
    mode = "DRAFT" if draft else "FULL"
    print(f"  Test: {label}")
    print(f"  Mode: {mode} | Resolution: {resolution}")
    print(f"{'='*55}")
    
    duration = get_audio_duration(audio_path)
    cost_per_sec = {
        ("720p", True): 0.005,
        ("720p", False): 0.02,
        ("1080p", True): 0.01,
        ("1080p", False): 0.04,
    }.get((resolution, draft), 0.02)
    
    est_cost = duration * cost_per_sec
    print(f"  Audio: {duration:.1f}s | Est. cost: ${est_cost:.3f}")
    
    start = time.time()
    
    # Split audio if > 10 seconds
    if duration > 10:
        chunks, _ = split_audio(audio_path, chunk_seconds=10)
    else:
        chunks = [audio_path]
    
    # Generate each chunk
    video_parts = []
    for i, chunk_path in enumerate(chunks):
        chunk_dur = get_audio_duration(chunk_path)
        print(f"  Generating chunk {i+1}/{len(chunks)} ({chunk_dur:.1f}s)...")
        
        chunk_start = time.time()
        url = run_pvideo(image_path, chunk_path, draft=draft, resolution=resolution)
        chunk_elapsed = time.time() - chunk_start
        
        if url:
            part_path = f"temp_part_{i}.mp4"
            if download(url, part_path):
                video_parts.append(part_path)
                print(f"  Chunk {i+1} done in {chunk_elapsed:.1f}s")
            else:
                print(f"  Chunk {i+1} download failed")
        else:
            print(f"  Chunk {i+1} generation failed")
    
    if not video_parts:
        print("  FAILED: No video parts generated")
        return
    
    # Stitch if multiple parts
    if len(video_parts) > 1:
        print(f"  Stitching {len(video_parts)} parts...")
        stitch_videos(video_parts, output_name)
    else:
        os.rename(video_parts[0], output_name)
    
    elapsed = time.time() - start
    
    # Cleanup temp files
    for chunk_path in chunks:
        if chunk_path != audio_path and os.path.exists(chunk_path):
            try:
                os.remove(chunk_path)
            except OSError:
                pass
    for part_path in video_parts:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
    
    if os.path.exists(output_name):
        size = os.path.getsize(output_name) / (1024 * 1024)
        print(f"\n  ✓ RESULT: {output_name} ({size:.1f} MB)")
        print(f"  ✓ Total time: {elapsed:.1f}s")
        print(f"  ✓ Est. cost: ${est_cost:.3f}")
    else:
        print(f"\n  ✗ FAILED to produce output")


def main():
    print("=" * 55)
    print("  PrunaAI P-Video Test Suite")
    print("=" * 55)
    
    # Find image
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        for f in os.listdir("."):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'thumbnail' in f.lower():
                image_path = f
                break
        if not image_path:
            for d in [".", "base_videos"]:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(d, f)
                            break
                if image_path:
                    break
    
    # Find audio
    audio_path = None
    if len(sys.argv) > 2:
        audio_path = sys.argv[2]
    else:
        for d in ["voice_references", ".", "temp"]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.lower().endswith('.wav'):
                        audio_path = os.path.join(d, f)
                        break
            if audio_path:
                break
    
    if not image_path:
        print("\nNo image found. Usage:")
        print("  python test_pruneai.py <image_path> <audio_path>")
        return
    
    if not audio_path:
        print("\nNo audio found. Usage:")
        print("  python test_pruneai.py <image_path> <audio_path>")
        return
    
    duration = get_audio_duration(audio_path)
    print(f"\n  Image: {image_path}")
    print(f"  Audio: {audio_path} ({duration:.1f}s)")
    
    if duration > 10:
        print(f"  Note: Audio > 10s — will split into chunks")
    
    # Cost preview
    print(f"\n  --- Cost Estimates ---")
    print(f"  Draft 720p:  ${duration * 0.005:.3f}")
    print(f"  Full 720p:   ${duration * 0.02:.3f}")
    print(f"  Draft 1080p: ${duration * 0.01:.3f}")
    print(f"  Full 1080p:  ${duration * 0.04:.3f}")
    
    proceed = input(f"\n  Run tests? (y/n) [y]: ").strip().lower()
    if proceed == 'n':
        return
    

    # Test 2: Full 720p
    test_single(image_path, audio_path,
                draft=False, resolution="720p",
                label="Full 720p — production quality",
                output_name="test_PVIDEO_full_720p.mp4")
    
    # Summary
    print(f"\n{'='*55}")
    print("  RESULTS SUMMARY")
    print(f"{'='*55}")
    
    results = [
        ("test_PVIDEO_draft_720p.mp4", "Draft 720p", duration * 0.005),
        ("test_PVIDEO_full_720p.mp4", "Full 720p", duration * 0.02),
    ]
    
    for path, label, cost in results:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  [OK] {label}: {path} ({size:.1f} MB) ~${cost:.3f}")
        else:
            print(f"  [--] {label}: not generated")
    
    print(f"\n  Open in VLC to compare quality!")
    print(f"  Compare with test_FABRIC_480p.mp4 if available")


if __name__ == "__main__":
    main()
