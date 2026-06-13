"""
RunPod Serverless Handler for LongCat-Video-Avatar 1.5
Storage-optimized version — fits within 50GB network volume
"""

import runpod
import os
import base64
import tempfile
import subprocess
import requests
import shutil
import time
from pathlib import Path

# Paths
WEIGHTS_DIR = "/runpod-volume/weights"
OUTPUT_DIR = "/runpod-volume/outputs"
LOCK_FILE = "/runpod-volume/.download_lock"
LONGCAT_VIDEO_WEIGHTS = f"{WEIGHTS_DIR}/LongCat-Video"
LONGCAT_AVATAR_WEIGHTS = f"{WEIGHTS_DIR}/LongCat-Avatar"

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_disk_usage_gb(path):
    """Returns used disk space in GB."""
    result = subprocess.run(["df", "-BG", path], capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")
    if len(lines) >= 2:
        parts = lines[1].split()
        used = int(parts[2].replace("G", ""))
        return used
    return 0


def clear_hf_cache():
    """Delete HuggingFace temp cache to free space."""
    for weights_path in [LONGCAT_VIDEO_WEIGHTS, LONGCAT_AVATAR_WEIGHTS]:
        hf_cache = os.path.join(weights_path, ".huggingface")
        if os.path.exists(hf_cache):
            shutil.rmtree(hf_cache, ignore_errors=True)
            print(f"Cleared HF cache at {hf_cache}")


def wait_for_lock(timeout=1800):
    """Wait if another worker is downloading weights."""
    waited = 0
    while os.path.exists(LOCK_FILE) and waited < timeout:
        print(f"Another worker is downloading weights, waiting... ({waited}s)")
        time.sleep(10)
        waited += 10
    if waited >= timeout:
        # Remove stale lock
        os.remove(LOCK_FILE)


def download_weights():
    """Download model weights — storage optimized, one file at a time."""

    video_done = Path(f"{LONGCAT_VIDEO_WEIGHTS}/.complete")
    avatar_done = Path(f"{LONGCAT_AVATAR_WEIGHTS}/.complete")

    if video_done.exists() and avatar_done.exists():
        print("Weights already downloaded, skipping.")
        return

    # Prevent multiple workers downloading simultaneously
    wait_for_lock()

    # Re-check after waiting
    if video_done.exists() and avatar_done.exists():
        print("Weights downloaded by another worker, skipping.")
        return

    # Set lock
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

    try:
        # --- Download LongCat-Video base (only essential files, skip large text encoder) ---
        if not video_done.exists():
            print("Downloading LongCat-Video essential weights...")
            os.makedirs(LONGCAT_VIDEO_WEIGHTS, exist_ok=True)

            # Download only the essential files — skip text_encoder (saves ~20GB)
            # We use the avatar model's built-in encoder instead
            essential_patterns = [
                "*.json",
                "*.md", 
                "dit/*",
                "lora/*",
                "scheduler/*",
                "tokenizer/*",
                "vae/*",
            ]

            for pattern in essential_patterns:
                print(f"Downloading {pattern}...")
                result = subprocess.run([
                    "huggingface-cli", "download",
                    "meituan-longcat/LongCat-Video",
                    "--include", pattern,
                    "--local-dir", LONGCAT_VIDEO_WEIGHTS,
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"Warning downloading {pattern}: {result.stderr[-200:]}")

                # Clean HF cache after each pattern to save space
                clear_hf_cache()

                used = get_disk_usage_gb("/runpod-volume")
                print(f"Disk used: {used}GB")

            video_done.touch()
            print("LongCat-Video essential weights done.")

        # --- Download Avatar weights ---
        if not avatar_done.exists():
            print("Downloading LongCat-Video-Avatar-1.5 weights...")
            os.makedirs(LONGCAT_AVATAR_WEIGHTS, exist_ok=True)

            result = subprocess.run([
                "huggingface-cli", "download",
                "meituan-longcat/LongCat-Video-Avatar-1.5",
                "--local-dir", LONGCAT_AVATAR_WEIGHTS,
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"Avatar download failed: {result.stderr[-500:]}")

            # Clean cache after download
            clear_hf_cache()

            used = get_disk_usage_gb("/runpod-volume")
            print(f"Disk used after avatar download: {used}GB")

            avatar_done.touch()
            print("Avatar weights done.")

    finally:
        # Always release lock
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)


def decode_input(value, suffix, tmp_dir):
    """Accepts base64 string or URL, saves to temp file."""
    tmp_path = os.path.join(tmp_dir, f"input{suffix}")

    if value.startswith("data:"):
        header, data = value.split(",", 1)
        file_bytes = base64.b64decode(data)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
    elif value.startswith("http://") or value.startswith("https://"):
        resp = requests.get(value, timeout=60)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
    else:
        file_bytes = base64.b64decode(value)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

    return tmp_path


def encode_video_base64(video_path):
    """Read video file and return as base64 string."""
    with open(video_path, "rb") as f:
        data = f.read()
    return "data:video/mp4;base64," + base64.b64encode(data).decode("utf-8")


def handler(job):
    """
    Main RunPod handler.

    Input:
    {
        "image_base64": "..." OR "image_url": "https://...",
        "wav_base64": "..."  OR "wav_url": "https://...",
        "prompt": "A person talking naturally",
        "width": 720,
        "height": 1280,
        "use_distill": true,
        "use_int8": true,
        "network_volume": true
    }
    """

    job_input = job.get("input", {})

    # Download weights (cached after first run)
    try:
        download_weights()
    except Exception as e:
        return {"error": f"Failed to download weights: {str(e)}"}

    # Parse inputs
    prompt = job_input.get("prompt", "A person talking naturally and clearly")
    width = job_input.get("width", 720)
    height = job_input.get("height", 1280)
    use_distill = job_input.get("use_distill", True)
    use_int8 = job_input.get("use_int8", True)
    audio_cfg = job_input.get("audio_cfg", 4.0)
    use_network_volume = job_input.get("network_volume", True)

    # Get inputs
    image_value = job_input.get("image_base64") or job_input.get("image_url")
    if not image_value:
        return {"error": "Missing image. Provide image_base64 or image_url."}

    audio_value = job_input.get("wav_base64") or job_input.get("wav_url")
    if not audio_value:
        return {"error": "Missing audio. Provide wav_base64 or wav_url."}

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            image_path = decode_input(image_value, ".jpg", tmp_dir)
            audio_path = decode_input(audio_value, ".wav", tmp_dir)
        except Exception as e:
            return {"error": f"Failed to decode inputs: {str(e)}"}

        # Output path
        timestamp = int(time.time())
        output_path = os.path.join(OUTPUT_DIR, f"avatar_{timestamp}.mp4")

        # Build inference command
        cmd = [
            "python", "/app/inference_avatar.py",
            "--model_path", LONGCAT_VIDEO_WEIGHTS,
            "--avatar_model_path", LONGCAT_AVATAR_WEIGHTS,
            "--ref_image_path", image_path,
            "--audio_path", audio_path,
            "--prompt", prompt,
            "--width", str(width),
            "--height", str(height),
            "--audio_cfg", str(audio_cfg),
            "--save_path", output_path,
        ]

        if use_distill:
            cmd.append("--use_distill")
        if use_int8:
            cmd.append("--use_int8")

        print(f"Running inference: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/app",
                timeout=3600
            )

            print("STDOUT:", result.stdout[-3000:])
            print("STDERR:", result.stderr[-3000:])

            if result.returncode != 0:
                return {"error": f"Inference failed: {result.stderr[-1000:]}"}

        except subprocess.TimeoutExpired:
            return {"error": "Job timed out after 1 hour"}
        except Exception as e:
            return {"error": f"Inference error: {str(e)}"}

        # Find output video
        if not os.path.exists(output_path):
            mp4_files = sorted(Path(OUTPUT_DIR).glob("*.mp4"), key=os.path.getmtime, reverse=True)
            if mp4_files:
                output_path = str(mp4_files[0])
            else:
                return {"error": "Video not found after inference."}

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Video generated: {output_path} ({size_mb:.1f} MB)")

        # Return result
        if use_network_volume:
            return {"video_path": output_path, "size_mb": round(size_mb, 1)}
        else:
            try:
                return {"video": encode_video_base64(output_path), "size_mb": round(size_mb, 1)}
            except Exception as e:
                return {"error": f"Failed to encode video: {str(e)}"}


runpod.serverless.start({"handler": handler})
