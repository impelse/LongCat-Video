"""
RunPod Serverless Handler for LongCat-Video-Avatar 1.5
Accepts: image (base64 or URL) + audio (base64 or URL) + prompt
Returns: video (base64) or video_path (if network volume used)
"""

import runpod
import os
import base64
import tempfile
import subprocess
import requests
import json
import time
from pathlib import Path

# Paths
WEIGHTS_DIR = "/runpod-volume/weights"
OUTPUT_DIR = "/runpod-volume/outputs"
LONGCAT_VIDEO_WEIGHTS = f"{WEIGHTS_DIR}/LongCat-Video"
LONGCAT_AVATAR_WEIGHTS = f"{WEIGHTS_DIR}/LongCat-Video-Avatar"

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_weights():
    """Download model weights on first run — cached to network volume."""
    longcat_video_done = Path(f"{LONGCAT_VIDEO_WEIGHTS}/.download_complete")
    longcat_avatar_done = Path(f"{LONGCAT_AVATAR_WEIGHTS}/.download_complete")

    if not longcat_video_done.exists():
        print("Downloading LongCat-Video base weights (~20GB)...")
        subprocess.run([
            "huggingface-cli", "download",
            "meituan-longcat/LongCat-Video",
            "--local-dir", LONGCAT_VIDEO_WEIGHTS
        ], check=True)
        longcat_video_done.touch()
        print("LongCat-Video weights downloaded.")

    if not longcat_avatar_done.exists():
        print("Downloading LongCat-Video-Avatar-1.5 weights...")
        subprocess.run([
            "huggingface-cli", "download",
            "meituan-longcat/LongCat-Video-Avatar-1.5",
            "--local-dir", LONGCAT_AVATAR_WEIGHTS
        ], check=True)
        longcat_avatar_done.touch()
        print("LongCat-Video-Avatar weights downloaded.")


def decode_input(value, suffix, tmp_dir):
    """
    Accepts base64 string or URL.
    Saves to a temp file and returns the path.
    """
    tmp_path = os.path.join(tmp_dir, f"input{suffix}")

    if value.startswith("data:"):
        # Base64 with data URI prefix
        header, data = value.split(",", 1)
        file_bytes = base64.b64decode(data)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

    elif value.startswith("http://") or value.startswith("https://"):
        # URL — download it
        resp = requests.get(value, timeout=60)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(resp.content)

    else:
        # Raw base64 (no prefix)
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
    Main RunPod handler function.

    Input:
    {
        "image_base64": "...",   # OR
        "image_url": "https://...",

        "wav_base64": "...",     # OR
        "wav_url": "https://...",

        "prompt": "A person talking naturally",  # optional
        "width": 720,            # optional, default 720
        "height": 1280,          # optional, default 1280
        "model_type": "avatar-v1.5",  # optional, default avatar-v1.5
        "use_distill": true,     # optional, default true (faster)
        "use_int8": true,        # optional, default true (less VRAM)
        "audio_cfg": 4.0,        # optional, 3-5 recommended
        "network_volume": false  # optional, return path instead of base64
    }

    Output (network_volume=false):
    { "video": "data:video/mp4;base64,..." }

    Output (network_volume=true):
    { "video_path": "/runpod-volume/outputs/video_xxx.mp4" }
    """

    job_input = job.get("input", {})

    # --- Download weights if not cached ---
    try:
        download_weights()
    except Exception as e:
        return {"error": f"Failed to download weights: {str(e)}"}

    # --- Parse inputs ---
    prompt = job_input.get("prompt", "A person talking naturally and clearly")
    width = job_input.get("width", 720)
    height = job_input.get("height", 1280)
    model_type = job_input.get("model_type", "avatar-v1.5")
    use_distill = job_input.get("use_distill", True)
    use_int8 = job_input.get("use_int8", True)
    audio_cfg = job_input.get("audio_cfg", 4.0)
    use_network_volume = job_input.get("network_volume", False)

    # Get image input
    image_value = job_input.get("image_base64") or job_input.get("image_url")
    if not image_value:
        return {"error": "Missing image input. Provide image_base64 or image_url."}

    # Get audio input
    audio_value = job_input.get("wav_base64") or job_input.get("wav_url")
    if not audio_value:
        return {"error": "Missing audio input. Provide wav_base64 or wav_url."}

    # --- Save inputs to temp files ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            image_path = decode_input(image_value, ".jpg", tmp_dir)
            audio_path = decode_input(audio_value, ".wav", tmp_dir)
        except Exception as e:
            return {"error": f"Failed to decode inputs: {str(e)}"}

        # --- Output path ---
        timestamp = int(time.time())
        output_filename = f"longcat_avatar_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # --- Build command ---
        cmd = [
            "python", "/app/inference_avatar.py",
            "--model_path", LONGCAT_VIDEO_WEIGHTS,
            "--avatar_model_path", LONGCAT_AVATAR_WEIGHTS,
            "--model_type", model_type,
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

        print(f"Running: {' '.join(cmd)}")

        # --- Run LongCat ---
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/app",
                timeout=3600  # 1 hour max
            )

            print("STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
            print("STDERR:", result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)

            if result.returncode != 0:
                return {"error": f"LongCat inference failed: {result.stderr[-1000:]}"}

        except subprocess.TimeoutExpired:
            return {"error": "Job timed out after 1 hour"}
        except Exception as e:
            return {"error": f"Inference error: {str(e)}"}

        # --- Check output exists ---
        if not os.path.exists(output_path):
            # Try to find any generated mp4
            mp4_files = list(Path(OUTPUT_DIR).glob(f"*{timestamp}*.mp4"))
            if mp4_files:
                output_path = str(mp4_files[0])
            else:
                return {"error": "Video not found after inference. Check logs."}

        print(f"Video generated: {output_path} ({os.path.getsize(output_path)} bytes)")

        # --- Return result ---
        if use_network_volume:
            return {"video_path": output_path}
        else:
            try:
                video_b64 = encode_video_base64(output_path)
                return {"video": video_b64}
            except Exception as e:
                return {"error": f"Failed to encode output video: {str(e)}"}


# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
