"""
One-off script to pre-download LongCat weights to network volume.
Run this ONCE as a RunPod job before using the main handler. Raul
"""

import os
import subprocess
from pathlib import Path

# Force everything to network volume
os.environ["HF_HOME"] = "/runpod-volume/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/runpod-volume/.cache/huggingface"

WEIGHTS_DIR = "/runpod-volume/weights"
LONGCAT_VIDEO_WEIGHTS = f"{WEIGHTS_DIR}/LongCat-Video"
LONGCAT_AVATAR_WEIGHTS = f"{WEIGHTS_DIR}/LongCat-Avatar"

os.makedirs(LONGCAT_VIDEO_WEIGHTS, exist_ok=True)
os.makedirs(LONGCAT_AVATAR_WEIGHTS, exist_ok=True)

def download(repo, local_dir, marker_name):
    marker = Path(f"{local_dir}/.complete")
    if marker.exists():
        print(f"✅ {marker_name} already downloaded, skipping.")
        return

    print(f"⬇️  Downloading {repo} → {local_dir}")
    result = subprocess.run([
        "huggingface-cli", "download", repo,
        "--local-dir", local_dir,
    ], check=False)

    if result.returncode == 0:
        marker.touch()
        print(f"✅ {marker_name} complete — .complete marker written")
    else:
        print(f"❌ {marker_name} failed")

# Download both models
download(
    "meituan-longcat/LongCat-Video",
    LONGCAT_VIDEO_WEIGHTS,
    "LongCat-Video base"
)

download(
    "meituan-longcat/LongCat-Video-Avatar-1.5",
    LONGCAT_AVATAR_WEIGHTS,
    "LongCat-Avatar-1.5"
)

# Show final disk usage
print("\n📦 Final disk usage:")
os.system("df -h /runpod-volume")
os.system("du -sh /runpod-volume/weights/*")
print("\n✅ All done — weights cached, ready for inference jobs.")
