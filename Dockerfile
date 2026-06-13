FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libsndfile1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 first
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
RUN pip install \
    transformers==4.40.0 \
    accelerate==0.30.0 \
    diffusers==0.27.2 \
    huggingface_hub==0.23.0 \
    safetensors==0.4.3 \
    tokenizers==0.19.1

# Install audio dependencies (fixed versions)
RUN pip install soundfile==0.12.1
RUN pip install librosa==0.10.1
RUN pip install openai-whisper
RUN pip install scipy==1.13.0

# Install video/image dependencies
RUN pip install \
    Pillow==10.3.0 \
    opencv-python-headless==4.9.0.80 \
    ffmpeg-python==0.2.0 \
    imageio==2.34.1 \
    imageio-ffmpeg==0.5.1

# Install RunPod SDK
RUN pip install runpod==1.7.0

# Install other utilities
RUN pip install \
    numpy==1.26.4 \
    einops==0.7.0 \
    omegaconf==2.3.0 \
    tqdm==4.66.4 \
    requests==2.31.0

# Copy repo contents
WORKDIR /app
COPY . /app

# Create necessary directories
RUN mkdir -p /runpod-volume/huggingface \
    && mkdir -p /runpod-volume/outputs \
    && mkdir -p /runpod-volume/weights

CMD ["python", "-u", "handler.py"]
