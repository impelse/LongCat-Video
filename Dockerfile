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
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Copy repo contents
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Install additional dependencies for RunPod handler
RUN pip install \
    runpod==1.7.0 \
    huggingface_hub \
    transformers \
    accelerate \
    diffusers \
    openai-whisper \
    ffmpeg-python \
    Pillow \
    numpy \
    scipy \
    soundfile \
    librosa

# Create necessary directories
RUN mkdir -p /runpod-volume/huggingface \
    && mkdir -p /runpod-volume/outputs \
    && mkdir -p /app/weights

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
