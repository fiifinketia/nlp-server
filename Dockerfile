FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set environment variables
ENV LC_ALL=C
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_DEVICE=auto

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PortAudio
RUN cd /tmp \
    && wget https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz \
    && tar -xvzf pa_stable_v190700_20210406.tgz \
    && cd portaudio && ./configure && make && make install && cd .. \
    && ldconfig \
    && rm -rf /tmp/portaudio*

# Install PyAudio
RUN pip install pyaudio --upgrade

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.5.1+cu121 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Clone and install Coqui TTS
RUN git clone https://github.com/bytlabs-io/coqui-ai-TTS /tmp/coqui-ai-TTS \
    && cd /tmp/coqui-ai-TTS && git checkout dev-akan \
    && pip install -e /tmp/coqui-ai-TTS[all] \
    && rm -rf /tmp/coqui-ai-TTS

# Copy application files
COPY requirements.txt /tmp/requirements.txt
COPY app.py /app/app.py
COPY start_server.py /app/start_server.py
COPY test_server.py /app/test_server.py
COPY config/ /app/config/
COPY services/ /app/services/
COPY examples/ /app/examples/
COPY README.md /app/README.md
COPY env.example /app/env.example

# Install Python dependencies
RUN pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Configure Git credentials
RUN git config --global credential.helper store

# Create directories
RUN mkdir -p /app/models /app/logs

# Clone the dataset
RUN cd /app/models \
    && git clone https://huggingface.co/hci-lab-dcug/ugtts-multispeaker-max42secs-total7hrs-sr22050-bibletts-finetuned ./ugtts-multispeaker

# Update speakers_file path in config.json if it exists
RUN if [ -f "/app/models/ugtts-multispeaker/config.json" ]; then \
        sed -i 's|"speakers_file": ".*"|"speakers_file": "/app/models/ugtts-multispeaker/speakers.pth"|g' /app/models/ugtts-multispeaker/config.json; \
    fi

# Set permissions
RUN chmod +x /app/start_server.py /app/test_server.py

# Set working directory
WORKDIR /app

# Expose port
EXPOSE 8000

# Default command
CMD ["python3", "start_server.py"] 