# NLP Server Containerization

This document provides instructions for building and running the NLP Server using Singularity or Docker containers.

## Prerequisites

### For Singularity
- Singularity/Apptainer installed on your system
- Root access or fakeroot capability
- CUDA drivers (if using GPU)

### For Docker
- Docker installed on your system
- NVIDIA Container Toolkit (if using GPU)
- CUDA drivers (if using GPU)

## Building the Container

### Singularity

1. **Build the container:**
   ```bash
   # Make the build script executable
   chmod +x build_singularity.sh
   
   # Build with default name (nlp-server.sif)
   ./build_singularity.sh
   
   # Or specify a custom name
   ./build_singularity.sh my-nlp-server.sif
   ```

2. **Manual build (alternative):**
   ```bash
   # With fakeroot (recommended)
   singularity build --fakeroot nlp-server.sif nlp-server.def
   
   # With sudo (if fakeroot not available)
   sudo singularity build nlp-server.sif nlp-server.def
   ```

### Docker

1. **Build the container:**
   ```bash
   docker build -t nlp-server:latest .
   ```

## Running the Container

### Singularity

1. **Basic run:**
   ```bash
   singularity run nlp-server.sif
   ```

2. **Run with custom port:**
   ```bash
   singularity run nlp-server.sif --port 8080
   ```

3. **Run with GPU support:**
   ```bash
   singularity run --nv nlp-server.sif
   ```

4. **Run with custom models directory:**
   ```bash
   singularity run --bind /path/to/your/models:/app/models nlp-server.sif
   ```

5. **Interactive shell:**
   ```bash
   singularity shell nlp-server.sif
   ```

6. **Execute specific commands:**
   ```bash
   # Test the server
   singularity exec nlp-server.sif python3 test_server.py
   
   # Run client example
   singularity exec nlp-server.sif python3 examples/client_example.py
   ```

### Docker

1. **Basic run:**
   ```bash
   docker run -p 8000:8000 nlp-server:latest
   ```

2. **Run with GPU support:**
   ```bash
   docker run --gpus all -p 8000:8000 nlp-server:latest
   ```

3. **Run with custom models directory:**
   ```bash
   docker run -p 8000:8000 -v /path/to/your/models:/app/models nlp-server:latest
   ```

4. **Interactive shell:**
   ```bash
   docker run -it nlp-server:latest /bin/bash
   ```

## Container Features

### Included Components

- **Base Image:** `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **PyTorch:** 2.5.1 with CUDA 12.1 support
- **Coqui TTS:** Custom fork with Akan language support
- **PortAudio:** For audio processing
- **Pre-trained Model:** Akan multi-speaker TTS model
- **FastAPI Server:** RESTful API for TTS operations

### Model Configuration

The container includes a pre-configured Akan TTS model:
- **Model Name:** `ugtts_multispeaker`
- **Language:** Akan (`aka`)
- **Speakers:** PT, IM, AN
- **Default Speaker:** PT
- **Model Path:** `/app/models/ugtts-multispeaker/`

### Environment Variables

You can customize the container behavior with these environment variables:

```bash
# Server configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Models configuration
MODELS_DIR=/app/models
LOG_LEVEL=INFO

# GPU configuration
CUDA_VISIBLE_DEVICES=0
TORCH_DEVICE=auto
```

## Usage Examples

### 1. Start the Server

```bash
# Singularity
singularity run nlp-server.sif

# Docker
docker run -p 8000:8000 nlp-server:latest
```

### 2. Test the Server

```bash
# From inside the container
singularity exec nlp-server.sif python3 test_server.py

# From host (if server is running)
python3 test_server.py
```

### 3. Use the API

```bash
# Get available models
curl http://localhost:8000/tts/models

# Synthesize speech
curl -X POST http://localhost:8000/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello in Akan",
    "model_name": "ugtts_multispeaker",
    "speaker": "PT",
    "length_scale": 1.0
  }'
```

### 4. Access API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Common Issues

1. **Permission Denied:**
   ```bash
   # For Singularity, try with fakeroot
   singularity build --fakeroot nlp-server.sif nlp-server.def
   
   # Or with sudo
   sudo singularity build nlp-server.sif nlp-server.def
   ```

2. **GPU Not Available:**
   ```bash
   # Check if CUDA is available
   singularity exec nlp-server.sif python3 -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Port Already in Use:**
   ```bash
   # Use a different port
   singularity run nlp-server.sif --port 8080
   ```

4. **Model Files Not Found:**
   ```bash
   # Check if model files exist
   singularity exec nlp-server.sif ls -la /app/models/ugtts-multispeaker/
   ```

### Debug Mode

To run in debug mode:

```bash
# Set environment variable
export DEBUG=true

# Run container
singularity run nlp-server.sif
```

### Logs

Container logs are stored in `/app/logs/` inside the container. To access them:

```bash
# View logs from host
singularity exec nlp-server.sif cat /app/logs/nlp_server.log

# Or mount logs directory
singularity run --bind /tmp/logs:/app/logs nlp-server.sif
```

## Performance Optimization

### GPU Usage

For optimal GPU performance:

1. **Ensure CUDA drivers are installed on the host**
2. **Use `--nv` flag with Singularity:**
   ```bash
   singularity run --nv nlp-server.sif
   ```
3. **Use `--gpus all` with Docker:**
   ```bash
   docker run --gpus all -p 8000:8000 nlp-server:latest
   ```

### Memory Optimization

- The container includes model caching for better performance
- Models are loaded on-demand to save memory
- Use `singularity exec` to unload models when not needed

## Security Considerations

1. **Network Access:** The server binds to `0.0.0.0:8000` by default
2. **File Permissions:** Container runs with appropriate file permissions
3. **Model Access:** Models are read-only inside the container
4. **API Security:** Consider adding authentication for production use

## Production Deployment

For production deployment:

1. **Use a reverse proxy (nginx, Apache)**
2. **Add SSL/TLS certificates**
3. **Implement authentication and rate limiting**
4. **Set up monitoring and logging**
5. **Use container orchestration (Kubernetes, Docker Swarm)**

## Support

For issues and questions:
1. Check the logs: `/app/logs/nlp_server.log`
2. Test the server: `python3 test_server.py`
3. Check API documentation: http://localhost:8000/docs
4. Review the main README.md for detailed API usage 