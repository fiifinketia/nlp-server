# NLP Server

A comprehensive NLP server for Text-to-Speech (TTS), Automatic Speech Recognition (ASR), Large Language Models (LLMs), and translation tasks. Currently supports TTS functionality with single-speaker and multi-speaker models.

## Features

- **Text-to-Speech (TTS)**: Support for single-speaker and multi-speaker TTS models
- **Model Management**: Dynamic loading/unloading of models
- **RESTful API**: FastAPI-based REST API with automatic documentation
- **Configuration Management**: Flexible model configuration system
- **Audio Processing**: Support for various audio formats and parameters

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for TTS models)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dcshci-nlp-server
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your models:
   - Create model directories in the `models/` folder
   - Place your TTS model files (`.pth` and `config.json`) in the appropriate directories
   - Update the model configuration in `config/models_config.py`

4. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:8000`

## Model Configuration

The server uses a configuration system to manage TTS models. Models are defined in `config/models_config.py`:

```python
# Example model configuration
"ugtts_multi": TTSModelConfig(
    name="ugtts_multi",
    model_type=ModelType.MULTI_SPEAKER,
    model_path="models/ugtts-multi/best_model.pth",
    config_path="models/ugtts-multi/config.json",
    description="Multi-speaker TTS model with multiple voices",
    language="en",
    speakers=["PT", "EN", "ES", "FR"],
    default_speaker="PT",
    length_scale=1.0
)
```

### Model Directory Structure

```
models/
├── ugtts-single/
│   ├── best_model.pth
│   └── config.json
├── ugtts-multi/
│   ├── best_model.pth
│   └── config.json
└── spanish-tts/
    ├── best_model.pth
    └── config.json
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### TTS Endpoints

#### Get Available Models
```bash
GET /tts/models
```

#### Load a Model
```bash
POST /tts/models/{model_name}/load
```

#### Unload a Model
```bash
DELETE /tts/models/{model_name}/unload
```

#### Synthesize Speech (JSON)
```bash
POST /tts/synthesize
Content-Type: application/json

{
    "text": "Hello, world!",
    "model_name": "ugtts_multi",
    "speaker": "PT",
    "length_scale": 1.0
}
```

#### Synthesize Speech (Form Data)
```bash
POST /tts/synthesize/file
Content-Type: multipart/form-data

text: "Hello, world!"
model_name: "ugtts_multi"
speaker: "PT"
length_scale: 1.0
```

#### Get Model Information
```bash
GET /tts/models/{model_name}/info
```

### Health Check
```bash
GET /health
```

## Usage Examples

### Python Client Example

```python
import requests

# Server URL
base_url = "http://localhost:8000"

# Get available models
response = requests.get(f"{base_url}/tts/models")
models = response.json()
print(f"Available models: {[m['name'] for m in models['models']]}")

# Load a model
model_name = "ugtts_multi"
requests.post(f"{base_url}/tts/models/{model_name}/load")

# Synthesize speech
tts_data = {
    "text": "Hello, this is a test of the TTS system.",
    "model_name": model_name,
    "speaker": "PT",
    "length_scale": 1.0
}

response = requests.post(f"{base_url}/tts/synthesize", json=tts_data)
result = response.json()

if result["success"]:
    print(f"Audio generated: {result['audio_path']}")
    print(f"Duration: {result['audio_info']['duration']:.2f} seconds")
else:
    print(f"Error: {result['error']}")
```

### cURL Examples

```bash
# Get all models
curl -X GET "http://localhost:8000/tts/models"

# Load a model
curl -X POST "http://localhost:8000/tts/models/ugtts_multi/load"

# Synthesize speech
curl -X POST "http://localhost:8000/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "model_name": "ugtts_multi",
    "speaker": "PT",
    "length_scale": 1.0
  }'

# Get audio file directly
curl -X POST "http://localhost:8000/tts/synthesize/file" \
  -F "text=Hello, world!" \
  -F "model_name=ugtts_multi" \
  -F "speaker=PT" \
  -F "length_scale=1.0" \
  --output output.wav
```

## Configuration

### Environment Variables

You can set the following environment variables:

- `MODELS_DIR`: Base directory for models (default: "models")
- `LOG_LEVEL`: Logging level (default: "INFO")

### Model Parameters

- `length_scale`: Controls speech speed (default: 1.0)
  - Values < 1.0: Faster speech
  - Values > 1.0: Slower speech
- `speaker`: Speaker ID for multi-speaker models
- `text`: Input text to synthesize

## Development

### Project Structure

```
dcshci-nlp-server/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── config/
│   ├── __init__.py
│   └── models_config.py  # Model configuration
├── services/
│   ├── __init__.py
│   └── tts_service.py    # TTS service implementation
└── models/               # Model files directory
    └── .gitkeep
```

### Adding New Models

1. Place your model files in the `models/` directory
2. Update the configuration in `config/models_config.py`
3. Restart the server

### Extending the Server

The server is designed to be easily extensible:

- Add new NLP services in the `services/` directory
- Extend the configuration system for new model types
- Add new API endpoints in `app.py`

## Troubleshooting

### Common Issues

1. **Model files not found**: Ensure model paths in configuration match actual file locations
2. **CUDA out of memory**: Reduce batch size or use CPU-only mode
3. **Audio quality issues**: Adjust `length_scale` parameter or check model quality

### Logs

The server uses structured logging with loguru. Check the console output for detailed information about model loading, synthesis, and errors.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 