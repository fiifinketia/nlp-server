from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import tempfile
from pathlib import Path

from config.models_config import DEFAULT_MODELS_CONFIG, ModelsConfig, TTSModelConfig
from services.tts_service import TTSService
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="NLP Server",
    description="A comprehensive NLP server for TTS, ASR, LLMs, and translation tasks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
tts_service = TTSService()

# Load models configuration
models_config = DEFAULT_MODELS_CONFIG

# Pydantic models for API requests/responses
class TTSRequest(BaseModel):
    text: str
    model_name: str
    speaker: Optional[str] = None
    length_scale: Optional[float] = None

class TTSResponse(BaseModel):
    success: bool
    audio_path: Optional[str] = None
    error: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    audio_info: Optional[Dict[str, Any]] = None

class ModelInfoResponse(BaseModel):
    name: str
    type: str
    language: str
    description: Optional[str] = None
    speakers: Optional[List[str]] = None
    default_speaker: Optional[str] = None
    length_scale: float
    is_active: bool
    is_loaded: bool

class ModelsResponse(BaseModel):
    models: List[ModelInfoResponse]
    total_models: int
    loaded_models: int

@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup"""
    logger.info("Starting NLP Server...")
    
    # Validate model configurations
    errors = models_config.validate_model_paths()
    if errors:
        logger.warning("Some model files are missing:")
        for error in errors:
            logger.warning(f"  - {error}")
    
    # Load active models
    active_models = models_config.get_active_tts_models()
    for model_name, model_config in active_models.items():
        if tts_service.load_model(model_config):
            logger.info(f"Successfully loaded model: {model_name}")
        else:
            logger.error(f"Failed to load model: {model_name}")

@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "message": "NLP Server is running",
        "version": "1.0.0",
        "services": ["TTS", "ASR", "LLM", "Translation"],
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    loaded_models = len(tts_service.get_loaded_models())
    return {
        "status": "healthy",
        "loaded_tts_models": loaded_models,
        "total_tts_models": len(models_config.tts_models)
    }

# TTS Endpoints
@app.get("/tts/models", response_model=ModelsResponse)
async def get_tts_models():
    """Get all available TTS models"""
    models = []
    for model_name, model_config in models_config.tts_models.items():
        model_info = ModelInfoResponse(
            name=model_config.name,
            type=model_config.model_type,
            language=model_config.language,
            description=model_config.description,
            speakers=model_config.speakers,
            default_speaker=model_config.default_speaker,
            length_scale=model_config.length_scale,
            is_active=model_config.is_active,
            is_loaded=tts_service.is_model_loaded(model_name)
        )
        models.append(model_info)
    
    return ModelsResponse(
        models=models,
        total_models=len(models),
        loaded_models=len(tts_service.get_loaded_models())
    )

@app.post("/tts/models/{model_name}/load")
async def load_tts_model(model_name: str):
    """Load a specific TTS model"""
    model_config = models_config.get_model_by_name(model_name)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in configuration")
    
    if tts_service.is_model_loaded(model_name):
        return {"message": f"Model '{model_name}' is already loaded"}
    
    success = tts_service.load_model(model_config)
    if success:
        return {"message": f"Model '{model_name}' loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}'")

@app.delete("/tts/models/{model_name}/unload")
async def unload_tts_model(model_name: str):
    """Unload a specific TTS model"""
    success = tts_service.unload_model(model_name)
    if success:
        return {"message": f"Model '{model_name}' unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not loaded")

@app.post("/tts/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech from text"""
    try:
        # Validate model exists and is loaded
        if not tts_service.is_model_loaded(request.model_name):
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{request.model_name}' is not loaded. Please load it first."
            )
        
        # Generate speech
        audio_path = tts_service.synthesize_speech(
            model_name=request.model_name,
            text=request.text,
            speaker=request.speaker,
            length_scale=request.length_scale
        )
        
        if audio_path is None:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")
        
        # Get model and audio information
        model_info = tts_service.get_model_info(request.model_name)
        audio_info = tts_service.get_audio_info(audio_path)
        
        return TTSResponse(
            success=True,
            audio_path=audio_path,
            model_info=model_info,
            audio_info=audio_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in synthesize_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/tts/synthesize/file")
async def synthesize_speech_file(
    text: str = Form(...),
    model_name: str = Form(...),
    speaker: Optional[str] = Form(None),
    length_scale: Optional[float] = Form(1.0)
):
    """Synthesize speech and return the audio file directly"""
    try:
        # Validate model exists and is loaded
        if not tts_service.is_model_loaded(model_name):
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{model_name}' is not loaded. Please load it first."
            )
        
        # Generate speech
        audio_path = tts_service.synthesize_speech(
            model_name=model_name,
            text=text,
            speaker=speaker,
            length_scale=length_scale
        )
        
        if audio_path is None:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")
        
        # Return the audio file
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename=f"tts_output_{model_name}.wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in synthesize_speech_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/tts/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    model_info = tts_service.get_model_info(model_name)
    if model_info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or not loaded")
    
    return model_info

# Configuration endpoints
@app.get("/config/models")
async def get_models_config():
    """Get the current models configuration"""
    return {
        "models_dir": models_config.models_dir,
        "tts_models": {
            name: {
                "name": config.name,
                "type": config.model_type,
                "model_path": config.model_path,
                "config_path": config.config_path,
                "language": config.language,
                "description": config.description,
                "speakers": config.speakers,
                "default_speaker": config.default_speaker,
                "length_scale": config.length_scale,
                "is_active": config.is_active
            }
            for name, config in models_config.tts_models.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 