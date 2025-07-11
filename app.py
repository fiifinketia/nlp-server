from fastapi import FastAPI, HTTPException, UploadFile, File, Form, APIRouter
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import tempfile
from pathlib import Path
import pandas as pd
import uuid as uuid_lib
from datetime import datetime
import shutil

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

# Storage API models
class StorageFileInfo(BaseModel):
    name: str
    path: str
    size: int
    modified: str
    is_file: bool
    is_dir: bool

class StorageListResponse(BaseModel):
    files: List[StorageFileInfo]
    total_files: int
    total_size: int

class StorageUploadResponse(BaseModel):
    success: bool
    filename: str
    file_path: str
    size: int
    message: str

class StorageDeleteResponse(BaseModel):
    success: bool
    message: str

# --- MOS Evaluation Utilities and Endpoints (XLSX version) ---
EVAL_TEXTS_XLSX = "eval_texts.xlsx"
EVAL_METRICS_XLSX = "eval_metrics.xlsx"
STORAGE_DIR = "storage"

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

def get_random_eval_text():
    df = pd.read_excel(EVAL_TEXTS_XLSX)
    if df.empty:
        raise HTTPException(
            status_code=500, detail="No texts available for evaluation."
        )
    row = df.sample(1).iloc[0]
    return row["id"], row["text"]

def log_eval_metric(uuid, id, text, model_name, audio_path, mos_score=None):
    columns = [
        "uuid",
        "id",
        "text",
        "model_name",
        "audio_path",
        "mos_score",
        "timestamp",
    ]
    if os.path.exists(EVAL_METRICS_XLSX):
        df = pd.read_excel(EVAL_METRICS_XLSX)
    else:
        df = pd.DataFrame(columns=columns)
    entry = {
        "uuid": uuid,
        "id": id,
        "text": text,
        "model_name": model_name,
        "audio_path": audio_path,
        "mos_score": mos_score,
        "timestamp": datetime.utcnow().isoformat(),
    }
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_excel(EVAL_METRICS_XLSX, index=False)

eval_router = APIRouter()

class EvalSynthesizeRequest(BaseModel):
    model_name: str
    speaker: Optional[str] = None
    length_scale: Optional[float] = None

class EvalSynthesizeResponse(BaseModel):
    uuid: str
    id: int
    text: str
    audio_path: str

class EvalRateRequest(BaseModel):
    uuid: str
    mos_score: int

@eval_router.post("/eval/synthesize", response_model=EvalSynthesizeResponse)
async def eval_synthesize(request: EvalSynthesizeRequest):
    id, text = get_random_eval_text()
    eval_uuid = str(uuid_lib.uuid4())
    audio_filename = f"{request.model_name}_{eval_uuid}.wav"
    audio_path = os.path.join(STORAGE_DIR, audio_filename)
    if not tts_service.is_model_loaded(request.model_name):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' is not loaded. Please load it first.",
        )
    synth_path = tts_service.synthesize_speech(
        model_name=request.model_name,
        text=text,
        output_path=audio_path,
        speaker=request.speaker,
        length_scale=request.length_scale,
    )
    if synth_path is None:
        raise HTTPException(status_code=500, detail="Failed to synthesize speech.")
    log_eval_metric(eval_uuid, id, text, request.model_name, audio_path)
    return EvalSynthesizeResponse(
        uuid=eval_uuid, id=id, text=text, audio_path=audio_path
    )

@eval_router.post("/eval/rate")
async def eval_rate(request: EvalRateRequest):
    if not os.path.exists(EVAL_METRICS_XLSX):
        raise HTTPException(status_code=404, detail="No evaluation metrics found.")
    df = pd.read_excel(EVAL_METRICS_XLSX)
    idx = df.index[df["uuid"] == request.uuid].tolist()
    if not idx:
        raise HTTPException(status_code=404, detail="UUID not found in metrics log.")
    df.at[idx[0], "mos_score"] = request.mos_score
    df.at[idx[0], "timestamp"] = datetime.utcnow().isoformat()
    df.to_excel(EVAL_METRICS_XLSX, index=False)
    return {"status": "success"}

# Register the router
app.include_router(eval_router)

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

# Storage API endpoints
@app.get("/storage", response_model=StorageListResponse)
async def list_storage_files(subdirectory: Optional[str] = None):
    """List all files in the storage directory"""
    try:
        base_path = Path(STORAGE_DIR)
        if subdirectory:
            target_path = base_path / subdirectory
            if not target_path.exists() or not target_path.is_dir():
                raise HTTPException(status_code=404, detail=f"Subdirectory '{subdirectory}' not found")
        else:
            target_path = base_path
        
        if not target_path.exists():
            return StorageListResponse(files=[], total_files=0, total_size=0)
        
        files = []
        total_size = 0
        
        for item in target_path.iterdir():
            try:
                stat = item.stat()
                file_info = StorageFileInfo(
                    name=item.name,
                    path=str(item.relative_to(base_path)),
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    is_file=item.is_file(),
                    is_dir=item.is_dir()
                )
                files.append(file_info)
                if item.is_file():
                    total_size += stat.st_size
            except (OSError, PermissionError):
                # Skip files we can't access
                continue
        
        return StorageListResponse(
            files=files,
            total_files=len(files),
            total_size=total_size
        )
    except Exception as e:
        logger.error(f"Error listing storage files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list storage files: {str(e)}")

@app.get("/storage/{file_path:path}")
async def download_storage_file(file_path: str):
    """Download a file from the storage directory"""
    try:
        # Ensure the file path is within the storage directory
        full_path = Path(STORAGE_DIR) / file_path
        storage_path = Path(STORAGE_DIR).resolve()
        
        if not full_path.resolve().is_relative_to(storage_path):
            raise HTTPException(status_code=400, detail="Access denied: Path outside storage directory")
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"File '{file_path}' not found")
        
        if not full_path.is_file():
            raise HTTPException(status_code=400, detail=f"'{file_path}' is not a file")
        
        return FileResponse(
            str(full_path),
            filename=full_path.name,
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
