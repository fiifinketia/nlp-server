from fastapi import FastAPI, HTTPException, UploadFile, File, Form, APIRouter, Depends
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

from config.models_config import DEFAULT_MODELS_CONFIG, ModelsConfig, TTSModelConfig, AutocorrectModelConfig
from services.tts_service import TTSService
from services.autocorrect_service import AutocorrectService  # Fixed import
from loguru import logger
from database.config import get_db, init_database  # Import DB session and init
from database.operations import DatabaseOperations  # Import DB operations
from sqlalchemy.ext.asyncio import AsyncSession  # For type hinting

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
autocorrect_service = AutocorrectService()  # Fixed variable name

# Configure autocorrect model
autocorrect_config = AutocorrectModelConfig(
    name="ug-autocorrect",
    dictionary_path="/tmp/ug-autocorrect/base_dictionary.txt",  # One word per line
    language="akan"  # or your target language
)

# Load models configuration
models_config = DEFAULT_MODELS_CONFIG
autocorrect_service.load_model(autocorrect_config)  # Fixed service name

# Pydantic models for API requests/responses
class TTSRequest(BaseModel):
    text: str
    model_name: str
    speaker: Optional[str] = None
    length_scale: Optional[float] = None
    autocorrect: Optional[bool] = False  # Added autocorrect parameter

class TTSResponse(BaseModel):
    success: bool
    audio_path: Optional[str] = None
    error: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    audio_info: Optional[Dict[str, Any]] = None
    corrected: Optional[str] = None  # Added corrected text field

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
SAMPLES_DIR = "./samples/ugtts"  # Added samples directory

# Ensure directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)


def apply_autocorrect_to_text(text: str) -> tuple[str, bool]:
    """
    Apply autocorrect to text if autocorrect service is available
    Returns: (corrected_text, was_corrected)
    """
    if not autocorrect_service.is_model_loaded("ug-autocorrect"):
        logger.warning("Autocorrect model not loaded, skipping correction")
        return text, False

    try:
        # Perform spell checking on the sentence
        spellcheck_results = autocorrect_service.spellcheck_sentence(
            text, "ug-autocorrect"
        )

        corrected_words = []
        was_corrected = False

        for result in spellcheck_results:
            if result["correct"]:
                corrected_words.append(result["word"])
            else:
                # Use the first suggestion if available, otherwise keep original word
                if result["suggestions"]:
                    corrected_words.append(result["suggestions"][0])
                    was_corrected = True
                    logger.info(
                        f"Corrected '{result['word']}' to '{result['suggestions'][0]}'"
                    )
                else:
                    corrected_words.append(result["word"])

        corrected_text = " ".join(corrected_words)
        return corrected_text, was_corrected

    except Exception as e:
        logger.error(f"Error during autocorrect: {str(e)}")
        return text, False


# Excel file logic for evaluation texts


# Remove these functions as they're no longer needed
# def get_random_eval_text():
# def log_eval_metric(uuid, id, text, model_name, audio_path, mos_score=None):

eval_router = APIRouter()

# Remove these models as they're no longer needed
# class EvalSynthesizeRequest(BaseModel):
# class EvalSynthesizeResponse(BaseModel):
# class EvalRateRequest(BaseModel):


# Evaluation submission models
class Demographics(BaseModel):
    gender: str
    ageRange: str
    educationLevel: str
    akanSpeaking: str
    akanReading: str
    akanWriting: str
    akanType: str
    akanTypeOther: str = ""


class AudioEvaluation(BaseModel):
    audioId: str
    speaker: str
    audioQuality: str
    voicePleasantness: str
    naturalness: str
    continuity: str
    listeningEase: str
    understandingEffort: str
    pronunciationAnomalies: str
    deliverySpeed: str
    intelligibilityNaturalness: str = ""
    wordClarity: str
    soundDistinguishability: str
    telephoneUsability: str


class UserExperienceSurvey(BaseModel):
    # Intention to use
    IN2: str = ""
    IN3: str = ""
    IN4: str = ""
    # Effort
    EF1: str = ""
    EF2: str = ""
    EF3: str = ""
    # Credibility
    CR1: str = ""
    CR2: str = ""
    CR3: str = ""
    # Satisfaction
    SA1: str = ""
    SA2: str = ""
    SA3: str = ""
    # Perceived usefulness
    PU1: str = ""
    PU2: str = ""
    PU3: str = ""
    # Perceived ease of use
    PE1: str = ""
    PE2: str = ""
    PE3: str = ""
    # Attitude towards using
    AT1: str = ""
    AT2: str = ""
    AT3: str = ""
    AT4: str = ""
    # Behavioral intention
    BI1: str = ""
    BI2: str = ""
    BI3: str = ""
    # Actual use
    AU1: str = ""
    AU2: str = ""


class EvaluationSubmission(BaseModel):
    demographics: Demographics
    originalEvaluations: List[AudioEvaluation]
    synthesizedEvaluations: List[AudioEvaluation]
    userExperienceSurvey: UserExperienceSurvey


# Remove these endpoints as they're no longer needed
# @eval_router.post("/eval/synthesize", response_model=EvalSynthesizeResponse)
# @eval_router.post("/eval/rate")

# New endpoint for original samples
@eval_router.get("/evaluation/original-samples", response_model=StorageListResponse)
async def list_original_samples():
    """List all original audio samples in ./samples/ugtts"""
    try:
        samples_path = Path(SAMPLES_DIR)
        
        if not samples_path.exists():
            return StorageListResponse(files=[], total_files=0, total_size=0)
        
        files = []
        total_size = 0
        
        # Look for audio files (common audio extensions)
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg'}
        
        for item in samples_path.rglob('*'):
            if item.is_file() and item.suffix.lower() in audio_extensions:
                try:
                    stat = item.stat()
                    file_info = StorageFileInfo(
                        name=item.name,
                        path=str(item.relative_to(samples_path)),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        is_file=True,
                        is_dir=False
                    )
                    files.append(file_info)
                    total_size += stat.st_size
                except (OSError, PermissionError):
                    # Skip files we can't access
                    continue
        
        # Sort files by name
        files.sort(key=lambda x: x.name)
        
        return StorageListResponse(
            files=files,
            total_files=len(files),
            total_size=total_size
        )
    except Exception as e:
        logger.error(f"Error listing original samples: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list original samples: {str(e)}")

@eval_router.get("/evaluation/original-samples/{file_path:path}")
async def download_original_sample(file_path: str):
    """Download an original audio sample from ./samples/ugtts"""
    try:
        # Ensure the file path is within the samples directory
        full_path = Path(SAMPLES_DIR) / file_path
        samples_path = Path(SAMPLES_DIR).resolve()

        if not full_path.resolve().is_relative_to(samples_path):
            raise HTTPException(status_code=400, detail="Access denied: Path outside samples directory")

        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample '{file_path}' not found")

        if not full_path.is_file():
            raise HTTPException(status_code=400, detail=f"'{file_path}' is not a file")

        # Determine media type based on file extension
        media_type = "audio/wav"  # default
        extension = full_path.suffix.lower()
        if extension == '.mp3':
            media_type = "audio/mpeg"
        elif extension == '.m4a':
            media_type = "audio/mp4"
        elif extension == '.flac':
            media_type = "audio/flac"
        elif extension == '.aac':
            media_type = "audio/aac"
        elif extension == '.ogg':
            media_type = "audio/ogg"

        return FileResponse(
            str(full_path),
            filename=full_path.name,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading sample {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download sample: {str(e)}")


# Add evaluation submission endpoint
@eval_router.post("/evaluation/submit")
async def submit_evaluation(submission: EvaluationSubmission):
    """Submit a complete evaluation with demographics and audio evaluations"""
    try:
        # Use a single file for all submissions
        db_filepath = os.path.join(STORAGE_DIR, "db.xlsx")

        # Check if file exists and load existing data
        if os.path.exists(db_filepath):
            with pd.ExcelFile(db_filepath) as xls:
                existing_demographics = (
                    pd.read_excel(xls, "Demographics")
                    if "Demographics" in xls.sheet_names
                    else pd.DataFrame()
                )
                existing_original = (
                    pd.read_excel(xls, "Original_Evaluations")
                    if "Original_Evaluations" in xls.sheet_names
                    else pd.DataFrame()
                )
                existing_synthesized = (
                    pd.read_excel(xls, "Synthesized_Evaluations")
                    if "Synthesized_Evaluations" in xls.sheet_names
                    else pd.DataFrame()
                )
                existing_survey = (
                    pd.read_excel(xls, "User_Experience_Survey")
                    if "User_Experience_Survey" in xls.sheet_names
                    else pd.DataFrame()
                )
        else:
            existing_demographics = pd.DataFrame()
            existing_original = pd.DataFrame()
            existing_synthesized = pd.DataFrame()
            existing_survey = pd.DataFrame()

        # Prepare new data
        submission_id = str(uuid_lib.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Add submission ID and timestamp to demographics
        demographics_dict = submission.demographics.dict()
        demographics_dict["submission_id"] = submission_id
        demographics_dict["timestamp"] = timestamp
        new_demographics = pd.DataFrame([demographics_dict])

        # Add submission ID and timestamp to original evaluations
        original_evaluations = []
        for eval in submission.originalEvaluations:
            eval_dict = eval.dict()
            eval_dict["submission_id"] = submission_id
            eval_dict["timestamp"] = timestamp
            original_evaluations.append(eval_dict)
        new_original = pd.DataFrame(original_evaluations)

        # Add submission ID and timestamp to synthesized evaluations
        synthesized_evaluations = []
        for eval in submission.synthesizedEvaluations:
            eval_dict = eval.dict()
            eval_dict["submission_id"] = submission_id
            eval_dict["timestamp"] = timestamp
            synthesized_evaluations.append(eval_dict)
        new_synthesized = pd.DataFrame(synthesized_evaluations)

        # Add submission ID and timestamp to user experience survey
        survey_dict = submission.userExperienceSurvey.dict()
        survey_dict["submission_id"] = submission_id
        survey_dict["timestamp"] = timestamp
        new_survey = pd.DataFrame([survey_dict])

        # Combine existing and new data
        combined_demographics = pd.concat(
            [existing_demographics, new_demographics], ignore_index=True
        )
        combined_original = pd.concat(
            [existing_original, new_original], ignore_index=True
        )
        combined_synthesized = pd.concat(
            [existing_synthesized, new_synthesized], ignore_index=True
        )
        combined_survey = pd.concat([existing_survey, new_survey], ignore_index=True)

        # Save to single Excel file
        with pd.ExcelWriter(db_filepath, engine="openpyxl") as writer:
            combined_demographics.to_excel(
                writer, sheet_name="Demographics", index=False
            )
            combined_original.to_excel(
                writer, sheet_name="Original_Evaluations", index=False
            )
            combined_synthesized.to_excel(
                writer, sheet_name="Synthesized_Evaluations", index=False
            )
            combined_survey.to_excel(
                writer, sheet_name="User_Experience_Survey", index=False
            )

            # Summary sheet with statistics
            summary_data = {
                "Metric": [
                    "Total Submissions",
                    "Total Original Evaluations",
                    "Total Synthesized Evaluations",
                    "Total User Experience Surveys",
                    "Last Updated",
                    "File Path",
                ],
                "Value": [
                    len(combined_demographics),
                    len(combined_original),
                    len(combined_synthesized),
                    len(combined_survey),
                    datetime.utcnow().isoformat(),
                    db_filepath,
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        logger.info(f"Evaluation submission {submission_id} saved to {db_filepath}")

        return {
            "success": True,
            "message": "Evaluation submitted successfully",
            "submission_id": submission_id,
            "file_path": db_filepath,
            "total_submissions": len(combined_demographics),
            "total_original": len(combined_original),
            "total_synthesized": len(combined_synthesized),
            "total_surveys": len(combined_survey),
            "timestamp": timestamp,
        }

    except Exception as e:
        logger.error(f"Error submitting evaluation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit evaluation: {str(e)}"
        )


# Register the router
app.include_router(eval_router)

@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup"""
    logger.info("Starting NLP Server...")
    # await init_database()  # Database initialization disabled
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
        "services": ["TTS", "ASR", "LLM", "Translation", "Autocorrect"],
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    loaded_models = len(tts_service.get_loaded_models())
    loaded_autocorrect_models = len(autocorrect_service.get_loaded_models())
    return {
        "status": "healthy",
        "loaded_tts_models": loaded_models,
        "total_tts_models": len(models_config.tts_models),
        "loaded_autocorrect_models": loaded_autocorrect_models
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

        # Apply autocorrect if requested
        corrected_text = None
        text_to_synthesize = request.text

        if request.autocorrect:
            text_to_synthesize, was_corrected = apply_autocorrect_to_text(request.text)
            if was_corrected:
                corrected_text = text_to_synthesize

        # Generate unique filename in storage directory
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"tts_{request.model_name}_{timestamp}_{uuid_lib.uuid4().hex[:8]}.wav"
        )
        audio_path = os.path.join(STORAGE_DIR, filename)

        # Generate speech
        result_path = tts_service.synthesize_speech(
            model_name=request.model_name,
            text=text_to_synthesize,
            output_path=audio_path,
            speaker=request.speaker,
            length_scale=request.length_scale,
        )

        if result_path is None:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")

        # Get model and audio information
        model_info = tts_service.get_model_info(request.model_name)
        audio_info = tts_service.get_audio_info(audio_path)

        return TTSResponse(
            success=True,
            audio_path=audio_path,
            model_info=model_info,
            audio_info=audio_info,
            corrected=corrected_text
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
    length_scale: Optional[float] = Form(1.0),
    autocorrect: Optional[bool] = Form(False)  # Added autocorrect parameter
):
    """Synthesize speech and return the audio file directly"""
    try:
        # Validate model exists and is loaded
        if not tts_service.is_model_loaded(model_name):
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{model_name}' is not loaded. Please load it first."
            )

        # Apply autocorrect if requested
        text_to_synthesize = text

        if autocorrect:
            text_to_synthesize, was_corrected = apply_autocorrect_to_text(text)
            if was_corrected:
                logger.info(f"Applied autocorrect: '{text}' -> '{text_to_synthesize}'")

        # Generate unique filename in storage directory
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{model_name}_{timestamp}_{uuid_lib.uuid4().hex[:8]}.wav"
        audio_path = os.path.join(STORAGE_DIR, filename)

        # Generate speech
        result_path = tts_service.synthesize_speech(
            model_name=model_name,
            text=text_to_synthesize,
            output_path=audio_path,
            speaker=speaker,
            length_scale=length_scale,
        )

        if result_path is None:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")

        # Return the audio file
        return FileResponse(audio_path, media_type="audio/wav", filename=filename)

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

# Autocorrect endpoints
@app.get("/autocorrect/models")
async def get_autocorrect_models():
    """Get all loaded autocorrect models"""
    loaded_models = autocorrect_service.get_loaded_models()
    models_info = []
    
    for model_name, config in loaded_models.items():
        model_info = autocorrect_service.get_model_info(model_name)
        if model_info:
            models_info.append(model_info)
    
    return {
        "models": models_info,
        "total_models": len(models_info)
    }

@app.post("/autocorrect/check")
async def check_spelling(text: str, model_name: str = "ug-autocorrect"):
    """Check spelling of a single word"""
    if not autocorrect_service.is_model_loaded(model_name):
        raise HTTPException(status_code=400, detail=f"Autocorrect model '{model_name}' is not loaded")
    
    is_correct = autocorrect_service.check_spelling(text, model_name)
    suggestions = [] if is_correct else autocorrect_service.suggest_corrections(text, model_name)
    
    return {
        "word": text,
        "correct": is_correct,
        "suggestions": suggestions
    }

@app.post("/autocorrect/sentence")
async def spellcheck_sentence(text: str, model_name: str = "ug-autocorrect"):
    """Perform spell checking on a sentence"""
    if not autocorrect_service.is_model_loaded(model_name):
        raise HTTPException(status_code=400, detail=f"Autocorrect model '{model_name}' is not loaded")
    
    results = autocorrect_service.spellcheck_sentence(text, model_name)
    
    # Also provide a corrected version of the sentence
    corrected_words = []
    for result in results:
        if result["correct"]:
            corrected_words.append(result["word"])
        else:
            if result["suggestions"]:
                corrected_words.append(result["suggestions"][0])
            else:
                corrected_words.append(result["word"])
    
    corrected_sentence = " ".join(corrected_words)
    
    return {
        "original": text,
        "corrected": corrected_sentence,
        "words": results
    }

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
