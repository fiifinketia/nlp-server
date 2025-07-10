from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import os


class ModelType(str, Enum):
    SINGLE_SPEAKER = "single_speaker"
    MULTI_SPEAKER = "multi_speaker"


class TTSModelConfig(BaseModel):
    """Configuration for a TTS model"""
    name: str = Field(..., description="Model name/identifier")
    model_type: ModelType = Field(..., description="Type of TTS model")
    model_path: str = Field(..., description="Path to the model file (.pth)")
    config_path: str = Field(..., description="Path to the config file (.json)")
    description: Optional[str] = Field(None, description="Model description")
    language: str = Field(..., description="Language code (e.g., 'aka')")
    speakers: Optional[List[str]] = Field(None, description="Available speakers for multi-speaker models")
    default_speaker: Optional[str] = Field(None, description="Default speaker for multi-speaker models")
    length_scale: float = Field(1.0, description="Default length scale for speech generation")
    is_active: bool = Field(True, description="Whether the model is active and available")
    
    class Config:
        use_enum_values = True


class ModelsConfig(BaseModel):
    """Main configuration for all NLP models"""
    models_dir: str = Field("models", description="Base directory for all models")
    tts_models: Dict[str, TTSModelConfig] = Field(default_factory=dict, description="TTS models configuration")
    
    def get_active_tts_models(self) -> Dict[str, TTSModelConfig]:
        """Get only active TTS models"""
        return {name: model for name, model in self.tts_models.items() if model.is_active}
    
    def get_model_by_name(self, model_name: str) -> Optional[TTSModelConfig]:
        """Get a specific model by name"""
        return self.tts_models.get(model_name)
    
    def validate_model_paths(self) -> List[str]:
        """Validate that all model files exist and return list of errors"""
        errors = []
        for model_name, model_config in self.tts_models.items():
            if not os.path.exists(model_config.model_path):
                errors.append(f"Model file not found for {model_name}: {model_config.model_path}")
            if not os.path.exists(model_config.config_path):
                errors.append(f"Config file not found for {model_name}: {model_config.config_path}")
        return errors


# Default configuration - you can modify this or load from a file
DEFAULT_MODELS_CONFIG = ModelsConfig(
    models_dir="models",
    tts_models={
        "ugtts_multispeaker": TTSModelConfig(
            name="ugtts_multispeaker",
            model_type=ModelType.MULTI_SPEAKER,
            model_path="models/ugtts-multispeaker/best_model.pth",
            config_path="models/ugtts-multispeaker/config.json",
            description="Multi-speaker TTS model for Akan language with multiple voices",
            language="aka",
            speakers=["PT", "IM", "AN"],
            default_speaker="PT",
            length_scale=1.0,
        ),
        "ugtts_im_speaker_v4": TTSModelConfig(
            name="ugtts_im_speaker_v4",
            model_type=ModelType.SINGLE_SPEAKER,
            model_path="models/ugtts-im-speaker-v4/model.pth",
            config_path="models/ugtts-im-speaker-v4/config.json",
            description="Single-speaker TTS model for Akan language with one voice (IM)",
            language="aka",
            length_scale=1.0,
        ),
        "ugtts_im_speaker_v3": TTSModelConfig(
            name="ugtts_im_speaker_v3",
            model_type=ModelType.SINGLE_SPEAKER,
            model_path="models/ugtts-im-speaker-v3/model.pth",
            config_path="models/ugtts-im-speaker-v3/config.json",
            description="Single-speaker TTS model for Akan language with one voice (IM)",
            language="aka",
            length_scale=1.0,
        ),
        "ugtts_im_speaker_v2_2": TTSModelConfig(
            name="ugtts_im_speaker_v2_2",
            model_type=ModelType.SINGLE_SPEAKER,
            model_path="models/ugtts-im-speaker-v2_2/model.pth",
            config_path="models/ugtts-im-speaker-v2_2/config.json",
            description="Single-speaker TTS model for Akan language with one voice (IM)",
            language="aka",
            length_scale=1.0,
        ),
    },
)
