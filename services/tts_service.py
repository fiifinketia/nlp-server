import os
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path
import soundfile as sf
import numpy as np
from TTS.api import TTS
from loguru import logger

from config.models_config import TTSModelConfig, ModelType


class TTSService:
    """Service for managing TTS models and performing text-to-speech operations"""
    
    def __init__(self):
        self._models: Dict[str, TTS] = {}
        self._model_configs: Dict[str, TTSModelConfig] = {}
        
    def load_model(self, model_config: TTSModelConfig) -> bool:
        """Load a TTS model into memory"""
        try:
            logger.info(f"Loading TTS model: {model_config.name}")
            
            # Validate model files exist
            if not os.path.exists(model_config.model_path):
                logger.error(f"Model file not found: {model_config.model_path}")
                return False
                
            if not os.path.exists(model_config.config_path):
                logger.error(f"Config file not found: {model_config.config_path}")
                return False
            
            # Load the TTS model
            tts_model = TTS(
                model_path=model_config.model_path,
                config_path=model_config.config_path
            )
            
            # Store the model and config
            self._models[model_config.name] = tts_model
            self._model_configs[model_config.name] = model_config
            
            logger.info(f"Successfully loaded TTS model: {model_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TTS model {model_config.name}: {str(e)}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a TTS model from memory"""
        try:
            if model_name in self._models:
                del self._models[model_name]
                del self._model_configs[model_name]
                logger.info(f"Unloaded TTS model: {model_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unload TTS model {model_name}: {str(e)}")
            return False
    
    def get_loaded_models(self) -> Dict[str, TTSModelConfig]:
        """Get all currently loaded models"""
        return self._model_configs.copy()
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded"""
        return model_name in self._models
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize input text for TTS"""
        # Basic text cleaning - you can extend this based on your needs
        text = text.strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Basic punctuation normalization
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text
    
    def synthesize_speech(
        self,
        model_name: str,
        text: str,
        output_path: Optional[str] = None,
        speaker: Optional[str] = None,
        length_scale: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Synthesize speech from text using the specified model
        
        Args:
            model_name: Name of the TTS model to use
            text: Text to synthesize
            output_path: Path to save the audio file (optional)
            speaker: Speaker to use (for multi-speaker models)
            length_scale: Speed of speech generation
            **kwargs: Additional parameters for TTS generation
            
        Returns:
            Path to the generated audio file or None if failed
        """
        try:
            # Check if model is loaded
            if model_name not in self._models:
                logger.error(f"Model {model_name} is not loaded")
                return None
            
            model = self._models[model_name]
            config = self._model_configs[model_name]
            
            # Clean the input text
            clean_text = self.clean_text(text)
            
            # Set default values
            if speaker is None and config.model_type == ModelType.MULTI_SPEAKER:
                speaker = config.default_speaker
            if length_scale is None:
                length_scale = config.length_scale
            
            # Generate output path if not provided
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.wav')
            
            # Prepare TTS parameters
            tts_params = {
                'text': clean_text,
                'file_path': output_path,
                'length_scale': length_scale,
                **kwargs
            }
            
            # Add speaker parameter for multi-speaker models
            if config.model_type == ModelType.MULTI_SPEAKER and speaker:
                tts_params['speaker'] = speaker
                
                # Validate speaker
                if config.speakers and speaker not in config.speakers:
                    logger.warning(f"Speaker '{speaker}' not in available speakers: {config.speakers}")
            
            # Generate speech
            logger.info(f"Generating speech for model {model_name}, speaker: {speaker}")
            model.tts_to_file(**tts_params)
            
            logger.info(f"Speech generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to synthesize speech with model {model_name}: {str(e)}")
            return None
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model"""
        if model_name not in self._models:
            return None
        
        config = self._model_configs[model_name]
        return {
            'name': config.name,
            'type': config.model_type,
            'language': config.language,
            'description': config.description,
            'speakers': config.speakers,
            'default_speaker': config.default_speaker,
            'length_scale': config.length_scale,
            'is_active': config.is_active
        }
    
    def get_audio_info(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Get information about an audio file"""
        try:
            data, sample_rate = sf.read(audio_path)
            duration = len(data) / sample_rate
            
            return {
                'sample_rate': sample_rate,
                'duration': duration,
                'channels': len(data.shape) if len(data.shape) > 1 else 1,
                'samples': len(data),
                'file_size': os.path.getsize(audio_path)
            }
        except Exception as e:
            logger.error(f"Failed to get audio info for {audio_path}: {str(e)}")
            return None 