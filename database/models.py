"""
Database models for NLP Server
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid as uuid_lib
from .config import Base

class EvalText(Base):
    """Model for evaluation texts"""
    __tablename__ = "eval_texts"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    language = Column(String(10), default="akan")
    category = Column(String(50), nullable=True)
    difficulty = Column(String(20), default="medium")  # easy, medium, hard
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)

class EvalMetric(Base):
    """Model for evaluation metrics"""
    __tablename__ = "eval_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid_lib.uuid4, unique=True, nullable=False, index=True)
    eval_text_id = Column(Integer, nullable=False)  # Foreign key to eval_texts
    original_text = Column(Text, nullable=False)
    corrected_text = Column(Text, nullable=True)  # Text after autocorrect
    model_name = Column(String(100), nullable=False)
    speaker = Column(String(100), nullable=True)
    length_scale = Column(Float, default=1.0)
    audio_path = Column(String(500), nullable=False)
    mos_score = Column(Integer, nullable=True)  # 1-5 Mean Opinion Score
    synthesis_duration = Column(Float, nullable=True)  # Time taken to synthesize
    audio_duration = Column(Float, nullable=True)  # Duration of generated audio
    autocorrect_used = Column(Boolean, default=False)
    user_feedback = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)  # Additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class ModelUsage(Base):
    """Model for tracking model usage statistics"""
    __tablename__ = "model_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # tts, autocorrect, etc.
    usage_count = Column(Integer, default=0)
    total_processing_time = Column(Float, default=0.0)
    average_processing_time = Column(Float, default=0.0)
    last_used = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class SystemLog(Base):
    """Model for system logs and events"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, etc.
    service = Column(String(50), nullable=False)  # tts, autocorrect, etc.
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    user_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AudioFile(Base):
    """Model for tracking generated audio files"""
    __tablename__ = "audio_files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=True)  # Size in bytes
    duration = Column(Float, nullable=True)  # Duration in seconds
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    format = Column(String(10), default="wav")
    model_name = Column(String(100), nullable=False)
    original_text = Column(Text, nullable=False)
    corrected_text = Column(Text, nullable=True)
    speaker = Column(String(100), nullable=True)
    length_scale = Column(Float, default=1.0)
    metadata = Column(JSON, nullable=True)
    eval_metric_uuid = Column(UUID(as_uuid=True), nullable=True)  # Link to evaluation
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_deleted = Column(Boolean, default=False)

class UserSession(Base):
    """Model for tracking user sessions"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    total_requests = Column(Integer, default=0)
    total_synthesis_time = Column(Float, default=0.0)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)