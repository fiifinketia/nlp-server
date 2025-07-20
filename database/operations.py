"""
Database operations for NLP Server
"""
import uuid as uuid_lib
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import selectinload
from loguru import logger
import random

from .models import EvalText, EvalMetric, ModelUsage, SystemLog, AudioFile, UserSession

class DatabaseOperations:
    """Database operations class"""

    @staticmethod
    async def create_eval_text(
        db: AsyncSession,
        text: str,
        language: str = "akan",
        category: Optional[str] = None,
        difficulty: str = "medium"
    ) -> EvalText:
        """Create a new evaluation text"""
        try:
            eval_text = EvalText(
                text=text,
                language=language,
                category=category,
                difficulty=difficulty
            )
            db.add(eval_text)
            await db.commit()
            await db.refresh(eval_text)
            logger.info(f"Created evaluation text with ID: {eval_text.id}")
            return eval_text
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create evaluation text: {e}")
            raise

    @staticmethod
    async def get_random_eval_text(db: AsyncSession, language: Optional[str] = None) -> Optional[EvalText]:
        """Get a random evaluation text"""
        try:
            query = select(EvalText).where(EvalText.is_active == True)
            if language:
                query = query.where(EvalText.language == language)

            result = await db.execute(query)
            texts = result.scalars().all()

            if not texts:
                logger.warning("No evaluation texts found")
                return None

            selected_text = random.choice(texts)
            logger.info(f"Selected evaluation text ID: {selected_text.id}")
            return selected_text
        except Exception as e:
            logger.error(f"Failed to get random evaluation text: {e}")
            return None

    @staticmethod
    async def create_eval_metric(
        db: AsyncSession,
        eval_text_id: int,
        original_text: str,
        model_name: str,
        audio_path: str,
        corrected_text: Optional[str] = None,
        speaker: Optional[str] = None,
        length_scale: float = 1.0,
        autocorrect_used: bool = False,
        synthesis_duration: Optional[float] = None,
        audio_duration: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> EvalMetric:
        """Create a new evaluation metric"""
        try:
            eval_metric = EvalMetric(
                eval_text_id=eval_text_id,
                original_text=original_text,
                corrected_text=corrected_text,
                model_name=model_name,
                speaker=speaker,
                length_scale=length_scale,
                audio_path=audio_path,
                autocorrect_used=autocorrect_used,
                synthesis_duration=synthesis_duration,
                audio_duration=audio_duration,
                meta=meta,
            )
            db.add(eval_metric)
            await db.commit()
            await db.refresh(eval_metric)
            logger.info(f"Created evaluation metric with UUID: {eval_metric.uuid}")
            return eval_metric
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create evaluation metric: {e}")
            raise

    @staticmethod
    async def update_eval_metric_score(
        db: AsyncSession,
        uuid: uuid_lib.UUID,
        mos_score: int,
        user_feedback: Optional[str] = None
    ) -> Optional[EvalMetric]:
        """Update MOS score for an evaluation metric"""
        try:
            query = select(EvalMetric).where(EvalMetric.uuid == uuid)
            result = await db.execute(query)
            eval_metric = result.scalar_one_or_none()

            if not eval_metric:
                logger.warning(f"Evaluation metric with UUID {uuid} not found")
                return None

            eval_metric.mos_score = mos_score
            eval_metric.user_feedback = user_feedback
            eval_metric.updated_at = datetime.utcnow()

            await db.commit()
            await db.refresh(eval_metric)
            logger.info(f"Updated MOS score for UUID: {uuid}")
            return eval_metric
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to update evaluation metric score: {e}")
            raise

    @staticmethod
    async def get_eval_metric_by_uuid(db: AsyncSession, uuid: uuid_lib.UUID) -> Optional[EvalMetric]:
        """Get evaluation metric by UUID"""
        try:
            query = select(EvalMetric).where(EvalMetric.uuid == uuid)
            result = await db.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get evaluation metric by UUID: {e}")
            return None

    @staticmethod
    async def create_audio_file(
        db: AsyncSession,
        filename: str,
        file_path: str,
        model_name: str,
        original_text: str,
        corrected_text: Optional[str] = None,
        speaker: Optional[str] = None,
        length_scale: float = 1.0,
        file_size: Optional[int] = None,
        duration: Optional[float] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        eval_metric_uuid: Optional[uuid_lib.UUID] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AudioFile:
        """Create a new audio file record"""
        try:
            audio_file = AudioFile(
                filename=filename,
                file_path=file_path,
                model_name=model_name,
                original_text=original_text,
                corrected_text=corrected_text,
                speaker=speaker,
                length_scale=length_scale,
                file_size=file_size,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                eval_metric_uuid=eval_metric_uuid,
                meta=meta,
            )
            db.add(audio_file)
            await db.commit()
            await db.refresh(audio_file)
            logger.info(f"Created audio file record: {audio_file.id}")
            return audio_file
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create audio file record: {e}")
            raise

    @staticmethod
    async def update_model_usage(
        db: AsyncSession,
        model_name: str,
        model_type: str,
        processing_time: float
    ) -> ModelUsage:
        """Update model usage statistics"""
        try:
            # Try to get existing record
            query = select(ModelUsage).where(
                ModelUsage.model_name == model_name,
                ModelUsage.model_type == model_type
            )
            result = await db.execute(query)
            model_usage = result.scalar_one_or_none()

            if model_usage:
                # Update existing record
                model_usage.usage_count += 1
                model_usage.total_processing_time += processing_time
                model_usage.average_processing_time = (
                    model_usage.total_processing_time / model_usage.usage_count
                )
                model_usage.last_used = datetime.utcnow()
                model_usage.updated_at = datetime.utcnow()
            else:
                # Create new record
                model_usage = ModelUsage(
                    model_name=model_name,
                    model_type=model_type,
                    usage_count=1,
                    total_processing_time=processing_time,
                    average_processing_time=processing_time,
                    last_used=datetime.utcnow()
                )
                db.add(model_usage)

            await db.commit()
            await db.refresh(model_usage)
            return model_usage
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to update model usage: {e}")
            raise

    @staticmethod
    async def log_system_event(
        db: AsyncSession,
        level: str,
        service: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> SystemLog:
        """Log a system event"""
        try:
            system_log = SystemLog(
                level=level,
                service=service,
                message=message,
                details=details,
                user_id=user_id,
                session_id=session_id
            )
            db.add(system_log)
            await db.commit()
            await db.refresh(system_log)
            return system_log
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to log system event: {e}")
            raise

    @staticmethod
    async def get_model_statistics(db: AsyncSession, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get model usage statistics"""
        try:
            query = select(ModelUsage)
            if model_name:
                query = query.where(ModelUsage.model_name == model_name)

            result = await db.execute(query)
            usages = result.scalars().all()

            stats = []
            for usage in usages:
                stats.append({
                    "model_name": usage.model_name,
                    "model_type": usage.model_type,
                    "usage_count": usage.usage_count,
                    "total_processing_time": usage.total_processing_time,
                    "average_processing_time": usage.average_processing_time,
                    "last_used": usage.last_used,
                    "created_at": usage.created_at
                })

            return stats
        except Exception as e:
            logger.error(f"Failed to get model statistics: {e}")
            return []

    @staticmethod
    async def get_evaluation_results(
        db: AsyncSession,
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get evaluation results with filters"""
        try:
            query = select(EvalMetric)

            if model_name:
                query = query.where(EvalMetric.model_name == model_name)
            if start_date:
                query = query.where(EvalMetric.created_at >= start_date)
            if end_date:
                query = query.where(EvalMetric.created_at <= end_date)

            query = query.order_by(EvalMetric.created_at.desc()).limit(limit)

            result = await db.execute(query)
            metrics = result.scalars().all()

            results = []
            for metric in metrics:
                results.append({
                    "uuid": str(metric.uuid),
                    "eval_text_id": metric.eval_text_id,
                    "original_text": metric.original_text,
                    "corrected_text": metric.corrected_text,
                    "model_name": metric.model_name,
                    "speaker": metric.speaker,
                    "length_scale": metric.length_scale,
                    "audio_path": metric.audio_path,
                    "mos_score": metric.mos_score,
                    "synthesis_duration": metric.synthesis_duration,
                    "audio_duration": metric.audio_duration,
                    "autocorrect_used": metric.autocorrect_used,
                    "user_feedback": metric.user_feedback,
                    "created_at": metric.created_at
                })

            return results
        except Exception as e:
            logger.error(f"Failed to get evaluation results: {e}")
            return []
