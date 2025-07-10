#!/usr/bin/env python3
"""
Startup script for the NLP Server
Handles environment variables and server initialization
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import uvicorn
from loguru import logger
import nest_asyncio
from pyngrok import ngrok

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging configuration"""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "logs/nlp_server.log")
    
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(exist_ok=True)
    
    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        log_file,
        level=log_level,
        rotation=os.getenv("LOG_ROTATION", "10 MB"),
        retention=os.getenv("LOG_RETENTION", "7 days"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

def validate_environment():
    """Validate environment configuration"""
    models_dir = os.getenv("MODELS_DIR", "models")
    if not Path(models_dir).exists():
        logger.warning(f"Models directory '{models_dir}' does not exist. Creating it...")
        Path(models_dir).mkdir(exist_ok=True)
    
    # Check for CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("CUDA is not available. Using CPU.")
    except ImportError:
        logger.warning("PyTorch not available. TTS functionality may be limited.")

def main():
    """Main startup function"""
    print("NLP Server Startup")
    print("=" * 50)

    # Setup logging
    setup_logging()
    logger.info("Starting NLP Server...")

    # Validate environment
    validate_environment()

    # Get server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Server configuration: {host}:{port} (debug={debug})")

    # Set CUDA device if specified
    cuda_device = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        logger.info(f"Set CUDA_VISIBLE_DEVICES to {cuda_device}")

    # Apply nest_asyncio
    nest_asyncio.apply()

    # Start ngrok tunnel
    ngrok_tunnel = ngrok.connect(port)
    print("TTS Server Public URL:", ngrok_tunnel.public_url)
    print("Server ready")

    # Start the server
    try:
        logger.info("Starting FastAPI server...")
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info" if debug else "warning"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
