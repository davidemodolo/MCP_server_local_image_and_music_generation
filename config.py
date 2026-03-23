#!/usr/bin/env python3
"""
Configuration file for the MCP server
"""

# Server configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
DEBUG = True

# Image generation configuration
IMAGE_MODEL = "segmind/SSD-1B"
IMAGE_DEFAULT_WIDTH = 1024
IMAGE_DEFAULT_HEIGHT = 1024
IMAGE_DEFAULT_STEPS = 20
IMAGE_DEFAULT_GUIDANCE_SCALE = 7.5
IMAGE_MIN_SIZE = 256
IMAGE_MAX_STEPS = 150
IMAGE_MAX_BATCH_SIZE = 4

# Audio generation configuration
AUDIO_MODEL = "cvssp/audioldm2"
AUDIO_DEFAULT_DURATION = 5.0  # seconds
AUDIO_MAX_DURATION = 30.0  # seconds
AUDIO_DEFAULT_NUM_STEPS = 200
AUDIO_DEFAULT_GUIDANCE_SCALE = 3.5

# Performance settings
MAX_WORKERS = 4
CACHE_SIZE = 100

# Paths
GENERATED_IMAGES_DIR = "generated_images"
GENERATED_AUDIO_DIR = "generated_audio"
MODELS_DIR = "models"

# Model information for documentation
MODEL_INFO = {
    "image": {
        "name": "segmind/SSD-1B",
        "size": "2.5GB",
        "recommended_vram": "4GB+",
        "description": "Distilled SDXL model for fast, high-quality local image generation",
    },
    "audio": {
        "name": "cvssp/audioldm2",
        "size": "1.1GB",
        "recommended_vram": "2GB+",
        "description": "AudioLDM2 model for generating sound effects, music, and speech from text",
    },
}
