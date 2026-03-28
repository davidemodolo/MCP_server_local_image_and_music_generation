#!/usr/bin/env python3
"""
Configuration file for the MCP server
"""

# Image generation configuration
IMAGE_DEFAULT_WIDTH = 1024
IMAGE_DEFAULT_HEIGHT = 1024
IMAGE_DEFAULT_STEPS = 20
IMAGE_DEFAULT_GUIDANCE_SCALE = 0.0
IMAGE_MAX_BATCH_SIZE = 4

# Audio generation configuration
AUDIO_MODEL = "stabilityai/stable-audio-open-1.0"
AUDIO_DEFAULT_DURATION = 10.0  # seconds
AUDIO_DEFAULT_NUM_STEPS = 200
AUDIO_DEFAULT_GUIDANCE_SCALE = 7.0

# Qwen3 Text-to-Speech configuration
QWEN3_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN3_TTS_DEFAULT_VOICE = "Vivian"  # or Ryan, Serena, etc.
QWEN3_TTS_DEFAULT_LANGUAGE = "Auto"

# Paths
GENERATED_IMAGES_DIR = "generated_images"
GENERATED_AUDIO_DIR = "generated_audio"
GENERATED_MODELS_DIR = "generated_models"

# 3D generation configuration (TripoSR image-to-3D)
MODEL3D_NAME = "stabilityai/TripoSR"
MODEL3D_DEFAULT_FORMAT = "glb"
MODEL3D_DEFAULT_MC_RESOLUTION = 96
MODEL3D_DEFAULT_CHUNK_SIZE = 8192
MODEL3D_DEFAULT_FOREGROUND_RATIO = 0.9
MODEL3D_DEFAULT_REMOVE_BACKGROUND = True
MODEL3D_MAX_MC_RESOLUTION = 320
