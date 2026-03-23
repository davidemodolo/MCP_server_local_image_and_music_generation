#!/usr/bin/env python3
"""
Main MCP tool server for local image and audio generation using open source models.
"""

import os
import sys
import logging
from typing import Optional, Any, List, Dict

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    IMAGE_DEFAULT_WIDTH,
    IMAGE_DEFAULT_HEIGHT,
    IMAGE_DEFAULT_STEPS,
    IMAGE_DEFAULT_GUIDANCE_SCALE,
    IMAGE_MAX_BATCH_SIZE,
    AUDIO_DEFAULT_DURATION,
    AUDIO_DEFAULT_NUM_STEPS,
    AUDIO_DEFAULT_GUIDANCE_SCALE,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "Missing MCP SDK. Install dependencies with: pip install -r requirements.txt"
    ) from exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("genai-image-audio")

image_generator: Optional[Any] = None
audio_generator: Optional[Any] = None


def _get_image_generator():
    global image_generator
    if image_generator is None:
        from image_gen.image_generator import ImageGenerator

        image_generator = ImageGenerator()
    return image_generator


def _get_audio_generator():
    global audio_generator
    if audio_generator is None:
        from audio_gen.audio_generator import AudioGenerator

        audio_generator = AudioGenerator()
    return audio_generator


@mcp.tool()
def generate_image(
    prompts: List[str],
    width: int = IMAGE_DEFAULT_WIDTH,
    height: int = IMAGE_DEFAULT_HEIGHT,
    steps: int = IMAGE_DEFAULT_STEPS,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = IMAGE_DEFAULT_GUIDANCE_SCALE,
    seed: Optional[int] = None,
    output_filenames: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Generate one image for each prompt and return prompt/path pairs."""
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError("prompts must be a non-empty list of strings")
    if len(prompts) > IMAGE_MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size exceeds limit: {len(prompts)} > {IMAGE_MAX_BATCH_SIZE}"
        )

    image_paths = _get_image_generator().generate_images(
        prompts=prompts,
        width=width,
        height=height,
        steps=steps,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        seed=seed,
        output_filenames=output_filenames,
    )

    return [
        {"prompt": prompt, "image_path": image_path}
        for prompt, image_path in zip(prompts, image_paths)
    ]


@mcp.tool()
def generate_audio(
    prompt: str,
    duration: float = AUDIO_DEFAULT_DURATION,
    num_inference_steps: int = AUDIO_DEFAULT_NUM_STEPS,
    guidance_scale: float = AUDIO_DEFAULT_GUIDANCE_SCALE,
    seed: Optional[int] = None,
    output_filename: Optional[str] = None,
) -> str:
    """Generate audio from a prompt and return the output file path."""
    if not prompt:
        raise ValueError("prompt is required")

    audio_path = _get_audio_generator().generate_audio(
        prompt=prompt,
        duration=duration,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        output_filename=output_filename,
    )
    return audio_path


@mcp.tool()
def health_check() -> dict:
    """Return a simple health payload."""
    return {"status": "healthy"}


if __name__ == "__main__":
    logger.info("Starting MCP server (stdio transport)")
    mcp.run()
