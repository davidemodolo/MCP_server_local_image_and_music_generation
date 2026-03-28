#!/usr/bin/env python3
"""
Main MCP tool server for local image, audio, and 3D generation using open source models.
"""

import os
import sys
import logging
from contextlib import redirect_stdout
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
    QWEN3_TTS_DEFAULT_VOICE,
    QWEN3_TTS_DEFAULT_LANGUAGE,
    MODEL3D_DEFAULT_FORMAT,
    MODEL3D_DEFAULT_MC_RESOLUTION,
    MODEL3D_DEFAULT_CHUNK_SIZE,
    MODEL3D_DEFAULT_FOREGROUND_RATIO,
    MODEL3D_DEFAULT_REMOVE_BACKGROUND,
    MODEL3D_MAX_MC_RESOLUTION,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "Missing MCP SDK. Install dependencies with: pip install -r requirements.txt"
    ) from exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("local_ai_gen")

image_generator: Optional[Any] = None
music_generator: Optional[Any] = None
speech_generator: Optional[Any] = None
model3d_generator: Optional[Any] = None


def _get_image_generator():
    global image_generator
    if image_generator is None:
        from image_gen.image_generator import ImageGenerator

        image_generator = ImageGenerator()
    return image_generator


def _get_music_generator():
    global music_generator
    if music_generator is None:
        from audio_gen.music_generator import MusicGenerator

        music_generator = MusicGenerator()
    return music_generator


def _get_speech_generator():
    global speech_generator
    if speech_generator is None:
        from audio_gen.speech_generator import SpeechGenerator

        speech_generator = SpeechGenerator()
    return speech_generator


def _get_model3d_generator():
    global model3d_generator
    if model3d_generator is None:
        from model3d_gen.model_generator import Model3DGenerator

        model3d_generator = Model3DGenerator()
    return model3d_generator


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
    output_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Generate one image for each prompt and return prompt/path pairs.

    Quality guidance for image->3D workflows:
    - Cleaner source images usually produce cleaner meshes.
    - For a fast-but-clean profile, prefer steps in the 12-20 range at 512x512,
      guidance around 4.5-5.5, and a negative prompt that suppresses seams,
      tiling, duplicated geometry, and texture-atlas artifacts.
    - If quality matters, generate 2-4 candidates and pick the cleanest image
      before sending it to generate_3d_model via image_path.
    """
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError("prompts must be a non-empty list of strings")
    if len(prompts) > IMAGE_MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size exceeds limit: {len(prompts)} > {IMAGE_MAX_BATCH_SIZE}"
        )

    with redirect_stdout(sys.stderr):
        image_paths = _get_image_generator().generate_images(
            prompts=prompts,
            width=width,
            height=height,
            steps=steps,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            seed=seed,
            output_filenames=output_filenames,
            output_dir=output_dir,
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
    output_dir: Optional[str] = None,
) -> str:
    """Generate audio from a prompt and return the output file path."""
    if not prompt:
        raise ValueError("prompt is required")

    with redirect_stdout(sys.stderr):
        audio_path = _get_music_generator().generate_audio(
            prompt=prompt,
            duration=duration,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            output_filename=output_filename,
            output_dir=output_dir,
        )
    return audio_path


@mcp.tool()
def generate_speech(
    text: str,
    voice: str = QWEN3_TTS_DEFAULT_VOICE,
    language: str = QWEN3_TTS_DEFAULT_LANGUAGE,
    seed: Optional[int] = None,
    output_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Generate speech from text using Qwen3-TTS and return output metadata."""
    if not text or not text.strip():
        raise ValueError("text is required")

    with redirect_stdout(sys.stderr):
        (
            speech_path,
            resolved_voice,
            resolved_language,
        ) = _get_speech_generator().generate_speech(
            text=text,
            voice=voice,
            language=language,
            seed=seed,
            output_filename=output_filename,
            output_dir=output_dir,
        )

    return {
        "speech_path": speech_path,
        "voice": resolved_voice,
        "language": resolved_language,
        "backend": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }


@mcp.tool()
def generate_3d_model(
    prompt: Optional[str] = None,
    image_path: Optional[str] = None,
    output_format: str = MODEL3D_DEFAULT_FORMAT,
    mc_resolution: int = MODEL3D_DEFAULT_MC_RESOLUTION,
    chunk_size: int = MODEL3D_DEFAULT_CHUNK_SIZE,
    remove_background: bool = MODEL3D_DEFAULT_REMOVE_BACKGROUND,
    foreground_ratio: float = MODEL3D_DEFAULT_FOREGROUND_RATIO,
    output_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    concept_image_steps: int = 12,
    concept_image_guidance_scale: float = 5.5,
    seed: Optional[int] = None,
) -> Dict[str, str]:
    """
    Generate a 3D model (GLB/OBJ) locally.

    Usage patterns:
    1) Text to 3D: provide `prompt` only. The tool first generates a concept image,
       then reconstructs a 3D mesh from that image.
    2) Image to 3D: provide `image_path`.

        Recommended quality playbook for cleaner meshes:
        - If you want cleaner source images for better 3D meshes, tune the image
            generation profile first.
        - Raise steps from 4 to 12-20.
        - Keep resolution at 512x512 but run 2-4 candidates and auto-pick the least
            artifacted one.
        - Slightly lower guidance (for example 4.5-5.5) to reduce overbaked edges.
        - Add a negative prompt targeting seams, tiling, duplicated limbs, and
            texture atlas artifacts.

        Practical recommendation:
        - For best control, use a two-step flow:
            1) call generate_image with the tuned settings,
            2) call generate_3d_model with image_path set to the selected image.
        - The prompt-only path is convenient and fast, but it uses a single concept
            image rather than a multi-candidate selection loop.
    """
    if not prompt and not image_path:
        raise ValueError("Provide either prompt or image_path")

    if prompt and image_path:
        raise ValueError("Provide only one of prompt or image_path")

    output_format = output_format.lower().strip()
    if output_format not in {"glb", "obj"}:
        raise ValueError("output_format must be 'glb' or 'obj'")

    mc_resolution = int(mc_resolution)
    if mc_resolution < 32 or mc_resolution > MODEL3D_MAX_MC_RESOLUTION:
        raise ValueError(
            f"mc_resolution must be between 32 and {MODEL3D_MAX_MC_RESOLUTION}"
        )

    source_image_path = image_path
    if prompt:
        image_results = generate_image(
            prompts=[prompt],
            width=1024,
            height=1024,
            steps=concept_image_steps,
            guidance_scale=concept_image_guidance_scale,
            seed=seed,
            output_dir=output_dir,
        )
        source_image_path = image_results[0]["image_path"]

    with redirect_stdout(sys.stderr):
        model_path = _get_model3d_generator().generate_model_from_image(
            image_path=source_image_path,
            output_format=output_format,
            mc_resolution=mc_resolution,
            chunk_size=chunk_size,
            remove_background=remove_background,
            foreground_ratio=foreground_ratio,
            output_filename=output_filename,
            output_dir=output_dir,
        )

    return {
        "model_path": model_path,
        "source_image_path": source_image_path,
        "backend": "stabilityai/TripoSR",
    }


@mcp.tool()
def health_check() -> dict:
    """Return a simple health payload."""
    return {"status": "healthy"}


if __name__ == "__main__":
    logger.info("Starting MCP server (stdio transport)")
    mcp.run()
