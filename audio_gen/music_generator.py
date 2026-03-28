#!/usr/bin/env python3
"""Music generation utilities using Stable Audio Open 1.0."""

import os
import uuid
import logging
import sys
from contextlib import redirect_stdout
from datetime import datetime
from typing import Optional

import soundfile as sf
import torch
from einops import rearrange

from config import AUDIO_MODEL, GENERATED_AUDIO_DIR


logger = logging.getLogger(__name__)


class MusicGenerator:
    """Music/sound generator using Stable Audio Open 1.0."""

    def __init__(self, model_name: str = AUDIO_MODEL):
        self.model_name = model_name
        self.model = None
        self.model_config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        if self.model is not None:
            return

        try:
            from stable_audio_tools import get_pretrained_model
        except ImportError as exc:
            raise ImportError(
                "stable-audio-tools package required for music generation. "
                "Install with: pip install -U stable-audio-tools"
            ) from exc

        try:
            with redirect_stdout(sys.stderr):
                self.model, self.model_config = get_pretrained_model(self.model_name)
        except Exception as exc:
            message = str(exc)
            if "Cannot access gated repo" in message or "restricted" in message:
                raise RuntimeError(
                    "Stable Audio model access is gated on Hugging Face. "
                    "Request access to 'stabilityai/stable-audio-open-1.0' and authenticate via: "
                    "huggingface-cli login"
                ) from exc
            raise

        self.model = self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        logger.info("Loaded music generation model on %s", self.device)

    def _resolve_output_path(
        self,
        output_filename: Optional[str],
        output_dir: Optional[str],
    ) -> str:
        env_output_dir = os.getenv("GENAI_OUTPUT_AUDIO_DIR")
        output_root = (
            output_dir
            or env_output_dir
            or os.path.join(os.getcwd(), GENERATED_AUDIO_DIR)
        )

        if output_filename and str(output_filename).strip():
            requested = str(output_filename).strip()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            requested = f"generated_{timestamp}_{uuid.uuid4().hex[:8]}"

        has_directory = os.path.dirname(requested) != ""
        if has_directory or os.path.isabs(requested):
            filepath = requested
        else:
            filepath = os.path.join(output_root, requested)

        if not filepath.lower().endswith(".wav"):
            filepath = f"{filepath}.wav"

        target_dir = os.path.dirname(filepath) or "."
        os.makedirs(target_dir, exist_ok=True)
        return filepath

    def generate_audio(
        self,
        prompt: str,
        duration: float = 10.0,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        output_filename: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """Generate music/audio from a text prompt and save as WAV."""
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt is required")

        self._load_model()

        try:
            from stable_audio_tools.inference.generation import generate_diffusion_cond
        except ImportError as exc:
            raise ImportError(
                "stable-audio-tools package required for music generation. "
                "Install with: pip install -U stable-audio-tools"
            ) from exc

        duration = float(duration)
        num_inference_steps = int(num_inference_steps)
        guidance_scale = float(guidance_scale)

        if duration <= 0:
            raise ValueError("duration must be > 0")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")

        max_duration = min(duration, 47.0)

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        conditioning = [
            {
                "prompt": str(prompt).strip(),
                "seconds_start": 0,
                "seconds_total": max_duration,
            }
        ]

        sample_rate = int(self.model_config["sample_rate"])
        sample_size = int(sample_rate * max_duration)

        with torch.no_grad():
            output = generate_diffusion_cond(
                self.model,
                steps=num_inference_steps,
                cfg_scale=guidance_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=self.device,
            )

        audio = rearrange(output, "b d n -> d (b n)").to(torch.float32)
        peak = torch.max(torch.abs(audio))
        if peak > 0:
            audio = audio.div(peak)
        audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu().numpy().T

        filepath = self._resolve_output_path(
            output_filename=output_filename,
            output_dir=output_dir,
        )
        sf.write(filepath, audio, sample_rate)

        logger.info("Music saved to %s", filepath)
        return filepath


if __name__ == "__main__":
    generator = MusicGenerator()
    # Example: generator.generate_audio("A hammer hitting a wooden anvil")
