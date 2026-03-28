#!/usr/bin/env python3
"""Speech generation utilities using Qwen3-TTS."""

import os
import uuid
import logging
import sys
from contextlib import redirect_stdout
from datetime import datetime
from typing import Any, Optional

import soundfile as sf
import torch

from config import (
    GENERATED_AUDIO_DIR,
    QWEN3_TTS_DEFAULT_LANGUAGE,
    QWEN3_TTS_DEFAULT_VOICE,
    QWEN3_TTS_MODEL,
)


logger = logging.getLogger(__name__)


class SpeechGenerator:
    """Text-to-speech generator using Qwen3-TTS."""

    def __init__(
        self,
        model_name: str = QWEN3_TTS_MODEL,
        default_voice: str = QWEN3_TTS_DEFAULT_VOICE,
        default_language: str = QWEN3_TTS_DEFAULT_LANGUAGE,
    ):
        self.model_name = model_name
        self.default_voice = default_voice
        self.default_language = default_language
        self.model: Optional[Any] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._voice_aliases = {
            "default": "Vivian",
            "vivian": "Vivian",
            "ryan": "Ryan",
            "serena": "Serena",
            "aiden": "Aiden",
            "uncle_fu": "Uncle_Fu",
            "dylan": "Dylan",
            "eric": "Eric",
        }

    def _load_model(self) -> None:
        if self.model is not None:
            return

        try:
            with redirect_stdout(sys.stderr):
                from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise ImportError(
                "qwen-tts package required for speech generation. "
                "Install with: pip install -U qwen-tts"
            ) from exc

        with redirect_stdout(sys.stderr):
            self.model = Qwen3TTSModel.from_pretrained(self.model_name)
        logger.info("Loaded Qwen3-TTS model on %s", self.device)

    def _resolve_voice(self, voice: Optional[str]) -> str:
        raw = str(voice or self.default_voice).strip()
        key = raw.lower()
        candidate = self._voice_aliases.get(key, raw)

        if self.model is not None and hasattr(self.model, "get_supported_speakers"):
            supported = self.model.get_supported_speakers() or []
            if supported and candidate not in supported:
                fallback = self._voice_aliases["default"]
                logger.warning(
                    "voice '%s' not in supported speakers, falling back to '%s'",
                    candidate,
                    fallback,
                )
                return fallback

        return candidate

    def _resolve_language(self, language: Optional[str]) -> str:
        candidate = str(language or self.default_language).strip() or "Auto"

        if self.model is not None and hasattr(self.model, "get_supported_languages"):
            supported = self.model.get_supported_languages() or []
            if supported and candidate not in supported:
                logger.warning(
                    "language '%s' not supported, falling back to 'Auto'", candidate
                )
                return "Auto"

        return candidate

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
            requested = f"speech_{timestamp}_{uuid.uuid4().hex[:8]}"

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

    def generate_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        seed: Optional[int] = None,
        output_filename: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """Generate speech from text using Qwen3-TTS and save as WAV."""
        if not text or not str(text).strip():
            raise ValueError("text is required")

        self._load_model()

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        resolved_voice = self._resolve_voice(voice)
        resolved_language = self._resolve_language(language)

        with redirect_stdout(sys.stderr):
            audios, sample_rate = self.model.generate_custom_voice(
                text=str(text).strip(),
                speaker=resolved_voice,
                language=resolved_language,
                non_streaming_mode=True,
            )

        if not audios:
            raise RuntimeError("Qwen3-TTS returned no audio")

        audio = audios[0]

        filepath = self._resolve_output_path(
            output_filename=output_filename,
            output_dir=output_dir,
        )
        sf.write(filepath, audio, int(sample_rate))

        logger.info(
            "Qwen speech saved to %s (voice=%s, language=%s)",
            filepath,
            resolved_voice,
            resolved_language,
        )
        return filepath, resolved_voice, resolved_language
