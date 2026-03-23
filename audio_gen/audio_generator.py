#!/usr/bin/env python3
"""
Audio generation using local open source models
"""

import os
import torch
import uuid
import soundfile as sf
from datetime import datetime
from typing import Optional
from diffusers import AudioLDM2Pipeline


class AudioGenerator:
    def __init__(self, model_name="cvssp/audioldm2"):
        """
        Initialize audio generator with a lightweight model
        """
        self.model_name = model_name
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Load the audio generation model"""
        try:
            # Load the AudioLDM2 model
            self.pipe = AudioLDM2Pipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.pipe = self.pipe.to(self.device)
            print(f"Loaded audio generation model on {self.device}")

        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to CPU if CUDA fails
            if self.device == "cuda":
                self.device = "cpu"
                self.pipe = AudioLDM2Pipeline.from_pretrained(
                    self.model_name, torch_dtype=torch.float32
                )
                self.pipe = self.pipe.to(self.device)
                print(f"Loaded audio generation model on CPU")

    def generate_audio(
        self,
        prompt,
        duration=5.0,
        num_inference_steps=200,
        guidance_scale=3.5,
        seed=None,
        output_filename: Optional[str] = None,
    ):
        """
        Generate audio from text prompt

        Args:
            prompt (str): Text description of the audio to generate
            duration (float): Duration in seconds
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Classifier-free guidance scale
            seed (int | None): Random seed for reproducibility
            output_filename (str | None): Custom filename for the output audio (without extension)

        Returns:
            str: Path to generated audio file
        """
        if self.pipe is None:
            raise Exception("Model not loaded")

        try:
            duration = float(duration)
            num_inference_steps = int(num_inference_steps)
            guidance_scale = float(guidance_scale)

            if duration <= 0:
                raise ValueError("duration must be > 0")
            max_duration = min(duration, 30.0)

            if num_inference_steps <= 0:
                raise ValueError("num_inference_steps must be > 0")
            
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))

            # Generate audio (AudioLDM2 outputs mostly single channel 16000Hz numpy arrays)
            audio = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                audio_length_in_s=max_duration,
                guidance_scale=guidance_scale,
                generator=generator
            ).audios[0]

            # Save audio
            if output_filename:
                filename_stem = output_filename
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_stem = f"generated_{timestamp}_{uuid.uuid4().hex[:8]}"
            filepath_stem = os.path.join("generated_audio", filename_stem)

            # Create directory if it doesn't exist
            os.makedirs("generated_audio", exist_ok=True)

            filepath = f"{filepath_stem}.wav"
            
            # Write audio file (AudioLDM2 sample rate is natively 16000)
            sf.write(filepath, audio, 16000)

            print(f"Audio saved to {filepath}")

            return filepath

        except Exception as e:
            print(f"Error generating audio: {e}")
            raise


# Example usage
if __name__ == "__main__":
    generator = AudioGenerator()
    # Example: generator.generate_audio("A hammer hitting a wooden anvil")
