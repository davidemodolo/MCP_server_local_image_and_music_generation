#!/usr/bin/env python3
"""
Image generation using local open source models
"""

import os
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import uuid
from datetime import datetime
from typing import Optional, List


class ImageGenerator:
    def __init__(self, model_name="segmind/SSD-1B"):
        """
        Initialize image generator with a lightweight model
        """
        self.model_name = model_name
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Load the image generation model"""
        try:
            # Use distilled SDXL model
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )

            # Use DPM++ 2M scheduler for faster sampling
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            self.pipe = self.pipe.to(self.device)
            print(f"Loaded image generation model on {self.device}")

        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to CPU if CUDA fails
            if self.device == "cuda":
                self.device = "cpu"
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.model_name, torch_dtype=torch.float32, use_safetensors=True
                )
                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config
                )
                self.pipe = self.pipe.to(self.device)
                print(f"Loaded image generation model on CPU")

    def generate_images(
        self,
        prompts,
        width=1024,
        height=1024,
        steps=20,
        negative_prompt=None,
        guidance_scale=7.5,
        seed=None,
        output_filenames: Optional[List[str]] = None,
    ):
        """
        Generate images from text prompts in a single batched request.

        Args:
            prompts (list[str]): Text descriptions of images to generate
            width (int): Image width
            height (int): Image height
            steps (int): Number of inference steps
            negative_prompt (str | None): Elements to avoid in generation
            guidance_scale (float): Classifier-free guidance scale
            seed (int | None): Random seed for reproducibility
            output_filenames (list[str] | None): Custom output filenames

        Returns:
            list[str]: Paths to generated images
        """
        if self.pipe is None:
            raise Exception("Model not loaded")

        try:
            width = int(width)
            height = int(height)
            steps = int(steps)
            guidance_scale = float(guidance_scale)

            # Stable Diffusion requires dimensions divisible by 8.
            if width % 8 != 0 or height % 8 != 0:
                raise ValueError("Width and height must be divisible by 8")

            if width < 256 or height < 256:
                raise ValueError("Width and height must be at least 256")

            if steps < 1 or steps > 150:
                raise ValueError("Steps must be between 1 and 150")

            if guidance_scale < 0:
                raise ValueError("guidance_scale must be >= 0")

            if not isinstance(prompts, list) or len(prompts) == 0:
                raise ValueError("prompts must be a non-empty list of strings")

            cleaned_prompts: List[str] = []
            for prompt in prompts:
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Each prompt must be a non-empty string")
                cleaned_prompts.append(prompt.strip())

            if output_filenames is not None and len(output_filenames) != len(cleaned_prompts):
                raise ValueError("output_filenames length must match prompts length")

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))

            # Generate all images in one pipeline call to leverage batched tensor math.
            result = self.pipe(
                cleaned_prompts,
                width=width,
                height=height,
                num_inference_steps=steps,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            images = result.images

            # Create directory if it doesn't exist
            os.makedirs("generated_images", exist_ok=True)

            filepaths: List[str] = []
            for i, image in enumerate(images):
                if output_filenames:
                    filename = output_filenames[i]
                    if not filename.lower().endswith(".png"):
                        filename += ".png"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"generated_{timestamp}_{i}_{uuid.uuid4().hex[:8]}.png"
                filepath = os.path.join("generated_images", filename)
                image.save(filepath)
                filepaths.append(filepath)

            print(f"Saved {len(filepaths)} image(s) to generated_images")

            return filepaths

        except Exception as e:
            print(f"Error generating images: {e}")
            raise


# Example usage
if __name__ == "__main__":
    generator = ImageGenerator()
    # Example: generator.generate_images(["a highly detailed sci-fi city, cyberpunk"])
