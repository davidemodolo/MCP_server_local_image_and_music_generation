#!/usr/bin/env python3
"""
3D model generation using TripoSR (image-to-3D).
"""

import os
import sys
import uuid
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from PIL import Image

from config import (
    MODEL3D_NAME,
    MODEL3D_DEFAULT_CHUNK_SIZE,
    MODEL3D_DEFAULT_MC_RESOLUTION,
    MODEL3D_DEFAULT_FOREGROUND_RATIO,
    MODEL3D_DEFAULT_REMOVE_BACKGROUND,
    GENERATED_MODELS_DIR,
)


class Model3DGenerator:
    def __init__(self, model_name: str = MODEL3D_NAME):
        self.model_name = model_name
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _repo_root() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _ensure_triposr_import_path(self) -> None:
        """Add common local checkout paths for TripoSR so `import tsr` can work."""
        repo_root = self._repo_root()
        candidate_paths = [
            os.path.join(repo_root, "third_party", "TripoSR"),
            os.path.join(repo_root, "models", "TripoSR"),
            os.environ.get("TRIPOSR_PATH", ""),
        ]

        for path in candidate_paths:
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

    def _lazy_load_model(self):
        if self.model is not None:
            return

        self._ensure_triposr_import_path()

        try:
            from tsr.system import TSR  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import TripoSR (`tsr`). Setup steps:\n"
                "1) git clone https://github.com/VAST-AI-Research/TripoSR.git third_party/TripoSR\n"
                "2) python setup.py\n"
                "Optional: set TRIPOSR_PATH to your local TripoSR folder."
            ) from exc

        self.model = TSR.from_pretrained(
            self.model_name,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(MODEL3D_DEFAULT_CHUNK_SIZE)
        self.model.to(self.device)

        logging.getLogger(__name__).info(
            "Loaded 3D generation model on %s", self.device
        )

    def _reload_on_cuda(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for GPU-only retry")
        self.device = "cuda:0"
        self.model = None
        torch.cuda.empty_cache()
        self._lazy_load_model()

    @staticmethod
    def _prepare_input_image(
        image_path: str,
        remove_background: bool,
        foreground_ratio: float,
    ) -> Image.Image:
        image = Image.open(image_path)

        if not remove_background:
            if image.mode == "RGBA":
                image_np = np.array(image).astype(np.float32) / 255.0
                image_np = (
                    image_np[:, :, :3] * image_np[:, :, 3:4]
                    + (1 - image_np[:, :, 3:4]) * 0.5
                )
                return Image.fromarray((image_np * 255.0).astype(np.uint8))
            return image.convert("RGB")

        try:
            import rembg
            from tsr.utils import remove_background as tsr_remove_background  # type: ignore
            from tsr.utils import resize_foreground  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Background removal dependencies are missing. Install with: python setup.py"
            ) from exc

        rembg_session = rembg.new_session()
        image = tsr_remove_background(image.convert("RGB"), rembg_session)
        image = resize_foreground(image, foreground_ratio)

        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = (
            image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
        )
        return Image.fromarray((image_np * 255.0).astype(np.uint8))

    @staticmethod
    def _resolve_output_path(
        output_filename: Optional[str],
        output_format: str,
        output_dir: Optional[str],
    ) -> str:
        env_output_dir = os.getenv("GENAI_OUTPUT_MODEL_DIR")
        output_root = (
            output_dir
            or env_output_dir
            or os.path.join(os.getcwd(), GENERATED_MODELS_DIR)
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

        if not filepath.lower().endswith(f".{output_format}"):
            filepath = f"{filepath}.{output_format}"

        target_dir = os.path.dirname(filepath) or "."
        os.makedirs(target_dir, exist_ok=True)
        return filepath

    def generate_model_from_image(
        self,
        image_path: str,
        output_format: str = "glb",
        mc_resolution: int = MODEL3D_DEFAULT_MC_RESOLUTION,
        chunk_size: int = MODEL3D_DEFAULT_CHUNK_SIZE,
        remove_background: bool = MODEL3D_DEFAULT_REMOVE_BACKGROUND,
        foreground_ratio: float = MODEL3D_DEFAULT_FOREGROUND_RATIO,
        output_filename: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """Generate a 3D mesh from one image and return output file path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"image_path does not exist: {image_path}")

        output_format = output_format.lower().strip()
        if output_format not in {"glb", "obj"}:
            raise ValueError("output_format must be 'glb' or 'obj'")

        self._lazy_load_model()

        if self.model is None:
            raise RuntimeError("3D model backend did not initialize")

        self.model.renderer.set_chunk_size(int(chunk_size))

        processed_image = self._prepare_input_image(
            image_path=image_path,
            remove_background=bool(remove_background),
            foreground_ratio=float(foreground_ratio),
        )

        try:
            with torch.no_grad():
                scene_codes = self.model([processed_image], device=self.device)

            # Lower marching-cubes resolution keeps meshes simpler and faster to generate.
            meshes = self.model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=int(mc_resolution),
            )
        except RuntimeError as e:
            message = str(e).lower()
            if self.device.startswith("cuda") and (
                "cudnn" in message or "cuda" in message
            ):
                torch.backends.cudnn.enabled = False
                logging.getLogger(__name__).warning(
                    "cuDNN disabled for GPU retry in 3D generation"
                )
                self._reload_on_cuda()
                with torch.no_grad():
                    scene_codes = self.model([processed_image], device=self.device)
                meshes = self.model.extract_mesh(
                    scene_codes,
                    has_vertex_color=True,
                    resolution=int(mc_resolution),
                )
            else:
                raise
        mesh = meshes[0]

        output_path = self._resolve_output_path(
            output_filename=output_filename,
            output_format=output_format,
            output_dir=output_dir,
        )
        mesh.export(output_path)

        logging.getLogger(__name__).info("3D model saved to %s", output_path)
        return output_path
