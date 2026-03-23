# Local MCP Server for Batched Image and Audio Generation

This repository provides a local MCP (Model Context Protocol) server exposing open source image and audio generation tools. Inference runs on your machine using fast and lightweight modern models.

## What You Get

- **Image Generation**: Driven by [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B) (A distilled SDXL model providing 1024x1024 high quality with half the size of SDXL). Batched prompts are supported.
- **Audio Generation**: Driven by [AudioLDM 2](https://huggingface.co/cvssp/audioldm2) for generating music, sound effects, and speech-style prompts.
- StdIO MCP server for integration with MCP-compatible clients.
- Local-first workflow (models downloaded on first run, outputs saved locally).
- Unified, clean dependencies using Hugging Face `diffusers`.

## Supported Models

### Image Generation
- **Model**: `segmind/SSD-1B`
- **Why**: Distilled version of Stable Diffusion XL (SDXL). It achieves massive quality leaps in prompt adherence, lighting, and anatomy, and it can accurately generate text—all while being 50% smaller and 60% faster than SDXL.
- **Download Size**: ~2.5GB
- **VRAM Requirement**: ~4GB+ for optimal hardware generation.

### Audio Generation
- **Model**: `cvssp/audioldm2`
- **Why**: Replaces earlier sequential generators like MusicGen. AudioLDM 2 gracefully handles a wide range of tasks—including complex music, environmental sound effects, and speech synthesis—in a single streamlined model.
- **Download Size**: ~1.1GB
- **VRAM Requirement**: ~2GB+ 

## Architecture

- `mcp_server/main.py`: MCP tool definitions and input validation
- `image_gen/image_generator.py`: SSD-1B pipeline and batched image generation
- `audio_gen/audio_generator.py`: AudioLDM2 loading and audio generation
- `config.py`: defaults and limits (including image batch limit)

## Requirements

- Python 3.8+
- Recommended for image generation: CUDA GPU with 6GB+ VRAM
- Recommended RAM: 16GB+
- For audio processing, FFmpeg might still be beneficial on your system depending on your underlying `soundfile` backends, but the core pipeline is entirely native via `diffusers` and `scipy`.

## Install

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 2. Run setup

```bash
python setup.py
```

`setup.py` creates output folders and installs the needed Python dependencies via `requirements.txt`.
*(Note: To ensure compatibility with AudioLDM2, `transformers` and `diffusers` are pinned to specific versions <4.47.0 and <0.33.0 respectively).*

## Run the MCP Server

```bash
python mcp_server/main.py
```

The process runs as a stdio MCP server and is intended to be started by your MCP client.

## MCP Tools

### `generate_image` (Batch API)

Generates one image per prompt in one request using **SSD-1B** (distilled SDXL). Recommended resolution is 1024x1024.

Request body:

```json
{
  "prompts": [
    "a cinematic sunrise over alpine mountains",
    "a watercolor village by a river"
  ],
  "width": 1024,
  "height": 1024,
  "steps": 20,
  "negative_prompt": "blurry, low quality",
  "guidance_scale": 7.5,
  "seed": 42
}
```

Response:

```json
[
  {
    "prompt": "a cinematic sunrise over alpine mountains",
    "image_path": "generated_images/generated_2026...0.png"
  },
  {
    "prompt": "a watercolor village by a river",
    "image_path": "generated_images/generated_2026...1.png"
  }
]
```

### `generate_audio`

Generates audio (music, sound effects, etc.) using **AudioLDM 2**.

Request body:

```json
{
  "prompt": "an uplifting ambient piano melody",
  "duration": 6.0,
  "num_inference_steps": 200,
  "guidance_scale": 3.5,
  "seed": 42
}
```

Returns a path like:

```text
generated_audio/generated_2026...wav
```

Notes:

- `duration` is capped at 30 seconds.
- `num_inference_steps` works best around `200` for optimal AudioLDM2 quality.

## Smoke Test

Verify the entire setup works by generating a sample image and audio track locally:

```bash
python test_setup.py
python smoke_test_generate.py
```

## Troubleshooting

- First run is slower because model weights are downloaded (around ~2.5GB for Image mapping, ~1.1GB for Audio mapping) into `models/` or your system HF cache.
- The stack will gracefully fall back to CPU inference if CUDA is missing, but rendering will take considerably longer.
