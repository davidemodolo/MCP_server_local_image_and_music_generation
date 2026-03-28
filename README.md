# Local AI Generation MCP Server (`local_ai_gen`)

This project runs a local MCP (Model Context Protocol) server that exposes tools for:

1. **Text to Image**
2. **Text to Music / Audio**
3. **Text to Speech**
4. **Image/Text to 3D Model**

## Models Used

- **Image Generation**: `segmind/SSD-1B`
  *A fast, distilled version of SDXL that works well on consumer GPUs.*
- **Music Generation**: `stabilityai/stable-audio-open-1.0` 
  *A high-quality open model for generating sound effects, short music tracks, and ambient audio.* *(Note: Requires accepting license terms)*
- **Speech Generation**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` 
  *A robust, multilingual text-to-speech model.*
- **3D Generation**: `stabilityai/TripoSR` 
  *A fast feed-forward model for image-to-3D reconstruction.*

## Requirements

- Python 3.10+
- NVIDIA GPU (8GB+ VRAM recommended for running everything smoothly)
- [Hugging Face User Access Token](https://huggingface.co/settings/tokens) (for `stable-audio-open-1.0`)

## Complete Setup Guide

### 1. Preparation and Hugging Face Authenication

The **Stable Audio Open** model is gated and requires you to accept its license before downloading.

1. Go to [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) on Hugging Face.
2. Log in and click to **Accept the Terms/License**.
3. Generate a User Access Token in your [Hugging Face Settings](https://huggingface.co/settings/tokens).
4. Run the Hugging Face CLI login locally:
    ```bash
    pip install -U "huggingface_hub"
    hf auth
    ```
    *(Paste your token when prompted. You do not need to add it as a git credential).*

### 2. Environment Installation

```bash
python -m venv .venv
source .venv/bin/activate
python setup.py
```

What `python setup.py` does:
1. Creates output directories (`generated_images`, `generated_audio`, `generated_models`, `models`)
2. Clones `third_party/TripoSR` if it is missing
3. Installs everything from `requirements.txt`
4. Compiles and installs `torchmcubes` against your active torch version (needed for 3D generation)
5. Installs `flash-attn`.

## Run the MCP server

To run the MCP server manually (via standard I/O):
```bash
python mcp_server/main.py
```

*Note: The first run of any specific tool will be slower because it has to download the weights for that model into your Hugging Face cache.*

## Smoke Test

You can run the included smoke test script to verify all models are working correctly:
```bash
python smoke_test_generate.py
```

## MCP Tools Exposed

- `generate_image`
- `generate_audio`
- `generate_speech`
- `generate_3d_model`
- `health_check`

## Notes

- Generated files are written to `generated_images`, `generated_audio`, and `generated_models` by default.
- For `generate_audio` and `generate_speech`, you can override the destination with:
- tool arg `output_dir` (highest priority), or
- env var `GENAI_OUTPUT_AUDIO_DIR`
- For `generate_image`, override destination with tool arg `output_dir` or env var `GENAI_OUTPUT_IMAGE_DIR`
- For `generate_3d_model`, override destination with tool arg `output_dir` or env var `GENAI_OUTPUT_MODEL_DIR`

## Using with MCP Clients (Cursor, Claude Desktop, etc.)

To use this server in an MCP-compatible client, add the following to your `mcp.json` (or the respective MCP configuration file for your client). Make sure to replace `<YOUR_PROJECT_PATH>` with the absolute path to where you cloned this repository:

```json
{
  "mcpServers": {
    "local_ai_gen": {
      "command": "<YOUR_PROJECT_PATH>/.venv/bin/python",
      "args": [
        "<YOUR_PROJECT_PATH>/mcp_server/main.py"
      ],
      "env": {}
    }
  }
}
```
