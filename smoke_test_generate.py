#!/usr/bin/env python3
"""
Smoke test for local_ai_gen MCP server.
Tests image generation (SSD-1B), music generation (Stable Audio), speech synthesis (Qwen3-TTS), and optional 3D.
"""

from mcp_server.main import (
    generate_image,
    generate_audio,
    generate_speech,
    generate_3d_model,
)


def main():
    image_prompts = [
        "a cinematic sunrise over alpine mountains, ultra detailed",
        "a watercolor village by a river at golden hour",
    ]
    music_prompt = "uplifting electronic ambient with soft pads and a gentle beat"
    speech_text = "Hello, this is a test of the speech synthesis system."

    print("Loading models and running smoke generation...\n")

    # Test image generation (SSD-1B)
    print("=== Image Generation (SSD-1B) ===")
    try:
        image_results = generate_image(
            prompts=image_prompts,
            width=512,
            height=512,
            steps=20,
            guidance_scale=7.5,
            seed=42,
        )
        for item in image_results:
            print(f"✅ Image: {item['image_path']}\n")
    except Exception as e:
        print(f"❌ Image generation failed: {e}\n")
        return 1

    # Test music generation (Stable Audio Open)
    print("=== Music Generation (Stable Audio Open) ===")
    audio_path = generate_audio(
        prompt=music_prompt,
        duration=6.0,
        num_inference_steps=100,
        guidance_scale=7.0,
        seed=42,
    )
    print(f"✅ Music: {audio_path}\n")

    # Test speech synthesis (Qwen3-TTS)
    print("=== Speech Synthesis (Qwen3-TTS) ===")
    speech_result = generate_speech(
        text=speech_text,
        voice="Vivian",
        language="en",
        seed=42,
    )
    print(f"✅ Speech: {speech_result['speech_path']}")
    print(
        f"   Voice: {speech_result['voice']}, Language: {speech_result['language']}\n"
    )

    # Test 3D model generation (optional)
    print("=== 3D Model Generation (TripoSR) ===")
    model_result = generate_3d_model(
        image_path=image_results[0]["image_path"],
        output_format="glb",
        mc_resolution=64,
        seed=42,
    )
    print(f"✅ 3D Model: {model_result['model_path']}\n")

    print("✅ Smoke test COMPLETE")
    return 0


if __name__ == "__main__":
    exit(main())
