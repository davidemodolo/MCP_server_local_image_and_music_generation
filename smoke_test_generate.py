#!/usr/bin/env python3
"""Small smoke test that generates one image and one audio clip locally."""

from mcp_server.main import generate_image, generate_audio


def main():
    image_prompts = [
        "a cinematic sunrise over alpine mountains, ultra detailed",
        "a watercolor village by a river at golden hour",
    ]
    audio_prompt = "an uplifting ambient piano melody with soft strings"

    print("Loading models and running smoke generation...")

    image_results = generate_image(
        prompts=image_prompts,
        width=1024,
        height=1024,
        steps=20,
        guidance_scale=7.5,
        seed=42,
    )
    for item in image_results:
        print(f"Image generated for '{item['prompt']}': {item['image_path']}")

    audio_path = generate_audio(
        prompt=audio_prompt,
        duration=6.0,
        num_inference_steps=200,
        guidance_scale=3.5,
        seed=42,
    )
    print(f"Audio generated: {audio_path}")

    print("Smoke test complete.")


if __name__ == "__main__":
    main()
