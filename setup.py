#!/usr/bin/env python3
"""Setup script to initialize the local MCP server environment."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def install_requirements(requirements_file: str, upgrade: bool = False):
    """Install required packages from a requirements file."""
    print(f"Installing packages from {requirements_file}...")
    command = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
    if upgrade:
        command.insert(4, "--upgrade")

    try:
        subprocess.check_call(command)
        print("Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)


def run_command(command: List[str], cwd: Optional[str] = None):
    """Run a shell command and fail fast on error."""
    try:
        subprocess.check_call(command, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Details: {e}")
        sys.exit(1)


def ensure_triposr_checkout(dest_dir: Path):
    """Clone TripoSR if missing and keep existing checkout otherwise."""
    if dest_dir.exists():
        print(f"Using existing TripoSR checkout: {dest_dir}")
        return

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning TripoSR into: {dest_dir}")
    run_command(
        [
            "git",
            "clone",
            "https://github.com/VAST-AI-Research/TripoSR.git",
            str(dest_dir),
        ]
    )


def install_torchmcubes():
    """Compile/install torchmcubes against the currently installed torch build."""
    print("Installing torchmcubes against current torch version...")

    # Required by torchmcubes' pyproject backend when build isolation is disabled.
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "scikit-build-core",
            "cmake",
            "ninja",
            "pybind11",
        ]
    )

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "--no-deps",
            "--no-build-isolation",
            "git+https://github.com/tatsy/torchmcubes.git",
        ]
    )


def create_directories():
    """Create necessary directories"""
    directories = ["generated_images", "generated_audio", "generated_models", "models"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def install_system_dependencies():
    """Install required system dependencies if running as root/sudo."""
    if os.geteuid() == 0:
        print("Installing system dependencies (sox)...")
        run_command(["apt-get", "update"])
        run_command(["apt-get", "install", "-y", "sox", "libsox-fmt-all"])
    else:
        print(
            "Skipping apt-get install (not root). To install system dependencies, run 'sudo apt-get install sox libsox-fmt-all' or build via Docker."
        )


def main():
    """Main setup function."""
    print("Setting up local MCP server for image/audio/speech/3D generation...")
    create_directories()

    install_system_dependencies()

    triposr_dir = Path("third_party") / "TripoSR"
    ensure_triposr_checkout(triposr_dir)

    # Unified stack for image/music/speech/3D.
    install_requirements("requirements.txt", upgrade=True)

    # Install torchmcubes against the active torch build to avoid ABI mismatch.
    install_torchmcubes()

    # install flash-attn
    print("Installing flash-attention...")
    run_command(
        [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"]
    )

    print("\nSetup complete!")


if __name__ == "__main__":
    main()
