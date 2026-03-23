#!/usr/bin/env python3
"""
Setup script to initialize the local MCP server environment
"""

import os
import subprocess
import sys


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)


def create_directories():
    """Create necessary directories"""
    directories = ["generated_images", "generated_audio", "models"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def main():
    """Main setup function"""
    print("Setting up local MCP server for image and audio generation...")
    create_directories()
    install_requirements()
    print("\nSetup complete!")

if __name__ == "__main__":
    main()
