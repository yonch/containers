#!/usr/bin/env python3
"""TTS worker for Vast.ai GPU instances.

Reads pre-signed S3 URLs from environment variables:
  SOURCE_URL  — GET URL for the source text document
  OUTPUT_URL  — PUT URL for the output mp3
  VOICE_URL   — (optional) GET URL for a reference voice clip

Downloads source text, runs Chatterbox TTS via tts-audiobook-tool with
faster-whisper STT validation, and uploads the resulting mp3.
"""

import os
import subprocess
import sys
import tempfile

import requests

from tts_audiobook_tool.api import AudiobookConfig, create_audiobook, init


def download(url, path):
    """Download a URL to a local file."""
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return len(resp.content)


def progress(stage, completed, total):
    """Report progress to stdout."""
    print(f"  [{stage}] {completed}/{total}")


def main():
    source_url = os.environ["SOURCE_URL"]
    output_url = os.environ["OUTPUT_URL"]
    voice_url = os.environ.get("VOICE_URL")

    # Optional tuning knobs via env vars
    chatterbox_type = os.environ.get("CHATTERBOX_TYPE", "multilingual")
    exaggeration = float(os.environ.get("EXAGGERATION", "-1"))
    cfg = float(os.environ.get("CFG", "-1"))

    print("Initializing tts-audiobook-tool...")
    model_name = init()
    print(f"  Model: {model_name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download source text
        source_path = os.path.join(tmpdir, "source.txt")
        nbytes = download(source_url, source_path)
        with open(source_path) as f:
            text = f.read().strip()
        print(f"Source: {nbytes} bytes, {len(text)} chars")

        if not text:
            print("ERROR: source document is empty", file=sys.stderr)
            sys.exit(1)

        # Download reference voice clip (if provided)
        voice_path = ""
        if voice_url:
            voice_path = os.path.join(tmpdir, "voice.wav")
            vbytes = download(voice_url, voice_path)
            print(f"Voice clip: {vbytes} bytes")

        # Generate audiobook via tts-audiobook-tool API
        project_dir = os.path.join(tmpdir, "project")
        config = AudiobookConfig(
            project_dir=project_dir,
            voice_clone_path=voice_path,
            chatterbox_type=chatterbox_type,
            chatterbox_exaggeration=exaggeration,
            chatterbox_cfg=cfg,
            stt_enabled=True,
            stt_variant="medium",
            stt_download_root="/models/whisper",
            export_type="flac",
            normalization="default",
            max_retries=1,
        )

        print(f"Generating audiobook ({chatterbox_type})...")
        result = create_audiobook(text, config, on_progress=progress)

        if result.error:
            print(f"ERROR: {result.error}", file=sys.stderr)
            sys.exit(1)

        print(
            f"Generation complete: {result.num_succeeded}/{result.num_segments} segments OK"
        )

        # Convert to mp3 with loudness normalization
        output_mp3 = os.path.join(tmpdir, "output.mp3")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", result.output_path,
                "-filter:a", "loudnorm=I=-16:LRA=11:TP=-1.5",
                "-codec:a", "libmp3lame", "-q:a", "2",
                output_mp3,
            ],
            check=True,
            capture_output=True,
        )

        # Upload
        file_size = os.path.getsize(output_mp3)
        print(f"Uploading {file_size} bytes...")

        with open(output_mp3, "rb") as f:
            put_resp = requests.put(
                output_url,
                data=f,
                headers={"Content-Type": "audio/mpeg"},
                timeout=600,
            )
            put_resp.raise_for_status()

        print(f"Done: HTTP {put_resp.status_code}")


if __name__ == "__main__":
    main()
