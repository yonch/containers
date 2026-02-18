#!/usr/bin/env python3
"""TTS worker for Vast.ai GPU instances.

Reads pre-signed S3 URLs from environment variables:
  SOURCE_URL  — GET URL for the source text document
  OUTPUT_URL  — PUT URL for the output mp3
  VOICE_URL   — (optional) GET URL for a reference voice clip

Downloads source text, runs Chatterbox TTS with Whisper STT validation,
converts to mp3 with loudness normalization, and uploads the result.
"""

import os
import sys
import tempfile
import subprocess

import requests
import torch
import torchaudio
import soundfile as sf
import numpy as np
import pysbd


def download(url, path):
    """Download a URL to a local file."""
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return len(resp.content)


def segment_text(text, max_words=40):
    """Split text into segments of roughly max_words using sentence boundaries."""
    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = segmenter.segment(text)

    segments = []
    current = []
    word_count = 0
    for sent in sentences:
        words = len(sent.split())
        if word_count + words > max_words and current:
            segments.append(" ".join(current))
            current = [sent]
            word_count = words
        else:
            current.append(sent)
            word_count += words
    if current:
        segments.append(" ".join(current))

    return segments


def main():
    source_url = os.environ["SOURCE_URL"]
    output_url = os.environ["OUTPUT_URL"]
    voice_url = os.environ.get("VOICE_URL")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
        voice_path = None
        if voice_url:
            voice_path = os.path.join(tmpdir, "voice.wav")
            vbytes = download(voice_url, voice_path)
            print(f"Voice clip: {vbytes} bytes")

        # Load models
        from chatterbox.tts import ChatterboxTTS
        import whisper

        print("Loading Chatterbox...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)

        whisper_root = os.environ.get("WHISPER_DOWNLOAD_ROOT", "/models/whisper")
        print("Loading Whisper medium...")
        stt_model = whisper.load_model("medium", download_root=whisper_root, device=device)

        # Segment text
        segments = segment_text(text)
        print(f"Segments: {len(segments)}")

        # Synthesize each segment
        project_dir = os.path.join(tmpdir, "project")
        os.makedirs(project_dir)
        segment_files = []

        for i, seg_text in enumerate(segments):
            wav = tts_model.generate(seg_text, audio_prompt_path=voice_path)

            if torch.is_tensor(wav):
                wav_np = wav.cpu().numpy()
                if wav_np.ndim == 1:
                    wav_np = wav_np[None, :]  # (1, samples) for torchaudio
                wav_tensor = torch.from_numpy(wav_np)
            else:
                wav_np = np.atleast_2d(np.array(wav))
                wav_tensor = torch.from_numpy(wav_np)

            seg_path = os.path.join(project_dir, f"seg_{i:04d}.wav")
            torchaudio.save(seg_path, wav_tensor, tts_model.sr)

            # STT validation
            result = stt_model.transcribe(seg_path)
            stt_text = result["text"].strip()
            print(f"  [{i+1}/{len(segments)}] STT: {stt_text[:80]}...")

            segment_files.append(seg_path)

        # Concatenate segments
        list_path = os.path.join(project_dir, "segments.txt")
        with open(list_path, "w") as f:
            for seg_path in segment_files:
                f.write(f"file '{seg_path}'\n")

        combined_wav = os.path.join(project_dir, "combined.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", combined_wav],
            check=True, capture_output=True,
        )

        # Loudness normalization + mp3 conversion
        output_mp3 = os.path.join(project_dir, "output.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", combined_wav,
             "-filter:a", "loudnorm=I=-16:LRA=11:TP=-1.5",
             "-codec:a", "libmp3lame", "-q:a", "2",
             output_mp3],
            check=True, capture_output=True,
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
