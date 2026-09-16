import os
import shutil
from pathlib import Path

import yt_dlp
from pydub import AudioSegment


# ============================================================
# CONFIGURATION
# ============================================================

# macOS Downloads directory
DOWNLOAD_DIR = Path.home() / "Downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Find system executables
FFMPEG_PATH = shutil.which("ffmpeg")
FFPROBE_PATH = shutil.which("ffprobe")
DENO_PATH = shutil.which("deno")


# ============================================================
# VALIDATE DEPENDENCIES
# ============================================================

def _validate_dependencies() -> None:
    """
    Check that required system dependencies are available.
    """

    missing = []

    if not FFMPEG_PATH:
        missing.append(
            "FFmpeg/ffprobe\n"
            "Install with: brew install ffmpeg"
        )

    if not DENO_PATH:
        missing.append(
            "Deno\n"
            "Install with: brew install deno"
        )

    if missing:
        raise RuntimeError(
            "\n\nMissing required dependencies:\n\n"
            + "\n\n".join(missing)
            + "\n"
        )


# ============================================================
# YOUTUBE DOWNLOAD
# ============================================================

def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL and convert it to WAV.

    Returns:
        Absolute path to the downloaded WAV file.
    """

    _validate_dependencies()

    print("Downloading YouTube audio...")
    print(f"Download directory: {DOWNLOAD_DIR}")

    output_template = str(
        DOWNLOAD_DIR / "%(title)s.%(ext)s"
    )

    ydl_opts = {
        # Best available audio
        "format": "bestaudio/best",

        # Save inside ~/Downloads
        "outtmpl": output_template,

        # Don't download playlists
        "noplaylist": True,

        # Retry settings
        "retries": 3,
        "fragment_retries": 3,

        # JavaScript runtime required by modern YouTube extraction
        "js_runtimes": {
            "deno": {
                "path": DENO_PATH,
            }
        },

        # Explicitly tell yt-dlp where FFmpeg is
        "ffmpeg_location": FFMPEG_PATH,

        # Convert downloaded audio to WAV
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        # Show useful errors while developing
        "quiet": False,

        # Don't print unnecessary progress bars
        "noprogress": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            # yt-dlp's original downloaded filename
            original_path = Path(
                ydl.prepare_filename(info)
            )

            # FFmpegExtractAudio changes the extension to .wav
            wav_path = original_path.with_suffix(".wav")

            print(f"Downloaded file: {original_path}")
            print(f"Final WAV file: {wav_path}")

            if not wav_path.exists():

                # Sometimes yt-dlp may report a different
                # extension/path, so search for matching WAV files.
                possible_files = list(
                    DOWNLOAD_DIR.glob(
                        f"{original_path.stem}.*"
                    )
                )

                wav_files = [
                    file
                    for file in possible_files
                    if file.suffix.lower() == ".wav"
                ]

                if wav_files:
                    wav_path = wav_files[0]

                else:
                    raise FileNotFoundError(
                        "\nYouTube download completed, but the "
                        "expected WAV file was not found.\n\n"
                        f"Expected:\n{wav_path}\n\n"
                        f"Download directory:\n{DOWNLOAD_DIR}"
                    )

            print(
                f"✓ YouTube audio downloaded successfully:\n"
                f"  {wav_path}"
            )

            return str(wav_path)

    except Exception as exc:
        raise RuntimeError(
            f"\nFailed to download YouTube audio.\n\n"
            f"URL: {url}\n"
            f"Error: {exc}"
        ) from exc


# ============================================================
# AUDIO → WAV CONVERSION
# ============================================================

def convert_to_wav(input_path: str) -> str:
    """
    Convert any supported audio/video file to WAV.

    The resulting WAV is:
        - Mono
        - 16 kHz
        - Suitable for speech transcription

    Returns:
        Absolute path to the converted WAV file.
    """

    _validate_dependencies()

    input_file = Path(input_path).expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(
            f"\nInput file does not exist:\n{input_file}"
        )

    output_file = input_file.with_name(
        f"{input_file.stem}_converted.wav"
    )

    print(f"Converting file to WAV...")
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")

    try:
        audio = AudioSegment.from_file(
            str(input_file)
        )

        # Normalize for speech transcription
        audio = (
            audio
            .set_channels(1)
            .set_frame_rate(16000)
        )

        audio.export(
            str(output_file),
            format="wav"
        )

    except Exception as exc:
        raise RuntimeError(
            f"\nFailed to convert audio/video to WAV.\n\n"
            f"Input: {input_file}\n"
            f"Error: {exc}"
        ) from exc

    if not output_file.exists():
        raise FileNotFoundError(
            f"Conversion finished but WAV was not created:\n"
            f"{output_file}"
        )

    print(
        f"✓ Conversion successful:\n"
        f"  {output_file}"
    )

    return str(output_file)


# ============================================================
# NORMALIZE WAV
# ============================================================

def normalize_wav(wav_path: str) -> str:
    """
    Normalize an existing WAV file to:
        - Mono
        - 16 kHz

    This is useful for YouTube downloads because yt-dlp may
    produce a WAV with a different sample rate/channel count.
    """

    wav_file = Path(wav_path).expanduser().resolve()

    if not wav_file.exists():
        raise FileNotFoundError(
            f"WAV file not found:\n{wav_file}"
        )

    output_file = wav_file.with_name(
        f"{wav_file.stem}_16khz.wav"
    )

    print("Normalizing audio to 16 kHz mono...")

    try:
        audio = AudioSegment.from_wav(
            str(wav_file)
        )

        audio = (
            audio
            .set_channels(1)
            .set_frame_rate(16000)
        )

        audio.export(
            str(output_file),
            format="wav"
        )

    except Exception as exc:
        raise RuntimeError(
            f"\nFailed to normalize WAV.\n\n"
            f"Input: {wav_file}\n"
            f"Error: {exc}"
        ) from exc

    print(
        f"✓ Audio normalized:\n"
        f"  {output_file}"
    )

    return str(output_file)


# ============================================================
# CHUNK AUDIO
# ============================================================

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list[str]:
    """
    Split a WAV file into smaller chunks.

    Default:
        10-minute chunks

    Returns:
        List of chunk file paths.
    """

    wav_file = Path(wav_path).expanduser().resolve()

    if not wav_file.exists():
        raise FileNotFoundError(
            f"\nWAV file not found:\n{wav_file}"
        )

    if chunk_minutes <= 0:
        raise ValueError(
            "chunk_minutes must be greater than 0."
        )

    print(
        f"Loading audio for chunking:\n"
        f"{wav_file}"
    )

    try:
        audio = AudioSegment.from_wav(
            str(wav_file)
        )

    except Exception as exc:
        raise RuntimeError(
            f"\nCould not read WAV file.\n\n"
            f"File: {wav_file}\n"
            f"Error: {exc}"
        ) from exc

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for index, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = wav_file.with_name(
            f"{wav_file.stem}_chunk_{index}.wav"
        )

        chunk.export(
            str(chunk_path),
            format="wav"
        )

        chunks.append(str(chunk_path))

    print(
        f"✓ Created {len(chunks)} audio chunk(s)."
    )

    return chunks


# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================

def process_input(source: str) -> list[str]:
    """
    Process either:

        1. YouTube URL
        2. Local audio/video file

    Pipeline:

        YouTube URL
            ↓
        yt-dlp
            ↓
        WAV
            ↓
        16 kHz Mono
            ↓
        Chunking
            ↓
        Transcription-ready audio

    Local file

        Local audio/video
            ↓
        FFmpeg/Pydub
            ↓
        WAV
            ↓
        16 kHz Mono
            ↓
        Chunking
            ↓
        Transcription-ready audio

    Returns:
        List of audio chunk paths.
    """

    if not source:
        raise ValueError(
            "Input source cannot be empty."
        )

    source = source.strip()

    # --------------------------------------------------------
    # YouTube URL
    # --------------------------------------------------------

    if (
        source.startswith("http://")
        or source.startswith("https://")
    ):

        print(
            "\n========================================"
        )
        print("YouTube input detected")
        print(
            "========================================"
        )

        wav_path = download_youtube_audio(
            source
        )

        # Normalize downloaded WAV
        wav_path = normalize_wav(
            wav_path
        )

    # --------------------------------------------------------
    # Local file
    # --------------------------------------------------------

    else:

        print(
            "\n========================================"
        )
        print("Local file detected")
        print(
            "========================================"
        )

        wav_path = convert_to_wav(
            source
        )

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print("Chunking audio")
    print(
        "========================================"
    )

    chunks = chunk_audio(
        wav_path,
        chunk_minutes=10
    )

    print(
        f"\n✓ Audio ready — "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks