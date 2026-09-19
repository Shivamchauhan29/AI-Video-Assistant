from faster_whisper import WhisperModel, BatchedInferencePipeline
import os
import time
import requests
from pydub import AudioSegment

from core.pipeline_logger import log_if, format_duration

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# int8 measured on par with (or faster than) float32 in local benchmarking,
# with essentially verbatim-matching transcript text vs the openai-whisper
# baseline, and a much smaller memory footprint — worth defaulting to.
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Whisper's default beam_size=5 is overkill for decoding speed; 1-2 measured
# no accuracy loss on test videos while cutting decode time noticeably.
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "2"))


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():
    """
    Lazily load the faster-whisper (CTranslate2) model, wrapped in a
    BatchedInferencePipeline for batched VAD-segment inference — loaded
    once and reused across every chunk/video in the process, same as the
    prior openai-whisper global-cache pattern.
    """

    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} (faster-whisper, {WHISPER_COMPUTE_TYPE}) ...")
        base_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
        _model = BatchedInferencePipeline(model=base_model)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> list:
    """
    Transcribe one chunk with faster-whisper (VAD-filtered, batched
    inference), preserving segment (and word) timing.

    Returns an ordered list of segments, timestamps relative to the start
    of this chunk (the caller is responsible for offsetting them into the
    full recording's timeline):
        [{"start": float, "end": float, "text": str, "words": [...]}, ...]
    """

    pipeline = load_model()

    # vad_filter skips silent/non-speech stretches instead of transcribing
    # them; word_timestamps=True adds DTW alignment cost per segment, but
    # it's what makes word-level timing available for clip generation.
    segments_iter, _info = pipeline.transcribe(
        chunk_path,
        task="transcribe",
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=True,
        word_timestamps=True,
    )

    segments = []
    for seg in segments_iter:
        entry = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        }
        if seg.words:
            entry["words"] = [
                {"start": float(w.start), "end": float(w.end), "word": w.word}
                for w in seg.words
            ]
        segments.append(entry)

    return segments


def _chunk_duration_seconds(chunk_path: str) -> float:
    return len(AudioSegment.from_wav(chunk_path)) / 1000.0


def segments_to_text(segments: list) -> str:
    """Flatten an ordered segment list back into the plain transcript string."""
    return " ".join(seg["text"] for seg in segments).strip()


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()

   



def transcribe_all(chunks: list, language: str = "english", logger=None) -> dict:
    """
    Transcribe every chunk in order and return:
        {"text": <flat transcript str>, "segments": <list|None>}

    "segments" is a chunk-offset-corrected, ordered list of
    {"start", "end", "text", "words"} covering the full recording — only
    populated for the Whisper (english) path. The Sarvam (hinglish) path
    has no timing information, so "segments" is None there, unchanged
    from its previous plain-text-only behavior.
    """

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")
    log_if(logger, "Transcription", f"Using {engine} for {len(chunks)} chunk(s)")

    if language.lower() == "hinglish":
        full_transcript = ""
        for i, chunk in enumerate(chunks):
            print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
            t0 = time.monotonic()
            full_transcript += transcribe_chunk_sarvam(chunk) + " "
            log_if(logger, "Transcription", f"Chunk {i + 1}/{len(chunks)} done ({format_duration(time.monotonic() - t0)})")

        print("Transcription complete.")
        log_if(logger, "Transcription", "Transcription complete")
        return {"text": full_transcript.strip(), "segments": None}

    # English / Whisper path — accumulate segments, offsetting each
    # chunk's timestamps by the cumulative duration of prior chunks so
    # they stay relative to the full recording, not the chunk.
    all_segments = []
    offset_seconds = 0.0

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        t0 = time.monotonic()

        chunk_segments = transcribe_chunk_whisper(chunk)
        for seg in chunk_segments:
            seg["start"] += offset_seconds
            seg["end"] += offset_seconds
            for w in seg.get("words", []):
                w["start"] += offset_seconds
                w["end"] += offset_seconds

        all_segments.extend(chunk_segments)
        offset_seconds += _chunk_duration_seconds(chunk)
        log_if(logger, "Transcription", f"Chunk {i + 1}/{len(chunks)} done ({format_duration(time.monotonic() - t0)})")

    print("Transcription complete.")
    log_if(logger, "Transcription", "Transcription complete")

    return {"text": segments_to_text(all_segments), "segments": all_segments}
