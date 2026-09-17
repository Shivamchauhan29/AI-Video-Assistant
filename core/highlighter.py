import os
import json

from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

MIN_WINDOW_SECONDS = 30.0
MAX_WINDOW_SECONDS = 60.0
OVERLAP_RATIO = 0.5
MAX_CLIPS = 3


def get_llm():
    # Same pattern as summarizer.py/extractor.py: Groq primary, Mistral fallback.
    groq_llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.2)
    mistral_llm = ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.2)
    return groq_llm.with_fallbacks([mistral_llm])


def generate_candidate_windows(
    segments: list,
    min_duration: float = MIN_WINDOW_SECONDS,
    max_duration: float = MAX_WINDOW_SECONDS,
    overlap_ratio: float = OVERLAP_RATIO,
) -> list:
    """
    Group consecutive segments into ~30-60s windows, never splitting a
    segment across a window boundary, sliding with ~50% overlap so no
    viable clip start is missed.

    Returns an ordered list of {"start": float, "end": float, "text": str}.
    """

    n = len(segments)
    if n == 0:
        return []

    windows = []
    start_idx = 0

    while start_idx < n:
        window_start = segments[start_idx]["start"]
        end_idx = start_idx

        # Grow the window until it reaches min_duration, but never add a
        # segment that would push it past max_duration.
        while end_idx + 1 < n:
            current_duration = segments[end_idx]["end"] - window_start
            if current_duration >= min_duration:
                break
            next_duration = segments[end_idx + 1]["end"] - window_start
            if next_duration > max_duration:
                break
            end_idx += 1

        window_segments = segments[start_idx:end_idx + 1]
        window_end = window_segments[-1]["end"]
        text = " ".join(s["text"] for s in window_segments).strip()

        windows.append({"start": window_start, "end": window_end, "text": text})

        if end_idx >= n - 1:
            break

        # Slide the next window's start to ~50% into this window's span.
        window_duration = window_end - window_start
        advance_to = window_start + window_duration * (1 - overlap_ratio)

        next_start_idx = start_idx + 1
        while next_start_idx <= end_idx and segments[next_start_idx]["start"] < advance_to:
            next_start_idx += 1
        if next_start_idx <= start_idx:
            next_start_idx = start_idx + 1

        start_idx = next_start_idx

    return windows


def build_scoring_chain():
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert short-form video editor. Score the following "
                "transcript excerpt for how well it would work as a standalone "
                "social-media highlight/reel, from 0 to 100. Consider:\n"
                "- Standalone coherence: does it make sense without outside context?\n"
                "- Hook strength: does the first ~5 seconds grab attention?\n"
                "- Information density / quotability: is it punchy and memorable?\n\n"
                "Respond with ONLY a JSON object of the exact form:\n"
                '{{"score": <integer 0-100>, "reason": "<one-line reason>"}}',
            ),
            ("human", "{text}"),
        ]
    )

    return prompt | llm | JsonOutputParser()


def score_window(chain, text: str) -> dict:
    """Score one window's text, returning {"score": int, "reason": str}."""
    try:
        result = chain.invoke({"text": text})
        score = int(result["score"])
        reason = str(result["reason"]).strip()
    except Exception as e:
        score, reason = 0, f"scoring failed: {e}"

    score = max(0, min(100, score))
    return {"score": score, "reason": reason}


def score_candidate_windows(windows: list) -> list:
    """Sequentially score every window. Returns windows with score/reason added."""
    if not windows:
        return []

    chain = build_scoring_chain()
    scored = []
    for window in windows:
        result = score_window(chain, window["text"])
        scored.append({**window, **result})

    return scored


def select_highlight_clips(scored_windows: list, max_clips: int = MAX_CLIPS) -> list:
    """
    Sort by score descending, greedily pick non-overlapping windows
    (discarding any candidate overlapping an already-picked one) until
    max_clips are selected or candidates run out.
    """
    ranked = sorted(scored_windows, key=lambda w: w["score"], reverse=True)

    selected = []
    for window in ranked:
        overlaps = any(
            window["start"] < picked["end"] and window["end"] > picked["start"]
            for picked in selected
        )
        if overlaps:
            continue

        selected.append(window)
        if len(selected) >= max_clips:
            break

    selected.sort(key=lambda w: w["start"])
    return selected


def generate_highlights(segments: list) -> list:
    """
    English/Whisper path only. Given transcript_segments, returns up to
    MAX_CLIPS non-overlapping highlight candidates:
        [{"start", "end", "score", "reason", "text"}, ...]
    """
    if not segments:
        return []

    windows = generate_candidate_windows(segments)
    scored = score_candidate_windows(windows)
    return select_highlight_clips(scored)
