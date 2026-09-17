import os

from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from core.pipeline_logger import log_if

MIN_WINDOW_SECONDS = 30.0
MAX_WINDOW_SECONDS = 60.0
OVERLAP_RATIO = 0.5
MAX_CLIPS = 3

# Guardrail from the requirements: batch at most this many windows into a
# single prompt. Videos with more candidates get multiple sequential
# batched calls instead of one call per window.
BATCH_SIZE = 15

TITLE_HARD_CAP = 100
TAGS_MAX_COUNT = 15
TAGS_MAX_CHARS = 500


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
    logger=None,
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

    log_if(logger, "Highlight Detection", f"Generated {len(windows)} candidate window(s)")
    return windows


def build_batch_chain():
    """
    One chain that scores AND generates Shorts metadata for a whole batch
    of candidate windows in a single call, instead of a separate call per
    window (scoring) plus a separate call per selected clip (metadata).
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert short-form video editor and YouTube Shorts "
                "strategist. You will be given a numbered list of transcript "
                "excerpts (candidate clips) from a longer video. Evaluate EACH "
                "one independently and return both a reel-worthiness score and "
                "ready-to-publish Shorts metadata for it.\n\n"
                "Score 0-100 based on:\n"
                "- Standalone coherence: does it make sense without outside context?\n"
                "- Hook strength: does the first ~5 seconds grab attention?\n"
                "- Information density / quotability: is it punchy and memorable?\n\n"
                "Also generate for each window:\n"
                "- title: a punchy, clickable title. Ideally under 60 characters, "
                f"never over {TITLE_HARD_CAP}.\n"
                "- description: a short YouTube Shorts description ending with "
                "#Shorts plus 2-4 other relevant hashtags.\n"
                f"- tags: {TAGS_MAX_COUNT - 7}-{TAGS_MAX_COUNT} short keyword tags "
                f"as a JSON array of strings, combined length under {TAGS_MAX_CHARS} characters.\n"
                "- reason: one line explaining the score.\n\n"
                "Respond with ONLY a JSON array, one object per window, preserving "
                "window_index from the input, in exactly this form:\n"
                '[{{"window_index": <int>, "score": <int 0-100>, "reason": "<string>", '
                '"title": "<string>", "description": "<string>", "tags": ["<string>", ...]}}, ...]',
            ),
            ("human", "{windows}"),
        ]
    )

    return prompt | llm | JsonOutputParser()


def _format_windows_prompt(windows: list) -> str:
    lines = [
        f"Window {i} [{w['start']:.1f}s - {w['end']:.1f}s]: {w['text']}"
        for i, w in enumerate(windows)
    ]
    return "\n\n".join(lines)


def _default_fields(reason: str) -> dict:
    return {"score": 0, "reason": reason, "title": "", "description": "", "tags": []}


def _normalize_fields(item: dict) -> dict:
    try:
        score = max(0, min(100, int(item.get("score", 0))))
    except (TypeError, ValueError):
        score = 0

    reason = str(item.get("reason", "")).strip()
    title = str(item.get("title", "")).strip()[:TITLE_HARD_CAP]
    description = str(item.get("description", "")).strip()

    tags = item.get("tags", [])
    if not isinstance(tags, list):
        tags = [tags]
    tags = [str(t).strip() for t in tags if str(t).strip()][:TAGS_MAX_COUNT]
    while tags and sum(len(t) for t in tags) > TAGS_MAX_CHARS:
        tags.pop()

    return {"score": score, "reason": reason, "title": title, "description": description, "tags": tags}


def _score_and_describe_batch(chain, batch: list) -> dict:
    """Runs one batched LLM call for `batch`. Returns {local_index: fields}."""
    try:
        result = chain.invoke({"windows": _format_windows_prompt(batch)})
    except Exception as e:
        return {i: _default_fields(f"batch scoring failed: {e}") for i in range(len(batch))}

    by_index = {}
    if isinstance(result, list):
        for item in result:
            try:
                idx = int(item["window_index"])
            except (KeyError, TypeError, ValueError):
                continue
            by_index[idx] = _normalize_fields(item)

    for i in range(len(batch)):
        if i not in by_index:
            by_index[i] = _default_fields("missing from batch response")

    return by_index


def score_candidate_windows(windows: list, logger=None) -> list:
    """
    Score every window AND generate its Shorts metadata (title, description,
    tags) in as few LLM calls as possible: one call per BATCH_SIZE windows,
    not one call per window.

    Returns windows with score/reason/title/description/tags added.
    """
    if not windows:
        return []

    chain = build_batch_chain()
    scored = list(windows)
    total = len(windows)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = windows[batch_start: batch_start + BATCH_SIZE]
        fields_by_local_index = _score_and_describe_batch(chain, batch)

        for local_i in sorted(fields_by_local_index.keys()):
            fields = fields_by_local_index[local_i]
            global_i = batch_start + local_i
            if 0 <= global_i < len(scored):
                scored[global_i] = {**scored[global_i], **fields}
                log_if(logger, "Highlight Detection", f"Scored window {global_i + 1}/{total} (score={fields['score']})")

    return scored


def select_highlight_clips(scored_windows: list, max_clips: int = MAX_CLIPS, logger=None) -> list:
    """
    Sort by score descending, greedily pick non-overlapping windows
    (discarding any candidate overlapping an already-picked one) until
    max_clips are selected or candidates run out. No LLM calls here — score
    is already populated by score_candidate_windows().
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
    log_if(logger, "Highlight Detection", f"Selected {len(selected)} final clip(s)")
    return selected


def generate_highlights(segments: list, logger=None) -> list:
    """
    English/Whisper path only. Given transcript_segments, returns up to
    MAX_CLIPS non-overlapping highlight candidates, each already carrying
    its Shorts metadata from the single batched scoring+metadata call:
        [{"start", "end", "score", "reason", "title", "description", "tags", "text"}, ...]
    """
    if not segments:
        return []

    windows = generate_candidate_windows(segments, logger=logger)
    scored = score_candidate_windows(windows, logger=logger)
    return select_highlight_clips(scored, logger=logger)
