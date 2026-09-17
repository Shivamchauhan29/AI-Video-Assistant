class PipelineLogger:
    """
    Shared, optional callback-based logger threaded through pipeline stage
    functions so the UI and CLI can surface live progress instead of a
    static wait.

    log() always prints, so main.py's CLI gets the same structured
    "[Stage] message" lines for free just by constructing a logger with no
    callback. The optional on_log callback lets a caller (the Streamlit UI)
    also mirror each line somewhere else in real time as it's emitted.

    Passing logger=None into any pipeline function is always a safe no-op —
    it emits no logs and changes no behavior, so existing callers that
    don't opt in (e.g. test.py) are unaffected.
    """

    def __init__(self, on_log=None):
        self._on_log = on_log

    def log(self, stage: str, message: str) -> None:
        line = f"[{stage}] {message}"
        print(line)
        if self._on_log:
            self._on_log(line)


def log_if(logger, stage: str, message: str) -> None:
    """Convenience for call sites: log only if a logger was provided."""
    if logger:
        logger.log(stage, message)


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
