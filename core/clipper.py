import os
import shutil
import subprocess

from core.pipeline_logger import log_if

FFMPEG_PATH = shutil.which("ffmpeg")

# Seconds of margin for the fast (input-side) seek before doing an
# accurate (output-side) seek + re-encode. Keeps cutting fast even for
# clips near the end of a long source video, without sacrificing the
# frame-accurate start/end timestamps re-encoding is meant to give us.
SEEK_MARGIN_SECONDS = 5.0

# Shorts-style 9:16 output size.
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
BACKGROUND_BLUR_SIGMA = 20

# Background: scale to fill/cover 1080x1920 (cropping overflow), then blur.
# Foreground: scale to fit width 1080 (aspect preserved), centered on the
# blurred background — the blurred bars fill the top/bottom pillarbox.
_VERTICAL_FILTER = (
    "[0:v]split=2[bg][fg];"
    f"[bg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
    f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},gblur=sigma={BACKGROUND_BLUR_SIGMA}[bgblur];"
    f"[fg]scale={OUTPUT_WIDTH}:-2[fgscaled];"
    "[bgblur][fgscaled]overlay=(W-w)/2:(H-h)/2[vout]"
)


def _cut_one_clip(video_path: str, start: float, end: float, output_path: str) -> None:
    if not FFMPEG_PATH:
        raise RuntimeError("ffmpeg not found on PATH.")

    coarse_seek = max(0.0, start - SEEK_MARGIN_SECONDS)
    fine_seek = start - coarse_seek
    duration = end - start

    cmd = [
        FFMPEG_PATH, "-y",
        "-ss", f"{coarse_seek:.3f}",
        "-i", video_path,
        "-ss", f"{fine_seek:.3f}",
        "-t", f"{duration:.3f}",
        "-filter_complex", _VERTICAL_FILTER,
        "-map", "[vout]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(
            f"ffmpeg failed cutting [{start:.2f}s - {end:.2f}s]:\n"
            f"{result.stderr[-2000:]}"
        )


def cut_clips(video_path: str, clips: list, output_dir: str, logger=None) -> list:
    """
    Cut each clip's [start, end] range out of video_path into its own
    re-encoded MP4 (re-encoding, not stream-copy, so the cut lands
    exactly on the given timestamps rather than the nearest keyframe).

    Returns clips with a "path" key added, everything else (score,
    reason, text, start, end) preserved unchanged.
    """

    os.makedirs(output_dir, exist_ok=True)
    log_if(logger, "Clip Cutting", f"Cutting {len(clips)} clip(s)...")

    results = []
    for i, clip in enumerate(clips):
        output_path = os.path.join(output_dir, f"highlight_{i}.mp4")
        _cut_one_clip(video_path, clip["start"], clip["end"], output_path)
        results.append({**clip, "path": output_path})
        log_if(logger, "Clip Cutting", f"{os.path.basename(output_path)} done")

    return results
