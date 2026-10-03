"""视频下载和全片均匀抽帧，产物仅保存在本地私有目录。"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

VIDEO_URL = re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch\?[^\s]+|youtu\.be/[^\s]+)")
MAX_VIDEO_BYTES = 128 * 1024 * 1024
MAX_VIDEO_SECONDS = 60 * 60
DEFAULT_VIDEO_FRAMES = 24
_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install the video extra: pip install -e '.[video]'") from exc


def _browser_cookies_option(value: str) -> tuple[str, str | None, None, None] | None:
    if not value:
        return None
    browser, separator, profile = value.partition(":")
    if browser.lower() != "edge" or (separator and not re.fullmatch(r"[A-Za-z0-9_. -]{1,80}", profile)):
        raise ValueError("youtube_cookies_from_browser must be edge or edge:<profile>")
    return ("edge", profile if separator else None, None, None)


def download_video(question: str, directory: Path, *, cookies_from_browser: str = "") -> Path | None:
    match = VIDEO_URL.search(question)
    if not match:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    cached = [path for path in directory.glob("video.*") if path.stem == "video" and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov", ".avi"}]
    if len(cached) > 1:
        raise RuntimeError(f"Multiple cached videos found in {directory}; choose the correct source")
    if cached:
        size = cached[0].stat().st_size
        if not 0 < size <= MAX_VIDEO_BYTES:
            raise ValueError(f"Cached video has invalid size: {cached[0].name}")
        return cached[0]
    cookie_option = _browser_cookies_option(cookies_from_browser)
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("Install the video extra: pip install -e '.[video]'") from exc
    options = {
            "format": "best[height<=720]/best",
            "outtmpl": str(directory / "video.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "max_filesize": MAX_VIDEO_BYTES,
        }
    if cookie_option:
        options["cookiesfrombrowser"] = cookie_option
    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(match.group(0).rstrip(".,)"), download=True)
            path = Path(downloader.prepare_filename(info))
            if not path.is_file():
                raise RuntimeError("Video downloader did not produce the expected file")
            if path.stat().st_size > MAX_VIDEO_BYTES:
                raise ValueError(f"Video exceeds {MAX_VIDEO_BYTES // (1024 * 1024)} MiB limit")
            return path
    except ValueError:
        raise
    except Exception as exc:
        method = " with Edge browser cookies" if cookie_option else ""
        raw = str(exc).lower()
        reason = (
            "browser cookie decryption unavailable (DPAPI)" if "dpapi" in raw else
            "Edge cookie database unavailable" if "could not copy chrome cookie database" in raw else
            "YouTube authentication required" if "sign in to confirm" in raw else
            f"{type(exc).__name__}"
        )
        raise RuntimeError(
            f"Video download failed{method}: {reason}; "
            "check local media cache, browser login, and yt-dlp availability"
        ) from None


def _video_duration(path: Path, ffmpeg: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        try:
            duration = float(result.stdout.strip())
            if duration > 0:
                return duration
        except ValueError:
            pass
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    match = _DURATION.search(result.stderr)
    if not match:
        raise RuntimeError(f"Cannot determine duration of {path.name}; full-video sampling is unavailable")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_video(path: Path, directory: Path, max_frames: int = DEFAULT_VIDEO_FRAMES) -> tuple[list[tuple[Path, float]], Path | None]:
    if max_frames < 0:
        raise ValueError("max_frames must be nonnegative")
    if path.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError(f"Video exceeds {MAX_VIDEO_BYTES // (1024 * 1024)} MiB limit")
    directory.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()
    duration = _video_duration(path, ffmpeg)
    if duration <= 0 or duration > MAX_VIDEO_SECONDS:
        raise ValueError(f"Video duration must be within 0-{MAX_VIDEO_SECONDS} seconds")
    for old_frame in directory.glob("frame-*.jpg"):
        old_frame.unlink()
    fps = min(2.0, max_frames / duration) if max_frames else 0.0
    if max_frames:
        pattern = directory / "frame-%03d.jpg"
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-vf", f"fps={fps:.8f},scale=960:-2", "-q:v", "4", "-frames:v", str(max_frames), str(pattern)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=180,
        )
    audio = directory / "audio.mp3"
    audio.unlink(missing_ok=True)
    audio_result = subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(audio)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=180,
    )
    frames = [(frame, index / fps) for index, frame in enumerate(sorted(directory.glob("frame-*.jpg")))] if max_frames else []
    if max_frames and not frames:
        raise RuntimeError(f"No frames could be decoded from {path.name}")
    return frames, audio if audio_result.returncode == 0 else None


def render_video_frame(path: Path, seconds: float) -> bytes:
    """Decode one requested frame without creating an intermediate file."""
    if not path.is_file() or path.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError("Video source is missing or exceeds the media size limit")
    ffmpeg = _ffmpeg()
    duration = _video_duration(path, ffmpeg)
    if duration <= 0 or duration > MAX_VIDEO_SECONDS or not 0 <= seconds < duration:
        raise ValueError(f"Video timestamp must be within 0-{duration:.2f} seconds")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{seconds:.3f}",
         "-i", str(path), "-frames:v", "1", "-vf", "scale=1600:-2",
         "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
        capture_output=True, check=False, timeout=60,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"Could not decode video frame at {seconds:.3f} seconds")
    return result.stdout
