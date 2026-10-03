import subprocess
import sys
from types import SimpleNamespace

import pytest

from gaia_agent import media


def test_extract_video_samples_full_duration_and_clears_stale_outputs(tmp_path):
    ffmpeg = media._ffmpeg()
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc=size=32x32:rate=1:duration=30", "-c:v", "mpeg4", str(video)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    output = tmp_path / "frames"
    output.mkdir()
    stale = output / "frame-999.jpg"
    stale.write_bytes(b"old")
    stale_audio = output / "audio.mp3"
    stale_audio.write_bytes(b"old")

    frames, audio = media.extract_video(video, output, max_frames=4)

    assert len(frames) == 4
    assert frames[-1][1] >= 22
    assert all(frame.is_file() for frame, _ in frames)
    assert not stale.exists()
    assert audio is None
    assert not stale_audio.exists()

    frames, audio = media.extract_video(video, output)
    assert len(frames) == media.DEFAULT_VIDEO_FRAMES == 24
    assert frames[-1][1] >= 28
    assert audio is None

    frames, audio = media.extract_video(video, output, max_frames=0)
    assert frames == []
    assert audio is None
    assert not list(output.glob("frame-*.jpg"))


def test_extract_video_rejects_unknown_duration(tmp_path, monkeypatch):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    monkeypatch.setattr(media, "_video_duration", lambda path, ffmpeg: 0)
    with pytest.raises(ValueError, match="Video duration"):
        media.extract_video(path, tmp_path / "frames")


def test_download_video_reuses_completed_local_media(tmp_path):
    cached = tmp_path / "video.mp4"
    cached.write_bytes(b"existing video evidence")

    result = media.download_video("Watch https://www.youtube.com/watch?v=example", tmp_path)

    assert result == cached
    assert cached.read_bytes() == b"existing video evidence"


def test_download_video_uses_edge_profile_only_after_cache_miss(tmp_path, monkeypatch):
    options_seen = []

    class FakeDownloader:
        def __init__(self, options):
            options_seen.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            assert download is True
            (tmp_path / "video.mp4").write_bytes(b"video")
            return {"id": "sample"}

        def prepare_filename(self, info):
            return str(tmp_path / "video.mp4")

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeDownloader))
    result = media.download_video("https://www.youtube.com/watch?v=sample", tmp_path, cookies_from_browser="edge:Default")
    assert result == tmp_path / "video.mp4"
    assert options_seen[0]["cookiesfrombrowser"] == ("edge", "Default", None, None)
    assert media.download_video("https://www.youtube.com/watch?v=sample", tmp_path, cookies_from_browser="invalid") == result
    assert len(options_seen) == 1


def test_download_video_redacts_downloader_error(tmp_path, monkeypatch):
    class FakeDownloader:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            raise RuntimeError("Failed to decrypt with DPAPI: private cookie data")

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeDownloader))
    with pytest.raises(RuntimeError, match="Video download failed with Edge browser cookies") as failure:
        media.download_video("https://www.youtube.com/watch?v=sample", tmp_path, cookies_from_browser="edge:Default")
    assert "DPAPI" in str(failure.value)
    assert "private cookie data" not in str(failure.value)


def test_render_video_frame_at_requested_second(tmp_path):
    ffmpeg = media._ffmpeg()
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=1:duration=3", "-c:v", "mpeg4", str(video)],
        check=True, capture_output=True, timeout=30,
    )
    frame = media.render_video_frame(video, 2.0)
    assert frame.startswith(b"\xff\xd8")
    assert len(frame) > 100
