"""Locate visual evidence and cache question-specific observations."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

from ..media import render_video_frame

MAX_REGION_PIXELS = 36_000_000
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_QUESTION_CHARS = 500
CACHE_VERSION = "vision-observation-v1"
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_TYPES = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
_UNCERTAIN = re.compile(r"\b(?:unclear|uncertain|unknown|cannot|can't|obscured|illegible|possibly|might)\b", re.I)


def _data_bytes(url: str) -> bytes:
    if not url.startswith("data:image/") or ";base64," not in url:
        raise ValueError("Prepared image must be a base64 data URL")
    encoded = url.split(",", 1)[1]
    if len(encoded) > MAX_IMAGE_BYTES * 2:
        raise ValueError("Prepared image exceeds the size limit")
    try:
        result = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Prepared image has invalid base64") from exc
    if len(result) > MAX_IMAGE_BYTES:
        raise ValueError("Prepared image exceeds the size limit")
    return result


def _region(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4 or any(type(item) not in {float, int} for item in value):
        raise ValueError("region must be [left, top, right, bottom] in normalized coordinates")
    left, top, right, bottom = map(float, value)
    if not all(math.isfinite(item) for item in (left, top, right, bottom)) or not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("region must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1")
    if (right - left) * (bottom - top) < 0.0001:
        raise ValueError("region is too small to inspect")
    return left, top, right, bottom


def _normalized_jpeg(source: bytes, region: tuple[float, float, float, float] | None) -> bytes:
    if len(source) > MAX_IMAGE_BYTES:
        raise ValueError("Visual source exceeds the size limit")
    with Image.open(io.BytesIO(source)) as opened:
        if opened.width * opened.height > MAX_REGION_PIXELS:
            raise ValueError("Visual source exceeds the pixel limit")
        image = ImageOps.exif_transpose(opened)
        if region:
            left, top, right, bottom = region
            box = (
                int(left * image.width), int(top * image.height),
                max(int(left * image.width) + 1, int(right * image.width)),
                max(int(top * image.height) + 1, int(bottom * image.height)),
            )
            image = image.crop(box)
        image.thumbnail((2000, 2000))
        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=90)
        return output.getvalue()


class VisualCatalog:
    def __init__(self, content: list[dict[str, Any]], attachment_path: str | Path | None = None):
        self.prepared: list[tuple[str, bytes]] = []
        label = "Attached image"
        for item in content:
            if item.get("type") == "input_text":
                candidate = str(item.get("text") or "").strip()
                label = candidate[:160] if candidate else label
            elif item.get("type") == "input_image":
                self.prepared.append((label, _data_bytes(str(item.get("image_url") or ""))))
        self.attachment_path = Path(attachment_path) if attachment_path else None

    def manifest(self) -> dict[str, Any]:
        assets = [
            {"locator": f"prepared:{index}", "label": label,
             "sha256": hashlib.sha256(image).hexdigest()}
            for index, (label, image) in enumerate(self.prepared)
        ]
        other = []
        if self.attachment_path and self.attachment_path.is_file():
            suffix = self.attachment_path.suffix.lower()
            if suffix in IMAGE_TYPES:
                other.append("attachment (original image)")
            elif suffix == ".pdf":
                import fitz

                with fitz.open(self.attachment_path) as document:
                    other.append(f"pdf:page:N (N=1..{document.page_count})")
            elif suffix in VIDEO_TYPES:
                other.append("video:second:S (S=timestamp in seconds)")
        return {"prepared": assets, "additional_locators": other}

    def _source_bytes(
        self, locator: str, region: tuple[float, float, float, float] | None,
    ) -> tuple[bytes, bool]:
        prepared = re.fullmatch(r"prepared:(\d+)", locator)
        if prepared:
            index = int(prepared.group(1))
            if index >= len(self.prepared):
                raise ValueError("Prepared image index is unavailable")
            return self.prepared[index][1], False
        if not self.attachment_path or not self.attachment_path.is_file():
            raise ValueError("This visual locator requires a local attachment")
        path = self.attachment_path
        if locator == "attachment" and path.suffix.lower() in IMAGE_TYPES:
            if path.stat().st_size > MAX_IMAGE_BYTES:
                raise ValueError("Original image exceeds the size limit")
            return path.read_bytes(), False
        page = re.fullmatch(r"pdf:page:(\d+)", locator)
        if page and path.suffix.lower() == ".pdf":
            import fitz

            number = int(page.group(1))
            with fitz.open(path) as document:
                if not 1 <= number <= document.page_count:
                    raise ValueError("PDF page is outside the document")
                selected = document[number - 1]
                clip = selected.rect
                if region:
                    left, top, right, bottom = region
                    clip = fitz.Rect(
                        clip.x0 + left * clip.width, clip.y0 + top * clip.height,
                        clip.x0 + right * clip.width, clip.y0 + bottom * clip.height,
                    )
                scale = min(3.0 if region else 2.0, math.sqrt(MAX_REGION_PIXELS / max(clip.width * clip.height, 1)))
                return selected.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False).tobytes("jpeg", jpg_quality=90), bool(region)
        second = re.fullmatch(r"video:second:(\d+(?:\.\d+)?)", locator)
        if second and path.suffix.lower() in VIDEO_TYPES:
            return render_video_frame(path, float(second.group(1))), False
        raise ValueError("Visual locator is unavailable for this attachment")

    def inspect(
        self,
        locator: str,
        question: str,
        region: list[float] | None,
        observe: Callable[[str, str, str], tuple[str, Any]],
        *,
        cache_dir: Path = Path(".runs") / "vision-observations",
    ) -> tuple[dict[str, Any], Any | None]:
        if not isinstance(locator, str) or len(locator) > 100:
            raise ValueError("Visual locator is invalid")
        if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"Visual question must contain 1 to {MAX_QUESTION_CHARS} characters")
        normalized_region = _region(region)
        source, already_cropped = self._source_bytes(locator, normalized_region)
        bitmap = _normalized_jpeg(source, None if already_cropped else normalized_region)
        sha256 = hashlib.sha256(bitmap).hexdigest()
        normalized_question = " ".join(question.split())
        cache_key = hashlib.sha256(
            json.dumps([CACHE_VERSION, sha256, normalized_question], ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cache_file = cache_dir / f"{cache_key}.json"
        prepared_match = re.fullmatch(r"prepared:(\d+)", locator)
        video_match = re.fullmatch(r"video:second:(\d+(?:\.\d+)?)", locator)
        frame_label = self.prepared[int(prepared_match.group(1))][0] if prepared_match else ""
        label_time = re.search(r"Video frame near (\d+(?:\.\d+)?) seconds", frame_label)
        page_match = re.fullmatch(r"pdf:page:(\d+)", locator) or re.search(r"PDF page (\d+)", frame_label)
        base = {
            "asset_id": f"visual:{sha256[:20]}", "sha256": sha256,
            "locator": locator, "region": list(normalized_region) if normalized_region else None,
            "question": normalized_question,
            "image_index": int(prepared_match.group(1)) if prepared_match else None,
            "timestamp": float(video_match.group(1)) if video_match else float(label_time.group(1)) if label_time else None,
            "page": int(page_match.group(1)) if page_match else None,
            "label": frame_label or locator,
        }
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if (cached.get("sha256") == sha256 and cached.get("question") == normalized_question
                    and isinstance(cached.get("observation"), str) and cached["observation"]):
                return {**base, "observation": cached["observation"], "description": cached["observation"],
                        "uncertain": bool(cached.get("uncertain")), "cache_hit": True}, None
        except (FileNotFoundError, ValueError, OSError, AttributeError):
            pass
        image_url = "data:image/jpeg;base64," + base64.b64encode(bitmap).decode("ascii")
        observation, usage = observe(image_url, locator, normalized_question)
        if not isinstance(observation, str) or not observation.strip():
            raise RuntimeError("Vision model returned no usable image description")
        observation = observation.strip()[:6_000]
        result = {**base, "observation": observation, "description": observation,
                  "uncertain": bool(_UNCERTAIN.search(observation)), "cache_hit": False}
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_dir, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(result, handle, ensure_ascii=False)
            os.replace(temporary, cache_file)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
        return result, usage
