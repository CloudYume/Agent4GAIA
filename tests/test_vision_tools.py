import base64
import io

import fitz
import pytest
from PIL import Image

from gaia_agent.tools.vision import VisualCatalog


def _data_url(color="red"):
    output = io.BytesIO()
    Image.new("RGB", (100, 100), color).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def test_inspect_pdf_page_region_and_reuse_cached_observation(tmp_path):
    path = tmp_path / "document.pdf"
    document = fitz.open()
    for number in range(1, 4):
        page = document.new_page(width=200, height=200)
        page.insert_text((20, 40), f"PAGE {number}")
    document.save(path)
    document.close()
    catalog = VisualCatalog([], path)
    assert catalog.manifest()["additional_locators"] == ["pdf:page:N (N=1..3)"]
    calls = []

    def observe(image_url, locator, question):
        calls.append((image_url, locator, question))
        return "PAGE 3 is visible.", {"prompt_tokens": 10}

    first, usage = catalog.inspect("pdf:page:3", "What is written?", [0, 0, 0.5, 0.5], observe, cache_dir=tmp_path / "cache")
    second, cached_usage = catalog.inspect("pdf:page:3", "What is written?", [0, 0, 0.5, 0.5], observe, cache_dir=tmp_path / "cache")
    assert first["locator"] == "pdf:page:3"
    assert first["region"] == [0.0, 0.0, 0.5, 0.5]
    assert first["observation"] == "PAGE 3 is visible."
    assert len(first["sha256"]) == 64
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert usage == {"prompt_tokens": 10} and cached_usage is None
    assert len(calls) == 1
    assert calls[0][0].startswith("data:image/jpeg;base64,")


def test_prepared_frame_preserves_index_and_timestamp(tmp_path):
    catalog = VisualCatalog([
        {"type": "input_text", "text": "Video frame near 42.0 seconds"},
        {"type": "input_image", "image_url": _data_url()},
    ])
    result, _ = catalog.inspect("prepared:0", "What color?", None, lambda *args: ("Red", None), cache_dir=tmp_path / "cache")
    assert result["image_index"] == 0
    assert result["timestamp"] == 42.0
    assert result["uncertain"] is False


def test_invalid_locator_or_region_never_calls_observer(tmp_path):
    catalog = VisualCatalog([{"type": "input_image", "image_url": _data_url()}])

    def forbidden(*args):
        pytest.fail("Observer must not run")

    with pytest.raises(ValueError, match="unavailable"):
        catalog.inspect("prepared:2", "Question?", None, forbidden, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="region"):
        catalog.inspect("prepared:0", "Question?", [0.5, 0, 0.4, 1], forbidden, cache_dir=tmp_path / "cache")
