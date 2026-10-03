from pathlib import Path

import pytest

from gaia_agent.sources import Task, _repo_file, course_tasks, download_course_attachments, download_gaia_attachment, gaia_tasks


def test_repo_file_requires_expected_split():
    assert _repo_file({"file_name": "figure.png"}, "test") == "2023/test/figure.png"
    assert _repo_file({"file_path": "2023/test/figure.png"}, "test") == "2023/test/figure.png"
    with pytest.raises(ValueError):
        _repo_file({"file_path": "../secret"}, "test")


def test_gaia_lazy_download_preserves_nested_repo_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "gaia_agent.sources.load_gaia_rows",
        lambda split, token: [{
            "task_id": "nested", "Question": "What?", "Level": 1,
            "file_name": "figure.pdf", "file_path": "2023/test/sub/figure.pdf",
        }],
    )
    requested = []

    def download(**kwargs):
        requested.append(kwargs["filename"])
        return str(tmp_path / "figure.pdf")

    monkeypatch.setattr("gaia_agent.sources.hf_hub_download", download)
    task = gaia_tasks("test", download=False)[0]
    assert task.repo_file == "2023/test/sub/figure.pdf"
    assert task.file_name == "figure.pdf"
    download_gaia_attachment(task, "test", None)
    assert requested == ["2023/test/sub/figure.pdf"]
    assert task.attachment_path == str(tmp_path / "figure.pdf")


def test_course_attachment_404_falls_back_to_gated_dataset(monkeypatch, tmp_path):
    class Response:
        def __init__(self, code, body=None):
            self.status_code = code
            self.body = body

        def raise_for_status(self):
            if self.status_code >= 400:
                import requests

                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return self.body

    def fake_get(url, timeout):
        if url.endswith("/questions"):
            return Response(200, [{"task_id": "abc", "question": "What?", "Level": "1", "file_name": "x.png"}])
        return Response(404)

    monkeypatch.setattr("gaia_agent.sources.requests.get", fake_get)
    monkeypatch.setattr(
        "gaia_agent.sources.load_gaia_rows",
        lambda split, token: [{"task_id": "abc", "file_name": "x.png"}],
    )
    monkeypatch.setattr("gaia_agent.sources.hf_hub_download", lambda **kwargs: str(tmp_path / "x.png"))
    tasks = course_tasks(token="test-token")
    assert tasks[0].attachment_path == str(tmp_path / "x.png")
    assert not tasks[0].attachment_error


def test_course_attachment_fallback_uses_cached_login_without_explicit_token(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            raise requests.HTTPError("404")

        def json(self):
            return [{"task_id": "abc", "question": "What?", "Level": "1", "file_name": "x.png"}]

    import requests

    monkeypatch.setattr("gaia_agent.sources.requests.get", lambda url, timeout: Response() if "/files/" in url else type("Questions", (), {"raise_for_status": lambda self: None, "json": Response.json})())
    monkeypatch.setattr("gaia_agent.sources.load_gaia_rows", lambda split, token: [{"task_id": "abc", "file_name": "x.png"}])
    captured = []

    def download(**kwargs):
        captured.append(kwargs)
        return str(tmp_path / "x.png")

    monkeypatch.setattr("gaia_agent.sources.hf_hub_download", download)
    tasks = course_tasks(token=None)
    assert tasks[0].attachment_path == str(tmp_path / "x.png")
    assert captured[0]["token"] is None


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        (b"<!doctype html><html>login</html>", "text/html; charset=utf-8"),
        (b"not a PNG file", "application/octet-stream"),
    ],
)
def test_course_200_with_wrong_attachment_content_uses_fallback(monkeypatch, tmp_path, content, content_type):
    monkeypatch.chdir(tmp_path)

    class Response:
        headers = {"Content-Type": content_type}

        def raise_for_status(self):
            pass

    response = Response()
    response.content = content
    monkeypatch.setattr("gaia_agent.sources.requests.get", lambda url, timeout: response)
    monkeypatch.setattr("gaia_agent.sources.load_gaia_rows", lambda split, token: [{"task_id": "abc", "file_name": "x.png"}])
    fallback = tmp_path / "fallback.png"
    monkeypatch.setattr("gaia_agent.sources.hf_hub_download", lambda **kwargs: str(fallback))
    task = Task("abc", "What?", 1, file_name="x.png")
    download_course_attachments([task])
    assert task.attachment_path == str(fallback)
    assert not (tmp_path / ".runs" / "course-files" / "abc" / "x.png").exists()


def test_course_attachments_with_same_name_are_stored_per_task(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    payloads = {"abc": b"\x89PNG\r\n\x1a\nfirst", "def": b"\x89PNG\r\n\x1a\nsecond"}

    class Response:
        headers = {"Content-Type": "image/png"}

        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    monkeypatch.setattr("gaia_agent.sources.requests.get", lambda url, timeout: Response(payloads[url.rsplit("/", 1)[-1]]))
    tasks = [Task(task_id, "What?", 1, file_name="same.png") for task_id in payloads]
    download_course_attachments(tasks)
    assert tasks[0].attachment_path != tasks[1].attachment_path
    assert {Path(task.attachment_path).read_bytes() for task in tasks} == set(payloads.values())
