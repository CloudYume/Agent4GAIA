from pathlib import Path

from scripts.build_space import build


def test_space_bundle_excludes_private_files(tmp_path: Path):
    destination = tmp_path / "space_bundle"
    build(destination)
    files = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    assert {"README.md", "app.py", "requirements.txt", "pyproject.toml", "config.example.toml"} <= files
    assert "gaia_agent/agent.py" in files
    assert {"gaia_agent/tools/__init__.py", "gaia_agent/tools/data.py", "gaia_agent/tools/vision.py"} <= files
    assert "config.toml" not in files
    assert all(not name.startswith(".runs/") for name in files)
