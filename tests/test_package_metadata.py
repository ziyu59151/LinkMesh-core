from pathlib import Path
import tomllib

from linkmesh import __version__


def test_package_version_matches_project_metadata() -> None:
    project = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(project.read_text(encoding="utf-8"))

    assert __version__ == metadata["project"]["version"]
    assert metadata["project"]["requires-python"] == ">=3.11"
