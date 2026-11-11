import shutil
from pathlib import Path

import nox

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "yes"

PYTHON_VERSIONS = ["3.11", "3.12", "3.13", "3.14", "3.15", "3.15t"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Build the sdist, install it with test extras and run the test suites."""
    dist_dir = Path(session.create_tmp()) / "dist"
    shutil.rmtree(dist_dir, ignore_errors=True)
    session.run("uv", "build", "--sdist", "--out-dir", str(dist_dir), external=True)
    (sdist,) = dist_dir.glob("*.tar.gz")
    session.install("--reinstall-package", "fastfloret", f"{sdist}[test]")

    session.run("python", "-Werror", "-c", "import fastfloret")
    session.run(
        "python", "-m", "pytest", "--pyargs", "fastfloret", "-Werror", *session.posargs
    )
    session.run("python", "tests/runtests.py", "-u")
