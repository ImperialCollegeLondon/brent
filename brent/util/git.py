# Standard imports
import subprocess

# Internal imports
import brent.paths


def get_commit() -> str:
    # NOTE: Solution adapted from https://stackoverflow.com/a/21901260 (accessed 2024-11-20)
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=brent.paths.ROOT_DIR,
        )
        .decode("ascii")
        .strip()
    )


def has_uncommitted_changes() -> bool:
    """Return whether the repository differs from its current HEAD commit."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=brent.paths.ROOT_DIR,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return bool(status.stdout.strip())
