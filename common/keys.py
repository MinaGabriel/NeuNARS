"""The DeepInfra API key: $DEEPINFRA_API_KEY if set, else the git-ignored file .deepinfra_key here or one folder up.
Never write the key into a tracked file: the repository is public. (Here: common/, so "one folder up" is the repository.)"""
import os
from pathlib import Path


def deepinfra_key():
    if os.environ.get("DEEPINFRA_API_KEY"):
        return os.environ["DEEPINFRA_API_KEY"]
    for folder in (Path(__file__).parent, Path(__file__).parent.parent):
        if (folder / ".deepinfra_key").exists():
            return (folder / ".deepinfra_key").read_text().strip()
    raise SystemExit("Set DEEPINFRA_API_KEY, or put the key in .deepinfra_key (git-ignored)")
