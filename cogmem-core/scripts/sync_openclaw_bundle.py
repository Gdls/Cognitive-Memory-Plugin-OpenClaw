"""Copy repo OpenClaw plugin files into the wheel package data directory."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "openclaw-cogmem"
DEST = ROOT / "src" / "cogmem_core" / "data" / "openclaw_plugin"
FILES = [
    "index.js",
    "client.js",
    "tools.js",
    "openclaw.plugin.json",
    "package.json",
    "README.md",
]


def sync() -> Path:
    if not SRC.exists():
        raise FileNotFoundError(f"OpenClaw plugin source not found: {SRC}")
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(SRC / name, DEST / name)
    skill_src = SRC / "skill" / "cogmem-memory-guide" / "SKILL.md"
    skill_dest = DEST / "skill" / "cogmem-memory-guide"
    skill_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(skill_src, skill_dest / "SKILL.md")
    return DEST


if __name__ == "__main__":
    print(sync())
