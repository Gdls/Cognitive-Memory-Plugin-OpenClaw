"""Build a pip-installable cogmem-core wheel (MemOS-style one-click install)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sync_openclaw_bundle import sync

    dest = sync()
    print(f"bundled OpenClaw plugin -> {dest}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "build"], cwd=ROOT)
    subprocess.check_call([sys.executable, "-m", "build", "--wheel"], cwd=ROOT)
    dist = ROOT / "dist"
    wheels = sorted(dist.glob("cogmem_core-*.whl"))
    if not wheels:
        raise SystemExit("wheel was not created in dist/")
    print(f"wheel: {wheels[-1]}")
    print("install: pip install " + str(wheels[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
