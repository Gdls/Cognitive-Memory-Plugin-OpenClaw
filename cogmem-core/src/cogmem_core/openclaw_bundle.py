"""Locate and install the bundled OpenClaw plugin from the cogmem-core wheel."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

PLUGIN_PACKAGE = "cogmem_core.data"
PLUGIN_DIRNAME = "openclaw_plugin"

DEFAULT_OPENCLAW_CONFIG = {
    "sidecarUrl": "http://127.0.0.1:8787",
    "autoRecall": True,
    "autoCapture": True,
    "autoBuild": True,
    "recallConversationId": "",
}


def bundled_plugin() -> object:
    return files(PLUGIN_PACKAGE).joinpath(PLUGIN_DIRNAME)


def default_install_dir() -> Path:
    return Path.home() / ".openclaw" / "plugins" / "openclaw-cogmem"


def install_openclaw_plugin(dest: str | Path | None = None) -> Path:
    """Copy the bundled OpenClaw plugin into an OpenClaw plugin directory."""
    target = Path(dest).expanduser().resolve() if dest else default_install_dir()
    target.mkdir(parents=True, exist_ok=True)
    copied = _copy_traversable(bundled_plugin(), target)
    if copied < 4:
        raise FileNotFoundError(
            f"bundled OpenClaw plugin is incomplete ({copied} files) at {target}"
        )
    return target


def plugin_manifest(dest: Path | None = None) -> dict[str, object]:
    root = dest or default_install_dir()
    package_json = root / "package.json"
    plugin_json = root / "openclaw.plugin.json"
    payload: dict[str, object] = {
        "install_dir": str(root),
        "sidecar_url": DEFAULT_OPENCLAW_CONFIG["sidecarUrl"],
    }
    if package_json.exists():
        payload["package"] = json.loads(package_json.read_text(encoding="utf-8"))
    if plugin_json.exists():
        payload["plugin"] = json.loads(plugin_json.read_text(encoding="utf-8"))
    return payload


def openclaw_config_snippet(install_dir: Path) -> str:
    path = str(install_dir).replace("\\", "/")
    return (
        '{\n'
        '  "plugins": {\n'
        '    "entries": {\n'
        '      "cogmem": {\n'
        '        "enabled": true,\n'
        '        "config": {\n'
        '          "sidecarUrl": "http://127.0.0.1:8787",\n'
        '          "autoRecall": true,\n'
        '          "autoCapture": true,\n'
        '          "autoBuild": true\n'
        '        },\n'
        '        "hooks": {\n'
        '          "allowPromptInjection": true,\n'
        '          "allowConversationAccess": true\n'
        '        }\n'
        '      }\n'
        '    },\n'
        f'    "load": {{ "paths": ["{path}"] }}\n'
        '  }\n'
        '}'
    )


def _copy_traversable(source: object, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in source.iterdir():
        name = item.name
        if name in {"__pycache__", ".DS_Store"} or name.endswith(".pyc"):
            continue
        target = dest / name
        if item.is_dir():
            copied += _copy_traversable(item, target)
            continue
        target.write_bytes(item.read_bytes())
        copied += 1
    return copied
