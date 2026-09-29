"""Command-line entry for the cogmem-core wheel: sidecar + OpenClaw plugin install."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


def _package_version() -> str:
    try:
        return version("cogmem-core")
    except PackageNotFoundError:
        return "0.3.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cogmem",
        description="CogMem core: run the memory sidecar or install the OpenClaw plugin.",
    )
    parser.add_argument("--version", action="store_true", help="Print cogmem-core version.")
    sub = parser.add_subparsers(dest="command")

    sidecar = sub.add_parser("sidecar", help="Start the CogMem HTTP sidecar.")
    sidecar.add_argument("--store", default=None)
    sidecar.add_argument("--host", default=None)
    sidecar.add_argument("--port", type=int, default=None)

    install = sub.add_parser(
        "openclaw-install",
        help="Copy the bundled OpenClaw plugin into ~/.openclaw/plugins/openclaw-cogmem.",
    )
    install.add_argument(
        "--dest",
        default=None,
        help="Plugin directory. Default: ~/.openclaw/plugins/openclaw-cogmem",
    )

    sub.add_parser("plugin-path", help="Print the default OpenClaw plugin install path.")

    args = parser.parse_args(argv)
    if args.version or args.command == "version":
        print(_package_version())
        return 0
    if args.command == "sidecar":
        return _run_sidecar(args)
    if args.command == "openclaw-install":
        return _install_openclaw(args.dest)
    if args.command == "plugin-path":
        from .openclaw_bundle import default_install_dir

        print(default_install_dir())
        return 0
    parser.print_help()
    return 0


def _run_sidecar(args: argparse.Namespace) -> int:
    import os

    from .sidecar.__main__ import main as sidecar_main

    argv: list[str] = []
    store = args.store or os.environ.get("COGMEM_STORE")
    host = args.host or os.environ.get("COGMEM_HOST")
    port = args.port
    if store:
        argv.extend(["--store", store])
    if host:
        argv.extend(["--host", host])
    if port is not None:
        argv.extend(["--port", str(port)])
    sidecar_main(argv or None)
    return 0


def _install_openclaw(dest: str | None) -> int:
    from .openclaw_bundle import install_openclaw_plugin, openclaw_config_snippet

    target = install_openclaw_plugin(dest)
    print(f"Installed OpenClaw plugin to: {target}")
    print("Next: enable it in ~/.openclaw/openclaw.json, then restart the gateway.")
    print()
    print(openclaw_config_snippet(Path(target)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _openclaw_install_entry() -> None:
    raise SystemExit(main(["openclaw-install", *sys.argv[1:]]))
