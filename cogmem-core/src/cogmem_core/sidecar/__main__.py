from __future__ import annotations

import argparse
import os

import uvicorn

from .app import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the CogMem HTTP sidecar.")
    parser.add_argument("--store", default=os.environ.get("COGMEM_STORE", "./cogmem_store"))
    parser.add_argument("--host", default=os.environ.get("COGMEM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("COGMEM_PORT", "8787")))
    args = parser.parse_args(argv)
    uvicorn.run(create_app(args.store), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
