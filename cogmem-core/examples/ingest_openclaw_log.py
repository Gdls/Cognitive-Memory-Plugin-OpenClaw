"""Ingest an OpenClaw session JSONL log into CogMem and print recalled context.

Default input is the bundled mini fixture. Pass your own session jsonl as the
first argument to ingest a real log.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cogmem_core import CogMem, OpenClawAdapter

DEFAULT_LOG = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "openclaw"
    / "mini_session.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", nargs="?", default=str(DEFAULT_LOG), help="OpenClaw session JSONL path")
    parser.add_argument("--query", default="用户之前问过什么天气？")
    parser.add_argument("--store", default="./cogmem_store")
    args = parser.parse_args()

    log_path = Path(args.log_path).expanduser().resolve()
    if not log_path.is_file():
        raise SystemExit(
            f"找不到 OpenClaw 日志: {log_path}\n"
            "请传入 jsonl 路径，例如: python -m examples.ingest_openclaw_log /path/to/session.jsonl"
        )

    store_path = Path(args.store).expanduser().resolve()
    adapter = OpenClawAdapter()
    conversion = adapter.convert(log_path)
    memory = CogMem(store_path=store_path)
    appended = memory.append_events(conversion.conversation_id, conversion.events)
    build = memory.build_memory(conversion.conversation_id)
    prompt = memory.format_prompt_context(conversion.conversation_id, args.query)

    print(f"log: {log_path}")
    print(f"store: {store_path}")
    print(f"conversation_id: {conversion.conversation_id}")
    print(f"records: {conversion.record_count}")
    print(f"events: {len(conversion.events)}")
    print(f"skipped: {conversion.skipped}")
    print(f"appended: {appended}")
    print(f"build: {build}")
    print()
    print(prompt)


if __name__ == "__main__":
    main()
