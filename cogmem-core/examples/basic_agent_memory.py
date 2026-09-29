from cogmem_core import CogMem


def main() -> None:
    memory = CogMem(store_path="./cogmem_store")
    memory.append_events(
        conversation_id="sess_001",
        events=[
            {"event_type": "user_message", "text": "帮我查订单状态"},
            {"event_type": "assistant_message", "text": "订单已经发货，预计 8 月 5 日送达。"},
        ],
    )
    print(memory.format_prompt_context("sess_001", "用户之前的订单状态是什么？"))


if __name__ == "__main__":
    main()
