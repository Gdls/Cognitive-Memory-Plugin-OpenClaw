from __future__ import annotations

from .schemas import CognitiveContext, ContextMemoryItem


def format_cognitive_context(context: CognitiveContext) -> str:
    lines = [
        "Cognitive Context Construction",
        "",
        "Question:",
        context.query,
        "",
        "Schema: Preference=guidance not evidence; Background=trajectory; Situation=current episode; Related=other episodes; Evidence=facts; Temporal=unabsorbed; Execution=tools.",
        "",
        "1. Preference Memory",
    ]
    append_items(lines, context.preference_memory)
    lines.append("")
    lines.append("2. Long-Term Background Memory")
    append_items(lines, context.long_term_background)
    lines.append("")
    lines.append("3. Current Situation")
    append_items(lines, context.current_situation)
    lines.append("")
    lines.append("4. Related Experience")
    append_items(lines, context.related_experience)
    lines.append("")
    lines.append("5. Supporting Evidence")
    append_items(lines, context.supporting_evidence)
    lines.append("")
    lines.append("6. Recent Temporal Memory")
    append_items(lines, context.temporal_memory)
    lines.append("")
    lines.append("7. Tool Execution Memory")
    append_items(lines, context.execution_memory)
    return "\n".join(lines)


def append_items(lines: list[str], items: list[ContextMemoryItem]) -> None:
    if not items:
        lines.append("- none")
        return
    for item in items:
        timestamp = f" time={item.timestamp}" if item.timestamp else ""
        score = f" score={item.score:.3f}" if item.score else ""
        status = str((item.metadata or {}).get("status") or "")
        status_bit = f" status={status}" if status and status != "active" else ""
        lines.append(f"- {item.item_id}{timestamp}{score}{status_bit}: {item.text}")
