"""Backward-compatible imports for event contracts."""

from .contracts.events import CogMemEvent, EventType, normalize_events

__all__ = ["CogMemEvent", "EventType", "normalize_events"]
