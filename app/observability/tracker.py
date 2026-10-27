"""AgentForge observability utilities.

Features
--------
- Per-request traces with stable trace IDs
- Thread-safe event recording and tracker registry
- Node/LLM/tool spans with latency and parent-span relationships
- Token usage extraction and per-component aggregation
- Optional input/output cost estimation
- Structured errors and compact trace summaries
- JSONL write-through persistence (events survive most process failures)

The tracker is intentionally dependency-free and does not store tracker objects in
LangGraph state. Store only ``trace_id`` and any returned summary in graph state.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRACE_DIR = Path(os.getenv("AGENTFORGE_TRACE_DIR", "traces")).expanduser()
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# The current span is context-local, so concurrent requests do not share parents.
_CURRENT_SPAN: ContextVar[tuple[str, str] | None] = ContextVar(
    "agentforge_current_span", default=None
)
_TRACKERS: dict[str, "ObservabilityTracker"] = {}
_REGISTRY_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _num(value: Any) -> int:
    """Convert a token count to a non-negative integer; unknown values become 0."""
    try:
        result = int(value or 0)
        return max(0, result)
    except (TypeError, ValueError, OverflowError):
        return 0


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _usage_sources(response: Any) -> list[Mapping[str, Any]]:
    """Collect common usage payloads, with more specific metadata taking priority."""
    sources: list[Mapping[str, Any]] = []

    if isinstance(response, Mapping):
        sources.append(_as_mapping(response.get("usage")))
        metadata = _as_mapping(response.get("response_metadata"))
        sources.append(_as_mapping(metadata.get("token_usage")))
        sources.append(_as_mapping(metadata.get("usage")))
        sources.append(metadata)

    metadata = _as_mapping(getattr(response, "response_metadata", None))
    sources.append(_as_mapping(metadata.get("token_usage")))
    sources.append(_as_mapping(metadata.get("usage")))
    sources.append(metadata)

    raw_usage = getattr(response, "usage", None)
    if isinstance(raw_usage, Mapping):
        sources.append(raw_usage)
    elif raw_usage is not None:
        sources.append({
            key: getattr(raw_usage, key)
            for key in (
                "prompt_tokens", "completion_tokens", "total_tokens",
                "input_tokens", "output_tokens",
            )
            if hasattr(raw_usage, key)
        })

    sources.append(_as_mapping(getattr(response, "usage_metadata", None)))
    return sources


def extract_usage(response: Any) -> dict[str, int]:
    """Normalize common LangChain, Groq, and OpenAI-style token usage formats.

    A zero count means either the provider reported zero or usage was unavailable;
    callers should inspect ``usage_available`` separately if that distinction matters.
    """
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    for source in _usage_sources(response):
        if "input_tokens" in source or "prompt_tokens" in source:
            value = source.get("input_tokens", source.get("prompt_tokens"))
            if value is not None:
                input_tokens = _num(value)
        if "output_tokens" in source or "completion_tokens" in source:
            value = source.get("output_tokens", source.get("completion_tokens"))
            if value is not None:
                output_tokens = _num(value)
        if "total_tokens" in source and source.get("total_tokens") is not None:
            total_tokens = _num(source.get("total_tokens"))

    input_value = input_tokens or 0
    output_value = output_tokens or 0
    total_value = total_tokens if total_tokens is not None else input_value + output_value
    available = input_tokens is not None or output_tokens is not None or total_tokens is not None

    return {
        "input_tokens": input_value,
        "output_tokens": output_value,
        "total_tokens": total_value,
        "usage_available": int(available),
    }


def estimate_cost(usage: Mapping[str, Any], model: str | None = None) -> float:
    """Estimate USD cost from configured rates per one million tokens.

    Supports global rates and optional model-specific overrides:
    MODEL_INPUT_COST_PER_1M, MODEL_OUTPUT_COST_PER_1M,
    MODEL_<NORMALIZED_MODEL_NAME>_INPUT_COST_PER_1M, and
    MODEL_<NORMALIZED_MODEL_NAME>_OUTPUT_COST_PER_1M.
    Rates are user-configured; this function does not fetch provider pricing.
    """
    def rate(env_name: str) -> float:
        try:
            return max(0.0, float(os.getenv(env_name, "0")))
        except (TypeError, ValueError):
            return 0.0

    model_key = ""
    if model:
        model_key = "MODEL_" + "_".join(
            part for part in "".join(ch if ch.isalnum() else "_" for ch in model.upper()).split("_") if part
        )

    input_rate = rate(f"{model_key}_INPUT_COST_PER_1M") if model_key else 0.0
    output_rate = rate(f"{model_key}_OUTPUT_COST_PER_1M") if model_key else 0.0
    if input_rate == 0.0:
        input_rate = rate("MODEL_INPUT_COST_PER_1M")
    if output_rate == 0.0:
        output_rate = rate("MODEL_OUTPUT_COST_PER_1M")

    return (
        _num(usage.get("input_tokens")) / 1_000_000 * input_rate
        + _num(usage.get("output_tokens")) / 1_000_000 * output_rate
    )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class ObservabilityTracker:
    """Collect events and metrics for one AgentForge request/trace."""

    def __init__(self, trace_id: Optional[str] = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.started_at = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.component_usage: dict[str, dict[str, int | float]] = defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "llm_calls": 0,
                "usage_missing_calls": 0,
            }
        )
        self._lock = threading.RLock()
        self._closed = False
        self._summary_recorded = False
        self.persistence_error: str | None = None
        self.path = TRACE_DIR / f"{self.trace_id}.jsonl"

    def _append_jsonl(self, record: Mapping[str, Any]) -> None:
        """Write each event immediately; a disk error must not break the agent."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                file.flush()
        except OSError as exc:
            self.persistence_error = f"{type(exc).__name__}: {exc}"

    def event(self, event_type: str, **payload: Any) -> dict[str, Any]:
        """Record and immediately persist one event.

        Standard fields are assigned last so payload values cannot spoof them.
        """
        with self._lock:
            record = {
                **payload,
                "timestamp": utc_now(),
                "trace_id": self.trace_id,
                "event": event_type,
            }
            self.events.append(record)
            self._append_jsonl(record)
            return record

    @contextmanager
    def span(self, name: str, kind: str = "node", **payload: Any) -> Iterator[None]:
        """Measure a node, LLM call, or tool call and record its parent span."""
        start = time.perf_counter()
        span_id = uuid.uuid4().hex[:16]
        current = _CURRENT_SPAN.get()
        parent_span_id = current[1] if current and current[0] == self.trace_id else None
        token = _CURRENT_SPAN.set((self.trace_id, span_id))
        reserved_span_fields = {"name", "kind", "span_id", "parent_span_id"}
        safe_payload = {
            key: value for key, value in payload.items()
            if key not in reserved_span_fields
        }
        common = {
            **safe_payload,
            "name": name,
            "kind": kind,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
        }
        status = "ok"
        error_message: str | None = None
        self.event("span_start", **common)
        try:
            yield
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            self.event(
                "span_error",
                **common,
                error_type=type(exc).__name__,
                error=error_message,
            )
            raise
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            try:
                self.event(
                    "span_end",
                    **common,
                    status=status,
                    latency_ms=latency_ms,
                    error=error_message,
                )
            finally:
                _CURRENT_SPAN.reset(token)

    def llm_usage(
        self,
        model: str,
        usage: Mapping[str, Any],
        component: str = "unknown",
        **payload: Any,
    ) -> dict[str, Any]:
        """Record an LLM call and aggregate usage globally and per component."""
        normalized = {
            "input_tokens": _num(usage.get("input_tokens")),
            "output_tokens": _num(usage.get("output_tokens")),
            "total_tokens": _num(usage.get("total_tokens")),
        }
        if normalized["total_tokens"] == 0:
            normalized["total_tokens"] = (
                normalized["input_tokens"] + normalized["output_tokens"]
            )
        usage_available = bool(usage.get("usage_available", any(normalized.values())))
        cost = estimate_cost(normalized, model=model)

        with self._lock:
            self.total_input_tokens += normalized["input_tokens"]
            self.total_output_tokens += normalized["output_tokens"]
            self.total_tokens += normalized["total_tokens"]
            self.total_cost_usd += cost
            component_stats = self.component_usage[component]
            component_stats["input_tokens"] += normalized["input_tokens"]
            component_stats["output_tokens"] += normalized["output_tokens"]
            component_stats["total_tokens"] += normalized["total_tokens"]
            component_stats["estimated_cost_usd"] += cost
            component_stats["llm_calls"] += 1
            if not usage_available:
                component_stats["usage_missing_calls"] += 1

        reserved_llm_fields = {
            "model", "component", "input_tokens", "output_tokens",
            "total_tokens", "usage_available", "estimated_cost_usd", "event",
            "trace_id", "timestamp",
        }
        safe_payload = {
            key: value for key, value in payload.items()
            if key not in reserved_llm_fields
        }
        return self.event(
            "llm_call",
            **safe_payload,
            model=model,
            component=component,
            **normalized,
            usage_available=usage_available,
            estimated_cost_usd=round(cost, 10),
        )

    def tool_call(self, tool: str, **payload: Any) -> dict[str, Any]:
        """Record a tool invocation; wrap the actual execution in span(kind='tool') for latency."""
        return self.event("tool_call", tool=tool, **payload)

    def error(
        self,
        component: str,
        exc: Exception,
        recoverable: bool = True,
        **payload: Any,
    ) -> dict[str, Any]:
        """Record a structured error without suppressing the original exception."""
        return self.event(
            "error",
            component=component,
            error_type=type(exc).__name__,
            error=str(exc),
            recoverable=bool(recoverable),
            **payload,
        )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            per_component = {
                name: {
                    **stats,
                    "estimated_cost_usd": round(float(stats["estimated_cost_usd"]), 10),
                }
                for name, stats in self.component_usage.items()
            }
            result: dict[str, Any] = {
                "trace_id": self.trace_id,
                "latency_ms": round((time.perf_counter() - self.started_at) * 1000, 2),
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "estimated_cost_usd": round(self.total_cost_usd, 10),
                "llm_calls": sum(int(v["llm_calls"]) for v in self.component_usage.values()),
                "event_count": len(self.events),
                "per_component": per_component,
            }
            if self.persistence_error:
                result["persistence_error"] = self.persistence_error
            return result

    def persist(self) -> Path:
        """Write a final summary; regular events are already written through to JSONL."""
        with self._lock:
            if self._summary_recorded:
                return self.path
            summary = self.summary()
            record = {
                "timestamp": utc_now(),
                "trace_id": self.trace_id,
                "event": "trace_summary",
                **summary,
            }
            self._append_jsonl(record)
            self._summary_recorded = True
            self._closed = True
            return self.path


# ---------------------------------------------------------------------------
# Registry and finalization
# ---------------------------------------------------------------------------
def get_tracker(state: dict[str, Any]) -> ObservabilityTracker:
    """Return the same in-process tracker for a graph execution's trace ID."""
    trace_id = state.get("trace_id")
    if not trace_id:
        trace_id = uuid.uuid4().hex
        state["trace_id"] = trace_id

    with _REGISTRY_LOCK:
        tracker = _TRACKERS.get(str(trace_id))
        if tracker is None:
            tracker = ObservabilityTracker(str(trace_id))
            _TRACKERS[str(trace_id)] = tracker
        return tracker


def persist_and_close(state: dict[str, Any]) -> dict[str, Any]:
    """Persist the final summary and always release the registry entry."""
    tracker = get_tracker(state)
    try:
        path = tracker.persist()
        summary = tracker.summary()
        summary["trace_file"] = str(path)
        return summary
    finally:
        with _REGISTRY_LOCK:
            _TRACKERS.pop(tracker.trace_id, None)