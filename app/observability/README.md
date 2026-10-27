# AgentForge Phase 5 Observability

Every user turn receives a unique trace ID.

AgentForge records:

- graph node execution
- node latency
- LLM latency
- browser-search execution
- token usage
- optional estimated cost
- errors
- final trace summary

Traces are stored locally in:

traces/<trace_id>.jsonl


## Events

### span_start

Indicates the beginning of a node/tool/LLM operation.

### span_end

Contains:

- status
- latency_ms

### llm_call

Contains:

- model
- input_tokens
- output_tokens
- total_tokens
- estimated_cost_usd

### tool_call

Records external tool execution.

### error

Records:

- component
- error type
- error message
- recoverability

### trace_summary

Contains the complete execution summary.


## Configuration

Add to `.env`:

AGENTFORGE_TRACE_DIR=traces

MODEL_INPUT_COST_PER_1M=0

MODEL_OUTPUT_COST_PER_1M=0

AGENTFORGE_DEBUG=false


Cost tracking is disabled by default because provider pricing can change.

Token counts and latency work even when cost is disabled.


## 429 handling

Provider rate-limit errors are recorded in the trace.

AgentForge does not repeatedly call the provider after a generation failure.

If RAG results exist, AgentForge can provide a deterministic fallback based on
the retrieved document evidence.