import sqlite3

from typing import Callable

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.checkpoint.sqlite import (
    SqliteSaver
)

from app.graph.state import AgentState

from app.graph.nodes import (
    memory_node,
    analyze_task,
    create_plan,
    execute_selected_agents,
    generate_answer,
    critic_node,
    prepare_retry,
)

from app.observability import (
    get_tracker,
    persist_and_close,
)


# ============================================================
# OBSERVED NODE WRAPPER
# ============================================================

def observed(
    name: str,
    fn: Callable
):

    """
    Adds:

    - execution tracing
    - latency
    - structured errors

    without storing the tracker object
    inside LangGraph state.
    """

    def wrapper(
        state: AgentState
    ):

        tracker = get_tracker(
            state
        )

        try:

            with tracker.span(
                name,
                kind="graph_node"
            ):

                result = fn(
                    state
                )


            if not isinstance(
                result,
                dict
            ):

                result = {}


            result["trace_id"] = (
                tracker.trace_id
            )

            return result


        except Exception as exc:

            tracker.error(
                name,
                exc,
                recoverable=True
            )

            errors = list(
                state.get(
                    "errors",
                    []
                )
            )

            errors.append({

                "component":
                    name,

                "type":
                    type(exc).__name__,

                "message":
                    str(exc)

            })


            return {

                "trace_id":
                    tracker.trace_id,

                "errors":
                    errors,

                "fatal_error":
                    str(exc)

            }


    wrapper.__name__ = name

    return wrapper


# ============================================================
# AFTER GENERATION
# ============================================================

def route_after_generate(
    state: AgentState
):

    # If generation failed, don't waste another LLM call on critic.
    if state.get(
        "generation_error"
    ):

        return "finalize"


    # If deterministic fallback was used,
    # also don't invoke critic.
    if state.get(
        "generation_fallback"
    ):

        return "finalize"


    if state.get(
        "fatal_error"
    ):

        return "finalize"


    return "critic"


# ============================================================
# AFTER CRITIC
# ============================================================

def route_after_critic(
    state: AgentState
):

    if state.get(
        "fatal_error"
    ):

        return "finalize"


    if state.get(
        "approved"
    ):

        return "finalize"


    if (
        state.get(
            "retry_count",
            0
        )
        >=
        state.get(
            "max_retries",
            2
        )
    ):

        return "finalize"


    return "retry"


# ============================================================
# FINALIZE
# ============================================================

def finalize_node(
    state: AgentState
):

    tracker = get_tracker(
        state
    )

    with tracker.span(
        "finalize",
        kind="observability"
    ):

        summary = persist_and_close(
            state
        )


    return {

        "trace_id":
            tracker.trace_id,

        "trace_summary":
            summary

    }


# ============================================================
# GRAPH
# ============================================================

builder = StateGraph(
    AgentState
)


builder.add_node(
    "memory",
    observed(
        "memory",
        memory_node
    )
)


builder.add_node(
    "analyze",
    observed(
        "analyze",
        analyze_task
    )
)


builder.add_node(
    "plan",
    observed(
        "plan",
        create_plan
    )
)


builder.add_node(
    "execute",
    observed(
        "execute",
        execute_selected_agents
    )
)


builder.add_node(
    "generate",
    observed(
        "generate",
        generate_answer
    )
)


builder.add_node(
    "critic",
    observed(
        "critic",
        critic_node
    )
)


builder.add_node(
    "retry",
    observed(
        "retry",
        prepare_retry
    )
)


builder.add_node(
    "finalize",
    finalize_node
)


# ============================================================
# NORMAL FLOW
# ============================================================

builder.add_edge(
    START,
    "memory"
)

builder.add_edge(
    "memory",
    "analyze"
)

builder.add_edge(
    "analyze",
    "plan"
)

builder.add_edge(
    "plan",
    "execute"
)

builder.add_edge(
    "execute",
    "generate"
)


# ============================================================
# GENERATION ROUTER
# ============================================================

builder.add_conditional_edges(

    "generate",

    route_after_generate,

    {

        "critic":
            "critic",

        "finalize":
            "finalize"

    }

)


# ============================================================
# CRITIC ROUTER
# ============================================================

builder.add_conditional_edges(

    "critic",

    route_after_critic,

    {

        "finalize":
            "finalize",

        "retry":
            "retry"

    }

)


# ============================================================
# RETRY
# ============================================================

builder.add_edge(
    "retry",
    "plan"
)


# ============================================================
# END
# ============================================================

builder.add_edge(
    "finalize",
    END
)


# ============================================================
# SQLITE CHECKPOINTING
# ============================================================

connection = sqlite3.connect(
    "checkpoints.sqlite",
    check_same_thread=False
)

checkpointer = SqliteSaver(
    connection
)


workflow = builder.compile(
    checkpointer=checkpointer
)