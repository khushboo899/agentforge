"""
AgentForge CLI entry point.
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv

from app.graph.workflow import workflow


load_dotenv()


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(
    question: str,
    thread_id: str,
    messages: list[dict[str, str]]
):

    state = {

        "question":
            question,

        "messages":
            messages,

        "retry_count":
            0,

        "max_retries":
            2,

        "trace_id":
            uuid.uuid4().hex[:16],

        "errors":
            []

    }


    config = {

        "configurable": {

            "thread_id":
                thread_id

        }

    }


    return workflow.invoke(
        state,
        config=config
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "🤖 AgentForge AI"
    )

    print(
        "Type 'exit' to stop."
    )


    thread_id = (
        uuid.uuid4().hex
    )

    messages = []


    while True:

        try:

            question = input(
                "\nYou: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\n👋 Goodbye!"
            )

            break


        if not question:

            continue


        if question.lower() == "exit":

            print(
                "\n👋 Goodbye!"
            )

            break


        try:

            result = run_agent(

                question,

                thread_id,

                messages

            )


            answer = (
                result.get(
                    "answer"
                )
                or
                "I couldn't generate an answer right now."
            )


            print(
                f"\nAgentForge:\n{answer}"
            )


            messages = result.get(
                "messages",
                messages
            )


            messages = messages + [

                {
                    "role":
                        "assistant",

                    "content":
                        answer

                }

            ]


            # ------------------------------------------------
            # OBSERVABILITY SUMMARY
            # ------------------------------------------------

            summary = result.get(
                "trace_summary"
            )


            if summary:

                print(

                    f"\n[Trace "
                    f"{summary.get('trace_id')}] "

                    f"{summary.get('latency_ms', 0)} ms | "

                    f"{summary.get('total_tokens', 0)} tokens | "

                    f"trace: "
                    f"{summary.get('trace_file', 'traces/')}"
                )


        except Exception as exc:

            print(

                f"\n[AgentForge] "
                f"Unexpected error: "
                f"{type(exc).__name__}: "
                f"{exc}"

            )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()