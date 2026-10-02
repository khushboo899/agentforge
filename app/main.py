from app.graph.workflow import graph
import os


config = {
    "configurable": {
        "thread_id": "agentforge-user-1"
    }
}


print("\n🤖 AgentForge AI")
print("Type 'exit' to stop.\n")


pdf_path = "documents/test.pdf"


if not os.path.exists(pdf_path):
    print(f"⚠️ PDF not found: {pdf_path}")
    pdf_path = None
else:
    print(f"📄 PDF loaded: {pdf_path}")


while True:

    question = input("\nYou: ")

    if question.lower() == "exit":

        print("\n👋 Goodbye!")

        break

    input_state = {
        "question": question
    }

    if pdf_path:
        input_state["pdf_path"] = pdf_path

    result = graph.invoke(
        input_state,
        config=config
    )

    answer = result.get(
        "answer",
        "No answer generated."
    )

    print("\n🤖 AgentForge:")
    print(answer)

    print("\n" + "=" * 60)