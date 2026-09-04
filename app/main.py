from app.graph.workflow import graph


messages = []

print("\n🤖 AgentForge AI")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("\n👋 Goodbye!")
        break

    result = graph.invoke(
        {
            "question": question,
            "messages": messages
        }
    )

    answer = result.get(
        "answer",
        "No answer generated."
    )

    messages = result.get(
        "messages",
        messages
    )

    print("\n🤖 AgentForge:")
    print(answer)
    print("\n" + "=" * 60 + "\n")