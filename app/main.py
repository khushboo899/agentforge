from app.graph.workflow import graph


question = """
Analyze whether a startup should enter the Indian
electric vehicle charging market.
Identify the major challenges, opportunities,
and important factors that should be researched.
"""


result = graph.invoke(
    {
        "question": question
    }
)


print("\n")
print("=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result["answer"])