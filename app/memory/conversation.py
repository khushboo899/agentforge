from typing import List, Dict


def add_user_message(
    messages: List[Dict[str, str]],
    question: str
):
    """
    Add the user's question
    to conversation history.
    """

    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    return messages


def add_assistant_message(
    messages: List[Dict[str, str]],
    answer: str
):
    """
    Add the assistant's answer
    to conversation history.
    """

    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return messages