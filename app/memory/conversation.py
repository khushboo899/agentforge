from typing import List, Dict


def add_user_message(
    messages: List[Dict[str, str]],
    question: str
):

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

    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return messages