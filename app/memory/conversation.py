from typing import List, Dict


def add_user_message(
    messages: List[Dict[str, str]],
    question: str
):

    updated_messages = messages.copy()

    updated_messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    return updated_messages


def add_assistant_message(
    messages: List[Dict[str, str]],
    answer: str
):

    updated_messages = messages.copy()

    updated_messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return updated_messages