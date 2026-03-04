import random
from typing import Dict, List

from prefix_cache_sim.tokenizer import Qwen2Tokenizer


def chatml_messages_to_prompt(
    messages: List[Dict[str, str]], tokenizer: Qwen2Tokenizer
) -> List[int]:
    tokens = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_tokens = tokenizer.encode_chatml_message(role, content)
        tokens.extend(msg_tokens)
    return tokens


def generate_fake_chatml_data(
    num_sessions: int = 100, messages_per_session: int = 5
) -> List[List[Dict[str, str]]]:
    system_prompts = [
        "You are a helpful assistant.",
        "You are a coding assistant.",
        "You are a friendly chatbot.",
    ]

    user_messages = [
        "Hello how are you",
        "What is Python",
        "Explain machine learning",
        "Write a function to add numbers",
        "What is the weather today",
        "Tell me a joke",
        "How do I learn programming",
        "What is AI",
        "Can you help me with math",
        "What are your capabilities",
    ]

    assistant_messages = [
        "I am doing well thank you",
        "Python is a high level programming language",
        "Machine learning is a subset of AI that enables systems to learn from data",
        "Here is a function: def add a b return a plus b",
        "I am sorry I do not have access to weather data",
        "Why did the developer go broke because he used up all his cache",
        "Start with basics and practice daily",
        "Artificial Intelligence is the simulation of human intelligence by machines",
        "I can help with various math problems",
        "I can answer questions write code and assist with many tasks",
    ]

    sessions = []

    base_system = system_prompts[0]

    session_variations = []
    for i in range(50):
        session_variations.append(
            {
                "system": system_prompts[i % len(system_prompts)],
                "user": user_messages[i % len(user_messages)],
                "assistant": assistant_messages[i % len(assistant_messages)],
            }
        )

    for session_idx in range(num_sessions):
        session = []

        session.append(
            {
                "role": "system",
                "content": base_system,
            }
        )

        for msg_idx in range(random.randint(2, messages_per_session)):
            variation = session_variations[(session_idx + msg_idx) % len(session_variations)]

            session.append(
                {
                    "role": "user",
                    "content": variation["user"],
                }
            )
            session.append(
                {
                    "role": "assistant",
                    "content": variation["assistant"],
                }
            )

        sessions.append(session)

    return sessions
