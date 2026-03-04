import json
import os
from typing import Dict, List, Optional

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download


def load_locomo_dataset(
    split: str = "train", max_sessions: Optional[int] = None
) -> List[List[Dict[str, str]]]:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    file_path = hf_hub_download(
        repo_id="Percena/locomo-mc10",
        filename="transformed/locomo_mc10_with_name.json",
        repo_type="dataset",
    )

    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    sessions = []
    for item in data:
        if max_sessions and len(sessions) >= max_sessions:
            break

        haystack_sessions = item.get("haystack_sessions", [])
        if not haystack_sessions:
            continue

        for session_msgs in haystack_sessions:
            if max_sessions and len(sessions) >= max_sessions:
                break

            session = []
            for msg in session_msgs:
                role = msg.get("role", "user")
                name = msg.get("name", "")
                content = msg.get("content", "")
                if name:
                    content = f"{name}: {content}"
                session.append({"role": role, "content": content})

            if session:
                sessions.append(session)

    return sessions
