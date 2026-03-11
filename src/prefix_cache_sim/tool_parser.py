import json
from typing import Any, Dict, List, Optional


class ToolCall:
    def __init__(
        self,
        id: str,
        type: str = "function",
        function: Optional[Dict[str, str]] = None,
    ):
        self.id = id
        self.type = type
        self.function = function or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function,
        }


def parse_tool_calls(message: Dict[str, Any]) -> List[ToolCall]:
    tool_calls = message.get("tool_calls", [])
    return [ToolCall(**tc) if isinstance(tc, dict) else tc for tc in tool_calls]


def format_tool_call_for_qwen(tool_call: ToolCall) -> str:
    if tool_call.type == "function":
        func_name = tool_call.function.get("name", "")
        func_args = tool_call.function.get("arguments", "{}")
        try:
            args_dict = json.loads(func_args) if isinstance(func_args, str) else func_args
            args_str = json.dumps(args_dict, ensure_ascii=False)
        except json.JSONDecodeError:
            args_str = func_args
        return f"call_{func_name}({args_str})"
    return ""


def format_tool_response_for_qwen(content: str) -> str:
    return f"response_{content}"


def convert_chatml_with_tools(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    converted = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "assistant":
            tool_calls = parse_tool_calls(msg)

            if tool_calls:
                tool_call_strs = [format_tool_call_for_qwen(tc) for tc in tool_calls]
                tool_call_content = "\n".join(tool_call_strs)

                if content:
                    converted.append(
                        {
                            "role": "assistant",
                            "content": content,
                        }
                    )
                    converted.append(
                        {
                            "role": "assistant",
                            "content": tool_call_content,
                        }
                    )
                else:
                    converted.append(
                        {
                            "role": "assistant",
                            "content": tool_call_content,
                        }
                    )
            else:
                converted.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
        elif role == "tool":
            tool_response = format_tool_response_for_qwen(content)
            converted.append(
                {
                    "role": "tool",
                    "content": tool_response,
                }
            )
        else:
            converted.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return converted
