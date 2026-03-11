import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prefix_cache_sim.tool_parser import (
    ToolCall,
    convert_chatml_with_tools,
    format_tool_call_for_qwen,
    format_tool_response_for_qwen,
    parse_tool_calls,
)


def test_parse_tool_calls():
    message = {
        "role": "assistant",
        "content": "Let me check the weather.",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Beijing", "unit": "celsius"}',
                },
            }
        ],
    }

    tool_calls = parse_tool_calls(message)

    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_123"
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function["name"] == "get_weather"


def test_format_tool_call_for_qwen():
    tool_call = ToolCall(
        id="call_123",
        type="function",
        function={"name": "get_weather", "arguments": '{"location": "Beijing"}'},
    )

    formatted = format_tool_call_for_qwen(tool_call)

    assert "call_get_weather" in formatted
    assert "Beijing" in formatted


def test_format_tool_response_for_qwen():
    response = format_tool_response_for_qwen("The weather is 15°C")

    assert response == "response_The weather is 15°C"


def test_convert_chatml_with_tools():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Beijing"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "15°C"},
        {"role": "assistant", "content": "It's 15°C."},
    ]

    converted = convert_chatml_with_tools(messages)

    assert len(converted) == 6
    assert converted[0]["role"] == "system"
    assert converted[1]["role"] == "user"
    assert converted[2]["role"] == "assistant"
    assert converted[2]["content"] == "Let me check"
    assert converted[3]["role"] == "assistant"
    assert "call_get_weather" in converted[3]["content"]
    assert converted[4]["role"] == "tool"
    assert converted[4]["content"].startswith("response_")
    assert converted[5]["role"] == "assistant"


def test_load_chatml_jsonl():
    from prefix_cache_sim.data_loader import load_chatml_jsonl

    sessions = load_chatml_jsonl("tests/fixtures/dummy_chatml.jsonl")

    assert len(sessions) == 5
    assert all(isinstance(session, list) for session in sessions)
    assert len(sessions[0]) == 5
    assert sessions[0][0]["role"] == "system"


def test_multiple_tool_calls():
    messages = [
        {"role": "user", "content": "Get weather and time"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"},
                },
            ],
        },
    ]

    converted = convert_chatml_with_tools(messages)

    assert len(converted) == 2
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "assistant"
    assert "call_get_weather" in converted[1]["content"]
    assert "call_get_time" in converted[1]["content"]


if __name__ == "__main__":
    print("Running tests...")
    test_parse_tool_calls()
    print("✓ test_parse_tool_calls")

    test_format_tool_call_for_qwen()
    print("✓ test_format_tool_call_for_qwen")

    test_format_tool_response_for_qwen()
    print("✓ test_format_tool_response_for_qwen")

    test_convert_chatml_with_tools()
    print("✓ test_convert_chatml_with_tools")

    test_load_chatml_jsonl()
    print("✓ test_load_chatml_jsonl")

    test_multiple_tool_calls()
    print("✓ test_multiple_tool_calls")

    print("\nAll tests passed!")
