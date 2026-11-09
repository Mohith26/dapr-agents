#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Tool calling, structured output and streaming for MistralChatClient.

These tests run the real Mistral SDK against an in-process fake of the
Mistral chat API (``httpx.MockTransport``), so they check both the request
body the SDK actually sends and how real SDK response objects are parsed.
"""

import json
from typing import Any, Dict, List

import httpx
import pytest
from mistralai.client import Mistral
from pydantic import BaseModel

from dapr_agents.llm.mistral.chat import MistralChatClient
from dapr_agents.llm.mistral.utils import content_to_text
from dapr_agents.tool import tool
from dapr_agents.types.message import LLMChatResponse, ToolMessage, UserMessage
from tests.llm.streaming_test_harness import StreamingComplianceHarness


class FakeMistralAPI:
    """Records chat requests and replies with queued JSON or SSE bodies."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self._replies: List[Any] = []

    def reply_json(self, body: Dict[str, Any]) -> None:
        self._replies.append(("json", body))

    def reply_stream(self, chunks: List[Dict[str, Any]]) -> None:
        self._replies.append(("sse", chunks))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        kind, payload = self._replies.pop(0)
        if kind == "json":
            return httpx.Response(200, json=payload)
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in payload)
        body += "data: [DONE]\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )


@pytest.fixture
def api() -> FakeMistralAPI:
    return FakeMistralAPI()


@pytest.fixture
def client(api: FakeMistralAPI) -> MistralChatClient:
    llm = MistralChatClient(api_key="test-key", model="mistral-large-latest")
    llm._client = Mistral(
        api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(api))
    )
    return llm


def completion(message: Dict[str, Any], finish_reason: str = "stop") -> Dict:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "model": "mistral-large-latest",
        "created": 1,
        "usage": {"prompt_tokens": 21, "completion_tokens": 7, "total_tokens": 28},
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", **message},
            }
        ],
    }


def chunk(delta: Dict[str, Any], finish_reason=None, usage=None) -> Dict:
    data = {
        "id": "cmpl-s",
        "object": "chat.completion.chunk",
        "model": "mistral-large-latest",
        "created": 1,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        data["usage"] = usage
    return data


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


class Person(BaseModel):
    name: str
    age: int


# --- tool calling ---------------------------------------------------------


def test_tools_are_sent_and_tool_calls_returned(client, api):
    api.reply_json(
        completion(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            finish_reason="tool_calls",
        )
    )

    resp = client.generate("Weather in Paris?", tools=[get_weather])

    sent = api.requests[0]
    assert sent["model"] == "mistral-large-latest"
    assert "stream" not in sent or sent["stream"] is False
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert sent["tools"][0]["function"]["parameters"]["required"] == ["city"]

    assert isinstance(resp, LLMChatResponse)
    msg = resp.get_message()
    assert resp.results[0].finish_reason == "tool_calls"
    assert msg.tool_calls[0].id == "call_1"
    assert msg.tool_calls[0].function.name == "get_weather"
    assert msg.tool_calls[0].function.arguments_dict == {"city": "Paris"}
    assert resp.metadata["provider"] == "mistral"
    assert resp.metadata["usage"]["prompt_tokens"] == 21
    assert resp.metadata["usage"]["completion_tokens"] == 7


def test_object_arguments_are_json_encoded(client, api):
    api.reply_json(
        completion(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Oslo"},
                        },
                    }
                ],
            },
            finish_reason="tool_calls",
        )
    )

    msg = client.generate("Weather?", tools=[get_weather]).get_message()

    assert isinstance(msg.tool_calls[0].function.arguments, str)
    assert json.loads(msg.tool_calls[0].function.arguments) == {"city": "Oslo"}
    assert msg.tool_calls[0].type == "function"


def test_tool_result_round_trip(client, api):
    """A tool call, its result, and the follow-up turn reach the API intact."""
    api.reply_json(
        completion(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Lima"}',
                        },
                    }
                ],
            },
            finish_reason="tool_calls",
        )
    )
    api.reply_json(completion({"content": "It is sunny in Lima."}))

    history: List[Any] = [UserMessage(content="Weather in Lima?")]
    first = client.generate(history, tools=[get_weather]).get_message()
    call = first.tool_calls[0]
    result = get_weather(**call.function.arguments_dict)
    history += [
        first,
        ToolMessage(tool_call_id=call.id, name=call.function.name, content=result),
    ]
    final = client.generate(history, tools=[get_weather]).get_message()

    assert final.content == "It is sunny in Lima."
    sent = api.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool"]
    assert sent[1]["tool_calls"][0]["id"] == "call_9"
    assert sent[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert sent[2]["tool_call_id"] == "call_9"
    assert sent[2]["content"] == "Sunny in Lima"


def test_tool_choice_is_forwarded(client, api):
    api.reply_json(completion({"content": "ok"}))
    client.generate("hi", tools=[get_weather], tool_choice="any")
    assert api.requests[0]["tool_choice"] == "any"


# --- content handling -----------------------------------------------------


def test_list_content_drops_thinking_and_joins_text(client, api):
    api.reply_json(
        completion(
            {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": [{"type": "text", "text": "internal"}],
                    },
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "there"},
                ]
            }
        )
    )
    assert client.generate("hi").get_message().content == "Hello there"


def test_content_to_text_edge_cases():
    assert content_to_text(None) is None
    assert content_to_text("") == ""
    assert content_to_text([]) == ""
    assert content_to_text(["a", {"type": "text", "text": "b"}]) == "ab"


# --- structured output ----------------------------------------------------


def test_structured_output_json_mode_uses_native_json_schema(client, api):
    api.reply_json(completion({"content": '{"name": "Ada", "age": 36}'}))

    person = client.generate("Who?", response_format=Person)

    assert person == Person(name="Ada", age=36)
    rf = api.requests[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Person"
    assert rf["json_schema"]["strict"] is True
    schema = rf["json_schema"]["schema"]
    assert set(schema["required"]) == {"name", "age"}
    assert schema["additionalProperties"] is False


def test_structured_output_function_call_mode(client, api):
    api.reply_json(
        completion(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_s",
                        "type": "function",
                        "function": {
                            "name": "Person",
                            "arguments": {"name": "Grace", "age": 85},
                        },
                    }
                ],
            },
            finish_reason="tool_calls",
        )
    )

    person = client.generate(
        "Who?", response_format=Person, structured_mode="function_call"
    )

    assert person == Person(name="Grace", age=85)
    sent = api.requests[0]
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "Person"}}
    assert sent["tools"][0]["function"]["name"] == "Person"


def test_structured_output_list_of_models(client, api):
    api.reply_json(
        completion(
            {
                "content": json.dumps(
                    {"objects": [{"name": "A", "age": 1}, {"name": "B", "age": 2}]}
                )
            }
        )
    )
    result = client.generate("Two people", response_format=List[Person])
    assert [p.name for p in result.objects] == ["A", "B"]


def test_invalid_structured_mode_rejected(client):
    with pytest.raises(ValueError, match="structured_mode"):
        client.generate("hi", response_format=Person, structured_mode="xml")


# --- streaming ------------------------------------------------------------


def test_streaming_text_follows_chunk_contract(client, api):
    api.reply_stream(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": "Hello"}),
            chunk({"content": [{"type": "text", "text": ", world"}]}),
            chunk(
                {"content": ""},
                finish_reason="stop",
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            ),
        ]
    )

    chunks = list(client.generate("Say hello", stream=True))

    assert api.requests[0]["stream"] is True
    StreamingComplianceHarness.assert_valid_stream_contract(chunks, "mistral")
    StreamingComplianceHarness.assert_reconstructs_text(chunks, "Hello, world")
    StreamingComplianceHarness.assert_snapshot_isolation(chunks)
    assert chunks[-1].metadata["usage"]["completion_tokens"] == 3
    assert chunks[-1].metadata.get("last_chunk") is True


def test_streaming_parallel_tool_calls_keep_separate_indexes(client, api):
    """Mistral streams each tool call whole with index 0; both must survive."""
    api.reply_stream(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Rome"},
                            },
                            "index": 0,
                        }
                    ]
                }
            ),
            chunk(
                {
                    "tool_calls": [
                        {
                            "id": "call_b",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Tokyo"}',
                            },
                            "index": 0,
                        }
                    ]
                },
                finish_reason="tool_calls",
            ),
        ]
    )

    chunks = list(client.generate("Two cities", tools=[get_weather], stream=True))

    StreamingComplianceHarness.assert_valid_stream_contract(chunks, "mistral")
    StreamingComplianceHarness.assert_reconstructs_tool_calls(
        chunks,
        [
            {"name": "get_weather", "arguments": {"city": "Rome"}},
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        ],
    )
    assert api.requests[0]["tools"][0]["function"]["name"] == "get_weather"


def test_streaming_skips_thinking_chunks(client, api):
    api.reply_stream(
        [
            chunk(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "hmm"}],
                        }
                    ],
                }
            ),
            chunk({"content": [{"type": "text", "text": "42"}]}, finish_reason="stop"),
        ]
    )
    chunks = list(client.generate("Answer?", stream=True))
    StreamingComplianceHarness.assert_reconstructs_text(chunks, "42")


def test_streaming_on_chunk_callback_lifecycle(api):
    from dapr_agents.llm.utils import StreamHandler

    def factory(on_chunk):
        api.reply_stream(
            [
                chunk({"role": "assistant", "content": "a"}),
                chunk({"content": "b"}, finish_reason="stop"),
            ]
        )
        sdk = Mistral(
            api_key="k", client=httpx.Client(transport=httpx.MockTransport(api))
        )
        raw = sdk.chat.stream(model="m", messages=[{"role": "user", "content": "x"}])
        return StreamHandler.process_stream(
            stream=raw, llm_provider="mistral", on_chunk=on_chunk
        )

    chunks = StreamingComplianceHarness.assert_on_chunk_callback_lifecycle(factory)
    StreamingComplianceHarness.assert_reconstructs_text(chunks, "ab")


def test_api_errors_are_wrapped(client, api):
    def boom(request):
        return httpx.Response(401, json={"message": "Unauthorized"})

    client._client = Mistral(
        api_key="k", client=httpx.Client(transport=httpx.MockTransport(boom))
    )
    with pytest.raises(ValueError, match="Mistral API error"):
        client.generate("hi")
