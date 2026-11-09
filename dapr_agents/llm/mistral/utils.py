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

"""Translate Mistral SDK responses into dapr-agents' unified LLM types.

Mistral's chat API is close to OpenAI's choice/delta shape, with three
differences that matter here:

- ``content`` may be a list of typed chunks (``text``, ``thinking``, ...)
  instead of a plain string, both in full responses and in stream deltas.
- Tool-call ``arguments`` may arrive as a JSON object rather than a string.
- Streaming yields ``CompletionEvent`` wrappers whose ``data`` is the chunk,
  and each streamed tool call arrives whole (id, name and arguments together)
  with ``index`` defaulting to 0, even when several calls are made in a turn.

The helpers below normalize those differences and then reuse the shared
choice-delta stream processor so Mistral streams follow the same chunk
contract as every other provider.
"""

import dataclasses
import json
import logging
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from dapr_agents.llm.utils.stream import process_choice_delta_stream
from dapr_agents.types.message import (
    AssistantMessage,
    FunctionCall,
    LLMChatCandidate,
    LLMChatResponse,
    LLMChatResponseChunk,
    ToolCall,
)

logger = logging.getLogger(__name__)


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Serialize a Mistral SDK model (or plain mapping) into a dict."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return dict(vars(obj))


def content_to_text(content: Any) -> Optional[str]:
    """Flatten Mistral message content into plain text.

    Mistral returns either a string or a list of typed content chunks. Text
    chunks are concatenated in order; reasoning (``thinking``) and other
    non-text chunks are skipped so they never leak into the assistant reply.

    Returns ``None`` when there is no content at all, so tool-call-only
    messages keep ``content=None`` like other providers.
    """
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
                continue
            data = _to_dict(chunk)
            if data.get("type", "text") == "text" and data.get("text") is not None:
                parts.append(str(data["text"]))
        return "".join(parts)
    return str(content)


def arguments_to_json(arguments: Any) -> str:
    """Return tool-call arguments as a JSON string.

    Mistral may send arguments as an object; the rest of dapr-agents (tool
    execution, the stream accumulator, observability) expects the OpenAI
    convention of a JSON-encoded string.
    """
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def _normalize_tool_calls(raw_tool_calls: Optional[Iterable[Any]]) -> List[Dict]:
    calls: List[Dict[str, Any]] = []
    for tc in raw_tool_calls or []:
        data = _to_dict(tc)
        fn = _to_dict(data.get("function"))
        calls.append(
            {
                "index": data.get("index"),
                "id": data.get("id"),
                "type": data.get("type") or "function",
                "function": {
                    "name": fn.get("name"),
                    "arguments": arguments_to_json(fn.get("arguments")),
                },
            }
        )
    return calls


def to_llm_chat_response(response: Any) -> LLMChatResponse:
    """Convert a Mistral ``ChatCompletionResponse`` into an ``LLMChatResponse``.

    Keeps tool calls (with arguments as JSON strings), the finish reason and
    usage, so agents can execute tools and observability can record tokens.
    """
    resp = _to_dict(response)

    candidates: List[LLMChatCandidate] = []
    for choice in resp.get("choices") or []:
        choice = _to_dict(choice)
        msg = _to_dict(choice.get("message"))
        if not msg:
            logger.warning("Skipping Mistral choice without a message: %s", choice)
            continue

        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = []
            for tc in _normalize_tool_calls(msg["tool_calls"]):
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        type=tc["type"],
                        function=FunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                )

        candidates.append(
            LLMChatCandidate(
                message=AssistantMessage(
                    content=content_to_text(msg.get("content")),
                    tool_calls=tool_calls,
                ),
                finish_reason=choice.get("finish_reason"),
                index=choice.get("index"),
            )
        )

    usage = resp.get("usage")
    metadata: Dict[str, Any] = {
        "provider": "mistral",
        "id": resp.get("id"),
        "model": resp.get("model"),
        "object": resp.get("object"),
        "created": resp.get("created"),
        "usage": _to_dict(usage) if usage is not None else None,
    }
    return LLMChatResponse(results=candidates, metadata=metadata)


def iter_mistral_stream_packets(raw_stream: Iterable[Any]) -> Iterator[Dict[str, Any]]:
    """Unwrap Mistral ``CompletionEvent``s into OpenAI-style chunk dicts.

    - ``event.data`` is unwrapped into a plain dict.
    - List content in deltas is flattened to text.
    - Tool-call arguments are JSON-encoded strings.
    - Each distinct tool-call id gets its own stable index. Mistral streams
      every tool call whole but leaves ``index`` at 0, which would otherwise
      make the accumulator merge parallel calls into one.
    """
    index_by_id: Dict[str, int] = {}

    for event in raw_stream:
        packet = getattr(event, "data", event)
        data = _to_dict(packet)

        choices = []
        for choice in data.get("choices") or []:
            choice = _to_dict(choice)
            delta = _to_dict(choice.get("delta"))

            if "content" in delta:
                delta["content"] = content_to_text(delta.get("content"))

            if delta.get("tool_calls"):
                tool_calls = _normalize_tool_calls(delta["tool_calls"])
                for tc in tool_calls:
                    call_id = tc.get("id")
                    if call_id:
                        if call_id not in index_by_id:
                            index_by_id[call_id] = len(index_by_id)
                        tc["index"] = index_by_id[call_id]
                    elif tc.get("index") is None:
                        tc["index"] = max(len(index_by_id) - 1, 0)
                delta["tool_calls"] = tool_calls

            choice["delta"] = delta
            choices.append(choice)

        data["choices"] = choices
        if data.get("usage") is not None:
            data["usage"] = _to_dict(data["usage"])
        yield data


def process_mistral_stream(
    raw_stream: Iterable[Any],
    *,
    enrich_metadata: Optional[Dict[str, Any]] = None,
    on_chunk: Optional[Callable] = None,
) -> Iterator[LLMChatResponseChunk]:
    """Normalize a Mistral chat stream into ``LLMChatResponseChunk``s."""
    yield from process_choice_delta_stream(
        raw_stream=iter_mistral_stream_packets(raw_stream),
        enrich_metadata=enrich_metadata,
        on_chunk=on_chunk,
    )


__all__ = [
    "arguments_to_json",
    "content_to_text",
    "iter_mistral_stream_packets",
    "process_mistral_stream",
    "to_llm_chat_response",
]
