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

import os
from unittest.mock import patch, MagicMock

from mistralai.client.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    UsageInfo,
)

from dapr_agents.llm.mistral.chat import MistralChatClient


def test_mistral_client_initialization():
    """Test default model and environment variable fallback."""
    client = MistralChatClient()
    assert client.model == "mistral-large-latest"

    with patch.dict(os.environ, {"MISTRAL_MODEL": "mistral-small"}, clear=False):
        client_env = MistralChatClient()
        assert client_env.model == "mistral-small"


@patch("dapr_agents.llm.mistral.client.Mistral")
def test_mistral_generate(mock_mistral_class):
    """Test basic text generation using mocked SDK."""
    mock_instance = MagicMock()
    mock_mistral_class.return_value = mock_instance

    mock_resp = ChatCompletionResponse(
        id="test-mistral-id",
        object="chat.completion",
        model="mistral-large-latest",
        created=0,
        usage=UsageInfo(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        choices=[
            ChatCompletionChoice(
                index=0,
                finish_reason="stop",
                message=AssistantMessage(content="Hello from Mistral"),
            )
        ],
    )

    mock_instance.chat.complete.return_value = mock_resp

    client = MistralChatClient(api_key="fake-key")
    response = client.generate("Say hello")

    assert response.get_message().content == "Hello from Mistral"
    assert response.metadata["provider"] == "mistral"
    assert response.metadata["usage"]["total_tokens"] == 7
    mock_instance.chat.complete.assert_called_once()


def test_mistral_from_prompty():
    """Test initializing MistralChatClient from a Prompty template."""
    prompty_content = """---
name: Mistral Test
model:
  api: chat
  configuration:
    type: mistral
    name: mistral-small
    endpoint: https://api.mistral.ai/v1
    api_key: dummy_key
  parameters:
    temperature: 0.5
    max_tokens: 100
---
system:
You are a helpful assistant.
"""
    client = MistralChatClient.from_prompty(prompty_content)

    assert client.model == "mistral-small"
    assert client.api_key == "dummy_key"
    assert client.endpoint == "https://api.mistral.ai/v1"

    assert client.prompty is not None
    assert client.prompty.model.parameters.temperature == 0.5
    assert client.prompty.model.parameters.max_tokens == 100
