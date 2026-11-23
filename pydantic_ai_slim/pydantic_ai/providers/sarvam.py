from __future__ import annotations as _annotations

import os

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer, merge_profile
from pydantic_ai.profiles._special_tokens import special_token_route_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Sarvam provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )

__all__ = ['SarvamProvider']


class SarvamProvider(_OpenAISDKProvider):
    """Provider for Sarvam Token Factory models.

    Sarvam Token Factory exposes an OpenAI-compatible Chat Completions API at
    `https://api.sarvam.ai/v2`.
    """

    @property
    def name(self) -> str:
        """Return the provider name."""
        return 'sarvam'

    @property
    def base_url(self) -> str:
        """Return the base URL."""
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        """Return the AsyncOpenAI client."""
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        """Get the model profile for a Sarvam Token Factory model.

        Sarvam serves its own `sarvam-105b` models alongside open-weight GLM,
        Gemma, and DeepSeek models. The profiles below encode the reasoning
        effort values and token-limit fields each model accepts.
        """
        model_name_lower = model_name.lower()

        if model_name_lower.startswith('sarvam-105b-conversations'):
            profile = OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                supports_thinking=False,
                openai_chat_supports_max_completion_tokens=False,
                supports_image_input=False,
            )
        elif model_name_lower.startswith('sarvam-105b'):
            profile = OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                supports_thinking=True,
                thinking_enabled_by_default=True,
                openai_reasoning_efforts=frozenset(('none', 'low', 'medium', 'high')),
                openai_chat_supports_max_completion_tokens=False,
                supports_image_input=False,
            )
        elif model_name_lower.startswith('glm5.3'):
            profile = OpenAIModelProfile(
                json_schema_transformer=InlineDefsJsonSchemaTransformer,
                supports_thinking=True,
                thinking_always_enabled=True,
                openai_reasoning_efforts=frozenset(('low', 'high')),
                supports_image_input=model_name_lower.startswith('glm5.3-flash'),
            )
        elif model_name_lower.startswith('gemma4'):
            profile = OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                supports_thinking=True,
                thinking_enabled_by_default=False,
                openai_reasoning_efforts=frozenset(('none', 'low', 'high', 'max')),
                openai_chat_supports_max_reasoning_effort=True,
                supports_image_input=True,
            )
        elif model_name_lower.startswith('deepseekv4.1-flash-1017'):
            profile = OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                supports_thinking=True,
                thinking_enabled_by_default=True,
                openai_reasoning_efforts=frozenset(('none', 'low', 'high', 'max')),
                openai_chat_supports_max_reasoning_effort=True,
                supports_image_input=True,
            )
        else:
            profile = OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)

        return merge_profile(profile, special_token_route_profile('sarvam', model_name))

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new Sarvam provider.

        Args:
            api_key: The API key to use for authentication. If not provided, the `SARVAM_API_KEY`
                environment variable will be used if available.
            base_url: The base URL to use for requests. If not provided, `https://api.sarvam.ai/v2`
                will be used, or `SARVAM_BASE_URL` if set.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key`, `base_url`
                and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for
                making HTTP requests.
        """
        api_key = api_key or os.getenv('SARVAM_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `SARVAM_API_KEY` environment variable or pass it via `SarvamProvider(api_key=...)` '
                'to use the Sarvam provider.'
            )

        default_headers = {'X-Sarvam-Integration': 'pydantic-ai'}

        assert openai_client is None or base_url is None, 'Cannot provide both `openai_client` and `base_url`'
        self._base_url = base_url or os.getenv('SARVAM_BASE_URL', 'https://api.sarvam.ai/v2')
        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(
                base_url=self.base_url,
                api_key=api_key,
                http_client=http_client,
                default_headers=default_headers,
            )
