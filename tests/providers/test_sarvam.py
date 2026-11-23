import httpx2
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models import infer_model
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers import infer_provider
    from pydantic_ai.providers.sarvam import SarvamProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_sarvam_provider_init():
    provider = SarvamProvider(api_key='test-key')

    assert provider.name == 'sarvam'
    assert provider.base_url == 'https://api.sarvam.ai/v2'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'test-key'
    assert provider.client.default_headers['X-Sarvam-Integration'] == 'pydantic-ai'


def test_sarvam_provider_env_key(env: TestEnv):
    env.set('SARVAM_API_KEY', 'env-key')
    provider = SarvamProvider()

    assert provider.client.api_key == 'env-key'


def test_sarvam_provider_missing_key(env: TestEnv):
    env.remove('SARVAM_API_KEY')

    with pytest.raises(UserError, match='Set the `SARVAM_API_KEY`'):
        SarvamProvider()


def test_infer_provider(env: TestEnv):
    env.set('SARVAM_API_KEY', 'key')

    provider = infer_provider('sarvam')

    assert isinstance(provider, SarvamProvider)


def test_infer_model(env: TestEnv):
    env.set('SARVAM_API_KEY', 'key')

    model = infer_model('sarvam:sarvam-105b')

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == 'sarvam-105b'
    assert isinstance(model.provider, SarvamProvider)


def test_sarvam_105b_profile():
    provider = SarvamProvider(api_key='key')
    profile = provider.model_profile('sarvam-105b')

    assert profile is not None
    assert profile.get('supports_thinking') is True
    assert profile.get('thinking_enabled_by_default') is True
    assert profile.get('openai_reasoning_efforts') == frozenset(('none', 'low', 'medium', 'high'))
    assert profile.get('openai_chat_supports_max_completion_tokens') is False
    assert profile.get('supports_image_input') is False
    assert profile.get('json_schema_transformer') is OpenAIJsonSchemaTransformer
    assert profile.get('may_tokenize_special_tokens_in_content') is False


def test_sarvam_105b_conversations_profile():
    provider = SarvamProvider(api_key='key')
    profile = provider.model_profile('sarvam-105b-conversations')

    assert profile is not None
    assert profile.get('supports_thinking') is False
    assert profile.get('openai_chat_supports_max_completion_tokens') is False
    assert profile.get('supports_image_input') is False
    assert profile.get('may_tokenize_special_tokens_in_content') is False


def test_glm_profile():
    provider = SarvamProvider(api_key='key')
    profile = provider.model_profile('glm5.3')

    assert profile is not None
    assert profile.get('supports_thinking') is True
    assert profile.get('thinking_always_enabled') is True
    assert profile.get('openai_reasoning_efforts') == frozenset(('low', 'high'))
    assert profile.get('supports_image_input') is False
    assert profile.get('json_schema_transformer') is InlineDefsJsonSchemaTransformer
    assert profile.get('may_tokenize_special_tokens_in_content') is True


def test_glm_flash_profile():
    provider = SarvamProvider(api_key='key')
    profile = provider.model_profile('glm5.3-flash')

    assert profile is not None
    assert profile.get('supports_image_input') is True
    assert profile.get('openai_reasoning_efforts') == frozenset(('low', 'high'))


def test_gemma_profile():
    provider = SarvamProvider(api_key='key')
    profile = provider.model_profile('gemma4')

    assert profile is not None
    assert profile.get('supports_thinking') is True
    assert profile.get('thinking_enabled_by_default') is False
    assert profile.get('openai_reasoning_efforts') == frozenset(('none', 'low', 'high', 'max'))
    assert profile.get('openai_chat_supports_max_reasoning_effort') is True
    assert profile.get('supports_image_input') is True


def test_deepseek_profile():
    provider = SarvamProvider(api_key='key')
    profile = provider.model_profile('deepseekv4.1-flash-1017')

    assert profile is not None
    assert profile.get('supports_thinking') is True
    assert profile.get('thinking_enabled_by_default') is True
    assert profile.get('openai_reasoning_efforts') == frozenset(('none', 'low', 'high', 'max'))
    assert profile.get('openai_chat_supports_max_reasoning_effort') is True
    assert profile.get('supports_image_input') is True


def test_unknown_model_profile():
    provider = SarvamProvider(api_key='key')
    profile = provider.model_profile('unknown-model')

    assert profile is not None
    assert profile.get('json_schema_transformer') is OpenAIJsonSchemaTransformer
    assert 'supports_thinking' not in profile


def test_sarvam_provider_with_openai_client():
    client = openai.AsyncOpenAI(api_key='foo')
    provider = SarvamProvider(openai_client=client)

    assert provider.client is client


def test_sarvam_provider_with_http_client():
    http_client = httpx2.AsyncClient()
    provider = SarvamProvider(api_key='foo', http_client=http_client)

    assert provider.client.api_key == 'foo'


def test_sarvam_provider_custom_base_url():
    provider = SarvamProvider(api_key='test-key', base_url='https://custom.endpoint.com/v2')

    assert provider.base_url == 'https://custom.endpoint.com/v2'
    assert str(provider.client.base_url).rstrip('/') == 'https://custom.endpoint.com/v2'


def test_sarvam_provider_env_base_url(env: TestEnv):
    env.set('SARVAM_API_KEY', 'key')
    env.set('SARVAM_BASE_URL', 'https://env.endpoint.com/v2')
    provider = SarvamProvider()

    assert provider.base_url == 'https://env.endpoint.com/v2'
