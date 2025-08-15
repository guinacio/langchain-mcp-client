"""
LLM provider configurations and creation utilities.

This module handles the creation and configuration of different
language model providers (OpenAI, Anthropic, Google, Ollama) with
support for system prompts and advanced parameters.
"""

from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from langchain_core.output_parsers import JsonOutputParser


# OpenAI reasoning models that require special parameter handling
OPENAI_REASONING_MODELS = {
    "o1-mini", "o1-preview", "o1", 
    "o3-mini", "o3", 
    "o4-mini", "o4-mini-preview", "o4", "o4-preview"
}

# Model configurations for each provider
LLM_PROVIDERS = {
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini", "o4-mini", "Other"],
        "default_model": "gpt-4o",
        "requires_api_key": True,
        "description": "OpenAI's GPT models",
        "supports_system_prompt": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "temperature_range": (0.0, 2.0),
        "default_max_tokens": 4096,
        "max_tokens_range": (1, 16384),
        "supports_timeout": True,
        "default_timeout": 600.0
    },
    "Anthropic": {
        "models": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307", "Other"],
        "default_model": "claude-3-5-sonnet-20240620",
        "requires_api_key": True,
        "description": "Anthropic's Claude models",
        "supports_system_prompt": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "temperature_range": (0.0, 1.0),
        "default_max_tokens": 4096,
        "max_tokens_range": (1, 8192),
        "supports_timeout": True,
        "default_timeout": 600.0
    },
    "Google": {
        "models": ["gemini-2.0-flash-001", "gemini-2.5-pro-exp-03-25", "Other"],
        "default_model": "gemini-2.0-flash-001",
        "requires_api_key": True,
        "description": "Google's Gemini models",
        "supports_system_prompt": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "temperature_range": (0.0, 2.0),
        "default_max_tokens": 8192,
        "max_tokens_range": (1, 32768),
        "supports_timeout": True,
        "default_timeout": 600.0,
        "convert_system_to_human": True  # Gemini-specific setting
    },
    "Ollama": {
        "models": ["granite3.3:8b", "qwen3:4b", "Other"],
        "default_model": "granite3.3:8b",
        "requires_api_key": False,
        "description": "Local Ollama models",
        "supports_system_prompt": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "temperature_range": (0.0, 2.0),
        "default_max_tokens": 4096,
        "max_tokens_range": (1, 32768),
        "supports_timeout": True,
        "default_timeout": 600.0
    }
}

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. You have access to various tools and can help users with a wide range of tasks.

When using tools:
- Always explain what you're doing before calling a tool
- Provide clear summaries of tool results
- If a tool fails, explain what went wrong and suggest alternatives

Be conversational, helpful, and thorough in your responses."""


def is_openai_reasoning_model(model_name: str) -> bool:
    """Check if a model is an OpenAI reasoning model that requires special parameter handling."""
    if not model_name:
        return False
    
    # Check exact matches first
    if model_name in OPENAI_REASONING_MODELS:
        return True
    
    # Check if the model name contains reasoning model indicators
    reasoning_indicators = ["o1-", "o3-", "o4-"]
    return any(indicator in model_name.lower() for indicator in reasoning_indicators)


def supports_streaming_for_reasoning_model(model_name: str) -> bool:
    """Check if a reasoning model supports streaming."""
    # o1 series doesn't support streaming, but o3/o4 series do
    o1_models = ["o1", "o1-mini", "o1-preview"]
    
    # Check exact matches for o1 models
    if model_name in o1_models:
        return False
    
    # Check if model name contains o1 indicators
    if "o1-" in model_name.lower():
        return False
    
    # o3 and o4 series support streaming
    return True


def get_available_providers() -> List[str]:
    """Get list of available LLM providers."""
    return list(LLM_PROVIDERS.keys())


def get_provider_models(provider: str) -> List[str]:
    """Get available models for a specific provider."""
    return LLM_PROVIDERS.get(provider, {}).get("models", [])


def get_default_model(provider: str) -> str:
    """Get the default model for a specific provider."""
    return LLM_PROVIDERS.get(provider, {}).get("default_model", "")


def requires_api_key(provider: str) -> bool:
    """Check if a provider requires an API key."""
    return LLM_PROVIDERS.get(provider, {}).get("requires_api_key", True)


def get_provider_description(provider: str) -> str:
    """Get description for a specific provider."""
    return LLM_PROVIDERS.get(provider, {}).get("description", "")


def supports_system_prompt(provider: str) -> bool:
    """Check if a provider supports system prompts."""
    return LLM_PROVIDERS.get(provider, {}).get("supports_system_prompt", False)


def get_default_temperature(provider: str) -> float:
    """Get default temperature for a provider."""
    return LLM_PROVIDERS.get(provider, {}).get("default_temperature", 0.7)


def get_temperature_range(provider: str) -> Tuple[float, float]:
    """Get temperature range for a provider."""
    return LLM_PROVIDERS.get(provider, {}).get("temperature_range", (0.0, 1.0))


def get_default_max_tokens(provider: str) -> int:
    """Get default max tokens for a provider."""
    return LLM_PROVIDERS.get(provider, {}).get("default_max_tokens", 4096)


def get_max_tokens_range(provider: str) -> Tuple[int, int]:
    """Get max tokens range for a provider."""
    return LLM_PROVIDERS.get(provider, {}).get("max_tokens_range", (1, 8192))


def get_default_timeout(provider: str) -> float:
    """Get default timeout for a provider."""
    return LLM_PROVIDERS.get(provider, {}).get("default_timeout", 600.0)


def supports_streaming(provider: str) -> bool:
    """Check if a provider supports streaming."""
    return LLM_PROVIDERS.get(provider, {}).get("supports_streaming", False)


def create_llm_model(
    llm_provider: str, 
    api_key: str, 
    model_name: str, 
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    system_prompt: Optional[str] = None,
    streaming: bool = True,
    ollama_url: Optional[str] = None
):
    """Create a language model based on the selected provider with advanced configuration."""
    # Check if this is an OpenAI reasoning model
    is_reasoning_model = (llm_provider == "OpenAI" and is_openai_reasoning_model(model_name))
    
    # Common parameters
    common_params = {
        "model": model_name,
    }
    
    # Handle temperature parameter
    # OpenAI reasoning models (o1/o3/o4 series) do NOT support temperature.
    # Explicitly set to None to override library defaults and avoid sending it.
    if is_reasoning_model:
        common_params["temperature"] = 1.0
    else:
        common_params["temperature"] = temperature
    
    # Add streaming if supported
    if streaming and supports_streaming(llm_provider):
        # For reasoning models, check if the specific model supports streaming
        if is_reasoning_model:
            # o1 series doesn't support streaming; o3/o4 do
            if supports_streaming_for_reasoning_model(model_name):
                common_params["streaming"] = True
        else:
            # Regular models - use streaming as normal
            common_params["streaming"] = True
    
    # Handle max_tokens vs max_completion_tokens
    if max_tokens:
        if is_reasoning_model:
            # Reasoning models require max_completion_tokens
            common_params["max_completion_tokens"] = max_tokens
        else:
            # Regular models use max_tokens
            common_params["max_tokens"] = max_tokens
    
    # Add timeout if specified and supported
    if timeout:
        common_params["timeout"] = timeout
    
    # Add reasoning-effort style parameter for OpenAI reasoning models only
    if is_reasoning_model and llm_provider == "OpenAI":
        common_params["reasoning_effort"] = "medium"
    
    # Create the base model
    if llm_provider == "OpenAI":
        llm = ChatOpenAI(
            openai_api_key=api_key,
            **common_params
        )
    elif llm_provider == "Anthropic":
        llm = ChatAnthropic(
            anthropic_api_key=api_key,
            **common_params
        )
    elif llm_provider == "Google":
        # Google-specific parameters
        google_params = common_params.copy()
        if system_prompt:
            # Gemini converts system messages to human messages
            google_params["convert_system_message_to_human"] = True
        
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            max_retries=2,
            **google_params
        )
    elif llm_provider == "Ollama":
        # Add Ollama-specific parameters
        ollama_params = common_params.copy()
        if ollama_url:
            ollama_params["base_url"] = ollama_url
        
        llm = ChatOllama(**ollama_params)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    # Store system prompt in the LLM object for later use by the agent
    # This is a workaround since LangGraph agents handle system prompts differently
    if system_prompt and supports_system_prompt(llm_provider):
        llm._system_prompt = system_prompt
    
    # Store if this is a reasoning model for reference
    llm._is_reasoning_model = is_reasoning_model
    
    return llm


def validate_provider_config(provider: str, api_key: str = None, model_name: str = None) -> Tuple[bool, str]:
    """
    Validate provider configuration.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if provider not in LLM_PROVIDERS:
        return False, f"Unknown provider: {provider}"
    
    provider_config = LLM_PROVIDERS[provider]
    
    # Check API key requirement
    if provider_config["requires_api_key"] and not api_key:
        return False, f"{provider} requires an API key"
    
    # Check model availability
    if model_name and model_name not in provider_config["models"] and model_name != "Other":
        return False, f"Model {model_name} not available for {provider}"
    
    return True, ""


def get_provider_config_info(provider: str) -> Dict:
    """Get complete configuration information for a provider."""
    if provider not in LLM_PROVIDERS:
        return {}
    
    config = LLM_PROVIDERS[provider].copy()
    config["name"] = provider
    return config


def validate_model_parameters(
    provider: str,
    temperature: float,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    model_name: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validate model parameters for a specific provider.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if provider not in LLM_PROVIDERS:
        return False, f"Unknown provider: {provider}"
    
    config = LLM_PROVIDERS[provider]
    
    # Check if this is a reasoning model
    is_reasoning = (provider == "OpenAI" and model_name and is_openai_reasoning_model(model_name))
    
    # Check for unsupported o1 series models
    if provider == "OpenAI" and model_name:
        if model_name in ["o1", "o1-mini", "o1-preview"] or "o1-" in model_name.lower():
            return False, f"o1 series models ({model_name}) are not supported due to unique API requirements. Please use o3-mini, o4-mini, or regular GPT models instead."
    
    # Validate temperature (reasoning models don't support it)
    if not is_reasoning:
        temp_min, temp_max = config["temperature_range"]
        if not (temp_min <= temperature <= temp_max):
            return False, f"Temperature must be between {temp_min} and {temp_max} for {provider}"
    else:
        # Reasoning models don't support temperature
        if temperature != 0.7:  # Only warn if user explicitly set a different temperature
            return False, f"Temperature is not supported for reasoning model {model_name}. Reasoning models use fixed temperature."
    
    # Validate max_tokens
    if max_tokens:
        token_min, token_max = config["max_tokens_range"]
        if not (token_min <= max_tokens <= token_max):
            return False, f"Max tokens must be between {token_min} and {token_max} for {provider}"
    
    # Validate timeout
    if timeout and timeout <= 0:
        return False, "Timeout must be greater than 0"
    
    return True, "" 