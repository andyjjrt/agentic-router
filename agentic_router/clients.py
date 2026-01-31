"""OpenAI client configuration for agents and embeddings."""

from typing import Literal

from openai import AsyncOpenAI

from agentic_router.core.config import AgentModelSettings, settings

AgentType = Literal[
    "difficulty_analyst", "difficulty_evaluator", "routing_decision_maker"
]


def get_agent_settings(agent_type: AgentType) -> AgentModelSettings:
    """Get the settings for a specific agent type.

    Args:
        agent_type: The type of agent

    Returns:
        The agent-specific settings
    """
    return getattr(settings, agent_type)


def get_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    agent_type: AgentType | None = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client with the given configuration.

    Args:
        api_key: The API key to use (overrides agent and default settings)
        base_url: The base URL to use (overrides agent and default settings)
        agent_type: The type of agent to get settings for

    Returns:
        An AsyncOpenAI client instance
    """
    resolved_api_key = api_key
    resolved_base_url = base_url

    if agent_type:
        agent_settings = get_agent_settings(agent_type)
        resolved_api_key = api_key or settings.get_agent_api_key(agent_settings)
        resolved_base_url = base_url or settings.get_agent_base_url(agent_settings)

    # Fall back to global settings if still not resolved
    resolved_api_key = resolved_api_key or settings.llm_api_key
    resolved_base_url = resolved_base_url or settings.llm_base_url

    return AsyncOpenAI(
        api_key=resolved_api_key or "NULL",
        base_url=resolved_base_url,
    )


def get_embedding_client(
    api_key: str | None = settings.get_embedding_api_key(),
    base_url: str = settings.get_embedding_base_url(),
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client for embeddings.

    Args:
        api_key: API key for the API (default: from settings)
        base_url: Base URL for the API (default: from settings)

    Returns:
        An AsyncOpenAI client instance
    """
    return AsyncOpenAI(
        api_key=api_key or "NULL",
        base_url=base_url,
    )


def get_model_name(
    model_name: str | None = None,
    agent_type: AgentType | None = None,
) -> str:
    """Get the model name for an agent.

    Args:
        model_name: The model name to use (overrides agent settings)
        agent_type: The type of agent to get settings for

    Returns:
        The resolved model name

    Raises:
        ValueError: If no model name can be resolved
    """
    resolved_model_name = model_name
    if agent_type and not model_name:
        agent_settings = get_agent_settings(agent_type)
        resolved_model_name = agent_settings.model_name

    if not resolved_model_name:
        raise ValueError(
            "model_name must be provided either directly or via agent settings"
        )

    return resolved_model_name
