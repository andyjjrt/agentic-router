import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentModelSettings(BaseSettings):
    """Settings for an individual agent's model configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )

    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class DifficultyAnalystSettings(AgentModelSettings):
    """Settings for the Difficulty Analyst agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        env_prefix="DIFFICULTY_ANALYST_",
    )


class DifficultyEvaluatorSettings(AgentModelSettings):
    """Settings for the Difficulty Evaluator agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        env_prefix="DIFFICULTY_EVALUATOR_",
    )


class RoutingDecisionMakerSettings(AgentModelSettings):
    """Settings for the Routing Decision Maker agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        env_prefix="ROUTING_DECISION_MAKER_",
    )


class EmbeddingSettings(AgentModelSettings):
    """Settings for embedding model configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        env_prefix="EMBEDDING_",
    )

    # Default embedding model name
    model_name: Optional[str] = "Qwen/Qwen3-Embedding-0.6B"


class Settings(BaseSettings):
    """Application settings managed by pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )

    # Data location settings
    data_dir: str = "data"

    # LLM Gateway settings (OpenAI-compatible API) - defaults for all agents
    llm_api_key: Optional[str] = None
    llm_base_url: str = "http://0.0.0.0:8000/v1"

    # Default model settings
    default_temperature: float = 0.5
    cache_seed: int = 42

    # Agent-specific settings (loaded from env with different prefixes)
    difficulty_analyst: DifficultyAnalystSettings = DifficultyAnalystSettings()
    difficulty_evaluator: DifficultyEvaluatorSettings = DifficultyEvaluatorSettings()
    routing_decision_maker: RoutingDecisionMakerSettings = (
        RoutingDecisionMakerSettings()
    )

    # Embedding settings (loaded from env with EMBEDDING_ prefix)
    embedding: EmbeddingSettings = EmbeddingSettings()

    def get_data_path(self, filename: str) -> str:
        """Get the full path for a file in the data directory."""
        return os.path.join(self.data_dir, filename)

    def get_agent_api_key(self, agent_settings: AgentModelSettings) -> str | None:
        """Get API key for an agent, falling back to default if not set."""
        return agent_settings.api_key or self.llm_api_key

    def get_agent_base_url(self, agent_settings: AgentModelSettings) -> str:
        """Get base URL for an agent, falling back to default if not set."""
        return agent_settings.base_url or self.llm_base_url

    def get_embedding_model_name(self) -> str | None:
        """Get the embedding model name."""
        return self.embedding.model_name

    def get_embedding_api_key(self) -> str | None:
        """Get API key for embedding, falling back to default if not set."""
        return self.embedding.api_key or self.llm_api_key

    def get_embedding_base_url(self) -> str:
        """Get base URL for embedding, falling back to default if not set."""
        return self.embedding.base_url or self.llm_base_url


# Singleton instance
settings = Settings()
