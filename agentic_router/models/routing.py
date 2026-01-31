"""Routing-related models."""

from pydantic import BaseModel, Field


class RoutingDecisionResult(BaseModel):
    """Result of routing decision."""

    query: str = Field(description="The original query")
    selected_models: list[str] = Field(description="List of capable models")
