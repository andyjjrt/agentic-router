"""Difficulty-related models."""

from pydantic import BaseModel, Field


class DifficultyAnalysisResult(BaseModel):
    """Result of difficulty analysis."""

    query: str = Field(description="The original query")
    difficulty: str = Field(description="The difficulty assessment")
    analysis: str = Field(description="The analysis summary")


class DifficultyEvaluationResult(BaseModel):
    """Result of difficulty evaluation for a Q&A pair."""

    question: str = Field(description="The original question")
    answer: str = Field(description="The model's answer")
    correctness: str = Field(description="Whether the answer was correct")
    analysis: str = Field(description="Raw analysis string")
