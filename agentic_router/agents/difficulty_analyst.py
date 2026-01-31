"""Difficulty analyst agent using raw OpenAI client."""

from textwrap import dedent
from typing import Any

from agentic_router.clients import get_model_name, get_openai_client
from agentic_router.models.difficulty import DifficultyAnalysisResult

DIFFICULTY_ANALYST_PROMPT = dedent("""
    Your role as an assistant is to analyze the difficulty of a given query for a large
    language model through a systematic long thinking process analysis. You will be
    provided with the user query and some context from past similar analyses. You need
    to evaluate the incoming query on several key dimensions: reasoning, comprehension,
    instruction following, agentic, knowledge retrieval, coding, multilingual. For each
    dimension, elaborate on the specific challenges and required capabilities. Now, try
    to analyze the following query through the above guidelines:
""").strip()

DIFFICULTY_ANALYSIS_USER_PROMPT = dedent("""
    **Context from similar past analyses:**
    {formatted_analyses}

    ---

    **Query to Analyze:**
    "{query}"
""").strip()


class DifficultyAnalystAgent:
    """Agent for analyzing query difficulty using raw OpenAI client."""

    def __init__(self, model_name: str | None = None):
        """Initialize the difficulty analyst agent.

        Args:
            model_name: The model name to use (overrides settings)
        """
        self.model_name = get_model_name(
            model_name=model_name,
            agent_type="difficulty_analyst",
        )
        self.client = get_openai_client(agent_type="difficulty_analyst")

    def _format_context(self, relevant_analyses: list[Any]) -> str:
        """Format relevant analyses into a context string."""
        if not relevant_analyses:
            return "No relevant past analyses found."

        formatted = "\n".join(
            [
                f"- {doc.page_content}" if hasattr(doc, "page_content") else f"- {doc}"
                for doc in relevant_analyses
            ]
        )
        return formatted

    async def analyze(
        self, query: str, relevant_analyses: list[Any]
    ) -> DifficultyAnalysisResult:
        """Analyze query difficulty.

        Args:
            query: The query to analyze
            relevant_analyses: List of relevant past analyses for context

        Returns:
            DifficultyAnalysisResult containing query, difficulty, and analysis
        """
        formatted_analyses = self._format_context(relevant_analyses)
        user_prompt = DIFFICULTY_ANALYSIS_USER_PROMPT.format(
            formatted_analyses=formatted_analyses,
            query=query,
        )

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": DIFFICULTY_ANALYST_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        result_output = response.choices[0].message.content or ""
        difficulty_assessment = result_output.strip().lower()

        return DifficultyAnalysisResult(
            query=query,
            difficulty=difficulty_assessment,
            analysis=f"The query is classified as '{difficulty_assessment}' based on LLM analysis.",
        )
