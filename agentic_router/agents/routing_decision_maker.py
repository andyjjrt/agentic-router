"""Routing decision maker agent using raw OpenAI client."""

from textwrap import dedent
from typing import Any

import numpy as np

from agentic_router.clients import get_model_name, get_openai_client
from agentic_router.models.difficulty import DifficultyAnalysisResult
from agentic_router.models.routing import RoutingDecisionResult

ROUTING_DECISION_MAKER_PROMPT = dedent("""
    You are an intelligent routing decision maker for a multi-agent system.
    Your task is to identify all AI models that can correctly answer the given query.
    You will be provided with the user's query, a difficulty analysis of that query,
    and several retrieved examples of past model responses.
    Based on the provided information, especially the past responses, identify all
    models that you believe can successfully answer the query.
    Your final answer must be a comma-separated list of the names of the chosen
    models (e.g., 'Model-A, Model-C').
""").strip()

ROUTING_DECISION_USER_PROMPT = dedent("""
    **Query to Route:**
    "{query}"

    **Difficulty Analysis of the Query:**
    {difficulty_analysis}

    ---

    **Retrieved Similar Examples for Context:**

    **Relevant Past Model Response Analyses (per model):**
    {formatted_responses}

    ---

    **Decision Task:**
    Based on the difficulty analyses and the models' demonstrated capabilities from past
    responses, which models can correctly answer the given query?

    The available models are: {available_models_str}

    Your final answer must be a comma-separated list of capable models from the list
    above (e.g., Model-A, Model-C).
""").strip()


class RoutingDecisionMakerAgent:
    """Agent for making routing decisions using raw OpenAI client."""

    def __init__(self, model_name: str | None = None):
        """Initialize the routing decision maker agent.

        Args:
            model_name: The model name to use (overrides settings)
        """
        self.model_name = get_model_name(
            model_name=model_name,
            agent_type="routing_decision_maker",
        )
        self.client = get_openai_client(agent_type="routing_decision_maker")

    def _create_anonymized_mapping(
        self, llm_list: dict[str, float]
    ) -> tuple[dict[str, str], dict[str, str], list[str]]:
        """Create anonymized model names to mitigate bias."""
        llm_names = list(llm_list.keys())
        anonymized_names = [f"Model-{chr(65 + i)}" for i in range(len(llm_names))]
        anonymized_map = dict(zip(llm_names, anonymized_names))
        reverse_anonymized_map = dict(zip(anonymized_names, llm_names))
        return anonymized_map, reverse_anonymized_map, anonymized_names

    def _format_responses(
        self, relevant_responses: dict[str, list[Any]], anonymized_map: dict[str, str]
    ) -> str:
        """Format relevant responses for the prompt."""
        formatted_responses = ""
        for model_name, docs in relevant_responses.items():
            anonymized_name = anonymized_map.get(model_name, model_name)
            formatted_responses += (
                f"\n\n**Retrieved Responses for {anonymized_name}:**\n"
            )
            if docs:
                formatted_responses += "\n".join(
                    [
                        f"  - {doc.page_content}"
                        if hasattr(doc, "page_content")
                        else f"  - {doc}"
                        for doc in docs
                    ]
                )
            else:
                formatted_responses += "  No relevant responses found for this model."
        return formatted_responses

    def _parse_decision(
        self, decision_str: str, reverse_anonymized_map: dict[str, str]
    ) -> list[str]:
        """Parse decision string and map back to real model names."""
        decision_str = decision_str.replace("TERMINATE", "").strip()
        potential_candidates = [name.strip() for name in decision_str.split(",")]

        candidate_llms = []
        for name in potential_candidates:
            if name in reverse_anonymized_map:
                candidate_llms.append(reverse_anonymized_map[name])

        return candidate_llms

    async def decide(
        self,
        query: str,
        difficulty_analysis: DifficultyAnalysisResult,
        relevant_responses: dict[str, list[Any]],
        llm_list: dict[str, float],
    ) -> RoutingDecisionResult:
        """Make routing decision.

        Args:
            query: The query to route
            difficulty_analysis: DifficultyAnalysisResult containing difficulty analysis
            relevant_responses: Dictionary of model responses
            llm_list: Dictionary mapping model names to costs

        Returns:
            RoutingDecisionResult containing selected models
        """
        anonymized_map, reverse_anonymized_map, anonymized_names = (
            self._create_anonymized_mapping(llm_list)
        )

        formatted_responses = self._format_responses(relevant_responses, anonymized_map)
        available_models_str = ", ".join(anonymized_names)

        user_prompt = ROUTING_DECISION_USER_PROMPT.format(
            query=query,
            difficulty_analysis=difficulty_analysis.difficulty,
            formatted_responses=formatted_responses
            if formatted_responses
            else "No relevant model responses found.",
            available_models_str=available_models_str,
        )

        llm_names = list(llm_list.keys())

        for loop_cnt in range(10):
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": ROUTING_DECISION_MAKER_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            result_output = response.choices[0].message.content or ""
            candidate_llms = self._parse_decision(result_output, reverse_anonymized_map)

            if candidate_llms:
                return RoutingDecisionResult(
                    query=query, selected_models=candidate_llms
                )
            else:
                print(
                    f"Attempt {loop_cnt + 1} Invalid decision '{result_output}'. "
                    "No valid models found."
                )

        # Fallback to random choice
        return RoutingDecisionResult(
            query=query, selected_models=[np.random.choice(llm_names)]
        )
