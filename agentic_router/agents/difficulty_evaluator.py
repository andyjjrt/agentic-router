"""Difficulty evaluator agent using raw OpenAI client."""

from textwrap import dedent
from typing import Any

from agentic_router.clients import get_model_name, get_openai_client
from agentic_router.models.difficulty import DifficultyEvaluationResult

DIFFICULTY_EVALUATOR_PROMPT = dedent("""
    Your role as an assistant involves thoroughly exploring the difficulty of a given query for a large language model through a systematic long thinking process analysis.
    Your analysis can systematically evaluate the incoming query including but not limited to several key dimensions: reasoning, comprehension, instruction following, agentic, knowledge retrieval, coding, multilingual.
    For each dimension, elaborate on the specific challenges and required capabilities, incorporating the provided keywords into your explanation.
    Please structure your response into Summary.
    In the Summary section, based on the analysis, explorations, and reflections from the Think section, systematically present the summary you think for the query difficulty. The summary should remain a clear, concise expression style and detail necessary difficulty description to reach the conclusion, formatted as follows: <summary> {final formatted, precise, and clear summary} </summary> Now, try to analyze the following query through the above guidelines:
""").strip()

DIFFICULTY_EVALUATOR_USER_PROMPT = dedent("""
    **Question:**
    {question}

    ---

    **Model Answer:**
    {answer}

    ---

    **Correctness:** {correctness}

    ---

    Please analyze the difficulty of this question and provide a detailed assessment
    that can be used for routing similar queries to appropriate models in the future.
""").strip()


class DifficultyEvaluatorAgent:
    """Agent for evaluating question difficulty based on Q&A pairs.

    This agent is used to generate difficulty analysis entries for the
    difficulty analysis database, which provides context to the
    DifficultyAnalystAgent when routing queries.
    """

    def __init__(self, model_name: str | None = None):
        """Initialize the difficulty evaluator agent.

        Args:
            model_name: The model name to use (overrides settings)
        """
        self.model_name = get_model_name(
            model_name=model_name,
            agent_type="difficulty_evaluator",
        )
        self.client = get_openai_client(agent_type="difficulty_evaluator")

    async def evaluate(
        self,
        question: str,
        answer: str,
        correctness: str = "unknown",
    ) -> DifficultyEvaluationResult:
        """Evaluate question difficulty.

        Args:
            question: The question to evaluate
            answer: The model's answer to the question
            correctness: Whether the answer was correct (correct/incorrect/unknown)

        Returns:
            DifficultyEvaluationResult containing the difficulty evaluation
        """
        user_prompt = DIFFICULTY_EVALUATOR_USER_PROMPT.format(
            question=question,
            answer=answer,
            correctness=correctness,
        )

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": DIFFICULTY_EVALUATOR_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        result_output = response.choices[0].message.content or ""

        return DifficultyEvaluationResult(
            question=question,
            answer=answer,
            correctness=correctness,
            analysis=result_output,
        )

    async def batch_evaluate(
        self,
        qa_pairs: list[dict[str, Any]],
    ) -> list[DifficultyEvaluationResult]:
        """Evaluate multiple Q&A pairs.

        Args:
            qa_pairs: List of dicts with 'question', 'answer', and optionally 'correctness'

        Returns:
            List of DifficultyEvaluationResult
        """
        results = []
        for qa in qa_pairs:
            evaluation = await self.evaluate(
                question=qa["question"],
                answer=qa["answer"],
                correctness=qa.get("correctness", "unknown"),
            )
            results.append(evaluation)
        return results
