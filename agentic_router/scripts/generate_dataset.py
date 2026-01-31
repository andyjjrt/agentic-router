import json
import os

import requests
import tiktoken
from datasets import load_dataset

from agentic_router.core.config import settings

# Global tiktoken encoder
ENCODER = tiktoken.get_encoding("cl100k_base")

# Price cache file path
PRICE_CACHE_FILE = settings.get_data_path("openrouter_prices.json")


def get_openrouter_prices(use_cache=True):
    """Fetches model pricing from the OpenRouter API, with optional caching."""
    # Try to load from cache first
    if use_cache and os.path.exists(PRICE_CACHE_FILE):
        try:
            with open(PRICE_CACHE_FILE, "r", encoding="utf-8") as f:
                print(f"Loading prices from cache: {PRICE_CACHE_FILE}")
                return json.load(f)
        except (json.JSONDecodeError, IOError, FileNotFoundError) as e:
            print(f"Failed to load cache: {e}. Fetching from API...")

    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        models_data = response.json()["data"]

        prices = {}
        for model in models_data:
            model_id = model["id"]
            prices[model_id] = {
                "input_cost_per_token": float(
                    model.get("pricing", {}).get("prompt", 0.0)
                )
                / 1_000_000,
                "output_cost_per_token": float(
                    model.get("pricing", {}).get("completion", 0.0)
                )
                / 1_000_000,
            }

        # Save to cache
        with open(PRICE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(prices, f, indent=2, ensure_ascii=False)
            print(f"Prices cached to: {PRICE_CACHE_FILE}")

        return prices
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch OpenRouter prices: {e}")
        return {}


def get_model_mapping():
    """Returns a manual mapping from dataset model names to OpenRouter model names."""
    return {
        # Anthropic Claude models
        "claude-3-5-haiku-20241022": "anthropic/claude-3.5-haiku",
        "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
        "claude-3-7-sonnet-20250219": "anthropic/claude-3.7-sonnet",
        "claude-3-7-sonnet-20250219-thinking-32k": "anthropic/claude-3.7-sonnet",
        "claude-opus-4-1-20250805": "anthropic/claude-opus-4",
        "claude-opus-4-1-20250805-thinking-16k": "anthropic/claude-opus-4",
        "claude-opus-4-1-20250805-thinking-16k-old": "anthropic/claude-opus-4",
        "claude-opus-4-20250514": "anthropic/claude-opus-4",
        "claude-opus-4-20250514-thinking-16k": "anthropic/claude-opus-4",
        "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
        "claude-sonnet-4-20250514-thinking-32k": "anthropic/claude-sonnet-4",
        "claude-sonnet-4-5-20250929-old": "anthropic/claude-sonnet-4.5",
        "claude-sonnet-4-5-20250929-thinking-32k": "anthropic/claude-sonnet-4.5",
        # OpenAI GPT models
        "chatgpt-4o-latest-20250326": "openai/chatgpt-4o-latest",
        "chatgpt-4o-latest-20250326-old": "openai/chatgpt-4o-latest",
        "gpt-4.1-2025-04-14": "openai/gpt-4.1",
        "gpt-4.1-mini-2025-04-14": "openai/gpt-4.1-mini",
        "gpt-4o-2024-11-20": "openai/gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18": "openai/gpt-4o-mini-2024-07-18",
        "gpt-5-chat": "openai/gpt-5",
        "gpt-5-high": "openai/gpt-5",
        "gpt-5-high-new-system-prompt": "openai/gpt-5",
        "gpt-5-mini-high": "openai/gpt-5-mini",
        "gpt-5-nano-high": "openai/gpt-5-nano",
        "gpt-5-old": "openai/gpt-5",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "o3-2025-04-16": "openai/o3",
        "o3-mini": "openai/o3-mini",
        "o4-mini-2025-04-16": "openai/o4-mini",
        # Google Gemini models
        "gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
        "gemini-2.0-flash-thinking-exp-01-21": "",
        "gemini-2.5-flash-lite-preview-06-17-thinking": "google/gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-preview-09-2025-no-thinking": "google/gemini-2.5-flash-lite-preview-09-2025",
        "gemini-2.5-flash-preview-04-17": "google/gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash-preview-09-2025": "google/gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-pro-preview-03-25": "google/gemini-2.5-pro-preview",
        "gemini-2.5-pro-preview-05-06": "google/gemini-2.5-pro-preview",
        "gemma-3-27b-it": "google/gemma-3-27b-it",
        "gemma-3n-e4b-it": "google/gemma-3n-e4b-it",
        # DeepSeek models
        "deepseek-r1-0528": "",
        "deepseek-v3-0324": "deepseek/deepseek-chat-v3-0324",
        "deepseek-v3.1": "deepseek/deepseek-chat-v3.1",
        "deepseek-v3.1-terminus": "deepseek/deepseek-v3.1-terminus",
        "deepseek-v3.1-terminus-thinking": "deepseek/deepseek-v3.1-terminus",
        "deepseek-v3.1-thinking": "deepseek/deepseek-chat-v3.1",
        "deepseek-v3.2-exp": "deepseek/deepseek-v3.2-exp",
        "deepseek-v3.2-exp-thinking": "deepseek/deepseek-v3.2-exp",
        # Qwen models
        "qwen-max-2025-01-25": "qwen/qwen-max",
        "qwen-vl-max-2025-08-13": "qwen/qwen-vl-max",
        "qwen3-235b-a22b": "qwen/qwen3-235b-a22b",
        "qwen3-235b-a22b-instruct-2507": "qwen/qwen3-235b-a22b-2507",
        "qwen3-235b-a22b-instruct-2507-invalid": "qwen/qwen3-235b-a22b-2507",
        "qwen3-235b-a22b-no-thinking": "qwen/qwen3-235b-a22b",
        "qwen3-235b-a22b-thinking-2507": "qwen/qwen3-235b-a22b-thinking-2507",
        "qwen3-30b-a3b": "qwen/qwen3-30b-a3b",
        "qwen3-30b-a3b-instruct-2507": "qwen/qwen3-30b-a3b",
        "qwen3-coder-480b-a35b-instruct": "qwen/qwen3-coder",
        "qwen3-max-2025-09-23": "qwen/qwen3-max",
        "qwen3-max-2025-09-26": "qwen/qwen3-max",
        "qwen3-max-preview": "qwen/qwen3-max",
        "qwen3-next-80b-a3b-instruct": "qwen/qwen3-next-80b-a3b-instruct",
        "qwen3-next-80b-a3b-thinking": "qwen/qwen3-next-80b-a3b-thinking",
        "qwen3-vl-235b-a22b-instruct": "qwen/qwen3-vl-235b-a22b-instruct",
        "qwen3-vl-235b-a22b-thinking": "qwen/qwen3-vl-235b-a22b-thinking",
        "qwq-32b": "qwen/qwq-32b",
        # xAI Grok models
        "grok-3-mini-beta": "x-ai/grok-3-mini-beta",
        "grok-3-mini-high": "x-ai/grok-3-mini-beta",
        "grok-3-preview-02-24": "x-ai/grok-3-beta",
        "grok-4-0709": "x-ai/grok-4",
        "grok-4-0709-old2": "x-ai/grok-4",
        "grok-4-fast": "x-ai/grok-4-fast",
        "grok-4-fast-reasoning": "x-ai/grok-4-fast",
        # Mistral models
        # "magistral-medium-2506": "mistralai/magistral-medium-2506",
        "magistral-medium-2506": "",
        "mistral-medium-2505": "mistralai/mistral-medium-3",
        "mistral-medium-2508": "mistralai/mistral-medium-3.1",
        "mistral-small-2506": "mistralai/mistral-small-3.2-24b-instruct",
        "mistral-small-3.1-24b-instruct-2503": "mistralai/mistral-small-3.1-24b-instruct",
        # Meta Llama models
        "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
        "llama-4-maverick-03-26-experimental": "meta-llama/llama-4-maverick",
        "llama-4-maverick-17b-128e-instruct": "meta-llama/llama-4-maverick",
        "llama-4-scout-17b-16e-instruct": "meta-llama/llama-4-scout",
        # Amazon Nova models
        "amazon-nova-experimental-chat-05-14": "amazon/nova-pro-v1",
        "amazon.nova-pro-v1:0": "amazon/nova-pro-v1",
        # Cohere models
        "command-a-03-2025": "cohere/command-a",
        # GLM models (Zhipu)
        "glm-4.5": "z-ai/glm-4.5",
        "glm-4.5-air": "z-ai/glm-4.5-air",
        "glm-4.5v": "z-ai/glm-4.5v",
        "glm-4.6": "z-ai/glm-4.6",
        # Hunyuan models (Tencent)
        "hunyuan-t1-20250711": "",
        "hunyuan-turbos-20250416": "",
        "hunyuan-vision-1.5-thinking": "",
        # Kimi (Moonshot) models
        "kimi-k2-0711-preview": "moonshotai/kimi-k2",
        "kimi-k2-0905-preview": "moonshotai/kimi-k2",
        # Minimax models
        "minimax-m1": "minimax/minimax-01",
        # NVIDIA models
        "nvidia-llama-3.3-nemotron-super-49b-v1.5": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        # StepFun models
        "step-1o-turbo-202506": "",
        "step-3": "stepfun-ai/step3",
        # IBM models
        "ibm-granite-h-small": "",
        # Microsoft MAI
        "mai-1-preview": "",
        # Baichuan/Ling models
        "ling-flash-2.0": "",
        "ring-flash-2.0": "",
        # Other models
        "MiMo-7B": "",
        "MiMo-VL-7B-RL-2508": "",
        "longcat-flash-chat": "meituan/longcat-flash-chat",
    }


def parse_weird_string(s):
    # Mock numpy array and dtype
    def array(data, dtype=None):
        return data

    context = {
        "array": array,
        "object": object,
        "dtype": "dtype",
        "int64": "int64",
        "float64": "float64",
        "nan": float("nan"),
    }

    try:
        # Fix missing commas between list elements (e.g. "}\n {")
        import re

        s = re.sub(r"\}\s*\{", "}, {", s)

        # We use eval because ast.literal_eval doesn't support function calls (array())
        return eval(s, context)
    except Exception as e:
        print(f"Failed to eval: {e}")
        return None


def extract_text_from_content(content):
    # content is a list of dicts like [{'type': 'text', 'text': '...', ...}]
    # or just a string?
    # Based on inspection, it's a list of dicts.
    text_parts = []
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
    return "".join(text_parts)


def process_row(row, prices, model_mapping):
    try:
        conv_a = parse_weird_string(row["conversation_a"])
        conv_b = parse_weird_string(row["conversation_b"])
    except Exception as e:
        print(f"Error parsing row {row['id']}: {e}")
        return [], []

    if not conv_a or not conv_b:
        return [], []

    # Assuming structure is list of messages
    # We want to extract the input (history) and the output (last message)

    # Check consistency
    if len(conv_a) < 2:
        return [], []

    # Input is everything up to the last message
    # For A and B, the input *should* be the same for the comparison to be valid.
    # We can check this or just assume.

    input_messages = conv_a[:-1]

    # Extract text from input messages for a simple "prompt" field if needed,
    # but storing the full messages structure is better for chat models.
    # RouterBench often uses a single string "prompt", but for chat it might be json.
    # We will store `messages` list.

    # Clean up input messages (remove numpy arrays if any remained?)
    # The parse_weird_string already returns clean lists/dicts (mostly).
    # But content inside might be complex.
    # Let's simplify content to just string if possible, or keep as is.
    # The inspector showed content as list of dicts.
    # We should probably convert content to string for simpler "prompt" field,
    # and keep "messages" as structured.

    prompt_text = ""
    for msg in input_messages:
        content_text = extract_text_from_content(msg["content"])
        prompt_text += f"{msg['role']}: {content_text}\n"

    # Tokenize input (disable special token check to handle any text)
    input_tokens = len(ENCODER.encode(prompt_text, disallowed_special=()))

    # Responses
    resp_a_obj = conv_a[-1]
    resp_b_obj = conv_b[-1]

    resp_a_text = extract_text_from_content(resp_a_obj["content"])
    resp_b_text = extract_text_from_content(resp_b_obj["content"])

    # Tokenize and calculate cost
    resp_a_tokens = len(ENCODER.encode(resp_a_text, disallowed_special=()))
    resp_b_tokens = len(ENCODER.encode(resp_b_text, disallowed_special=()))

    model_a_id = row["model_a"]
    model_b_id = row["model_b"]

    if model_a_id not in model_mapping:
        raise ValueError(
            f"Model '{model_a_id}' not found in model_mapping. Please add it."
        )
    if model_b_id not in model_mapping:
        raise ValueError(
            f"Model '{model_b_id}' not found in model_mapping. Please add it."
        )

    openrouter_model_a = model_mapping.get(model_a_id)
    openrouter_model_b = model_mapping.get(model_b_id)

    # Skip if OpenRouter model name is empty string
    if openrouter_model_a == "":
        return [], [model_a_id]
    if openrouter_model_b == "":
        return [], [model_b_id]

    if openrouter_model_a not in prices:
        raise ValueError(
            f"OpenRouter model '{openrouter_model_a}' (mapped from '{model_a_id}') not found in prices. Please update the mapping."
        )
    if openrouter_model_b not in prices:
        raise ValueError(
            f"OpenRouter model '{openrouter_model_b}' (mapped from '{model_b_id}') not found in prices. Please update the mapping."
        )

    price_a = prices[openrouter_model_a]
    price_b = prices[openrouter_model_b]

    input_cost_a = input_tokens * price_a.get("input_cost_per_token", 0.0)
    input_cost_b = input_tokens * price_b.get("input_cost_per_token", 0.0)
    output_cost_a = resp_a_tokens * price_a.get("output_cost_per_token", 0.0)
    output_cost_b = resp_b_tokens * price_b.get("output_cost_per_token", 0.0)
    cost_a = input_cost_a + output_cost_a
    cost_b = input_cost_b + output_cost_b

    # Determine scores
    winner = row["winner"]
    score_a = 0.0
    score_b = 0.0

    if winner == "model_a":
        score_a = 1.0
        score_b = 0.0
    elif winner == "model_b":
        score_a = 0.0
        score_b = 1.0
    elif winner == "tie":
        score_a = 0.5
        score_b = 0.5
    elif winner == "both_bad":
        score_a = 0.0
        score_b = 0.0

    # Create entries
    entry_a = {
        "id": f"{row['id']}_a",
        "model": model_a_id,
        "openrouter_model": openrouter_model_a,
        "prompt": prompt_text.strip(),
        "messages": input_messages,  # Keep structured
        "response": resp_a_text,
        "input_tokens": input_tokens,
        "response_tokens": resp_a_tokens,
        "response_cost": cost_a,
        "score": score_a,
        "language": row["language"],
        "source": "arena-expert-5k",
    }

    entry_b = {
        "id": f"{row['id']}_b",
        "model": model_b_id,
        "openrouter_model": openrouter_model_b,
        "prompt": prompt_text.strip(),
        "messages": input_messages,
        "response": resp_b_text,
        "input_tokens": input_tokens,
        "response_tokens": resp_b_tokens,
        "response_cost": cost_b,
        "score": score_b,
        "language": row["language"],
        "source": "arena-expert-5k",
    }

    return [entry_a, entry_b], []


def main():
    print("Fetching OpenRouter model prices...")
    prices = get_openrouter_prices()
    if not prices:
        print("Could not fetch prices. Aborting.")
        return

    model_mapping = get_model_mapping()

    print("Loading dataset...")
    ds = load_dataset("lmarena-ai/arena-expert-5k", split="train")

    output_file = settings.get_data_path("router_bench_dataset.jsonl")
    print(f"Generating dataset to {output_file}...")

    count = 0
    skipped_models = set()
    with open(output_file, "w", encoding="utf-8") as f:
        for row in ds:
            entries, skipped = process_row(row, prices, model_mapping)
            skipped_models.update(skipped)
            for entry in entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
                count += 1

    print(f"Done. Generated {count} entries.")
    if skipped_models:
        print(f"\nSkipped models (empty OpenRouter mapping): {sorted(skipped_models)}")


if __name__ == "__main__":
    main()
