# Agentic Router

An intelligent multi-agent routing system that analyzes query difficulty and routes requests to the most capable language models.

## Features

- **Difficulty Analysis**: Evaluates queries across multiple dimensions (reasoning, comprehension, coding, etc.)
- **Difficulty Evaluation**: Evaluates model responses for dataset generation
- **Smart Routing**: Routes queries to capable models based on past performance
- **FAISS-based Similarity Search**: Efficient embedding storage and retrieval for routing decisions
- **OpenAI-Compatible**: Works with any OpenAI-compatible LLM gateway

## Installation

```bash
# Install dependencies
uv sync

# Install dev dependencies (includes ruff and pre-commit)
uv sync --group dev
```

## Configuration

Create a `.env` file in the project root:

```env
LLM_API_KEY=your-api-key
LLM_BASE_URL=http://localhost:8000/v1
```

### Configuration Options

#### Global Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | API key for LLM gateway | `None` |
| `LLM_BASE_URL` | Base URL for LLM gateway | `http://0.0.0.0:8000/v1` |
| `DEFAULT_TEMPERATURE` | Default temperature for generation | `0.5` |
| `DATA_DIR` | Directory for data files | `data` |

#### Agent-Specific Settings

Each agent can be configured independently with its own model, API key, and base URL. Use the following prefixes:

| Agent | Prefix | Example |
|-------|--------|---------|
| Difficulty Analyst | `DIFFICULTY_ANALYST_` | `DIFFICULTY_ANALYST_MODEL_NAME` |
| Difficulty Evaluator | `DIFFICULTY_EVALUATOR_` | `DIFFICULTY_EVALUATOR_MODEL_NAME` |
| Routing Decision Maker | `ROUTING_DECISION_MAKER_` | `ROUTING_DECISION_MAKER_MODEL_NAME` |
| Embedding | `EMBEDDING_` | `EMBEDDING_MODEL_NAME` |

Available options per agent:
- `{PREFIX}MODEL_NAME` - Model name to use
- `{PREFIX}API_KEY` - API key (falls back to `LLM_API_KEY`)
- `{PREFIX}BASE_URL` - Base URL (falls back to `LLM_BASE_URL`)

#### Embedding Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL_NAME` | Model name for embeddings | `Qwen/Qwen3-Embedding-0.6B` |
| `EMBEDDING_API_KEY` | API key for embedding model | Falls back to `LLM_API_KEY` |
| `EMBEDDING_BASE_URL` | Base URL for embedding API | Falls back to `LLM_BASE_URL` |

## Scripts

### Generate Dataset

Generate a dataset from RouterBench:

```bash
uv run python -m agentic_router.scripts.generate_dataset
```

### Generate Difficulty Analysis

Analyze difficulty for dataset entries and store with FAISS embeddings:

```bash
uv run python -m agentic_router.scripts.generate_difficulty_analysis \
    --input data/router_bench_dataset.jsonl \
    --output data/difficulty_analysis_db.pkl \
    --batch-size 10 \
    --max-concurrency 10
```

Options:
- `--input` - Input JSONL dataset file
- `--output` - Output pickle file for the database
- `--model` - Model name for difficulty evaluation
- `--embedding-model` - Embedding model name
- `--batch-size` - Batch size for processing (default: 10)
- `--max-concurrency` - Maximum concurrent analysis tasks (default: 10)
- `--continue` - Continue from existing database file
- `--redo-failed` - Retry failed analysis entries
- `--save-interval` - Save progress every N entries (default: 50)
- `--deduplicate` - Deduplicate entries by prompt
- `--limit` - Limit number of entries (for testing)
- `--debug` - Enable debug logging

## Development

### Setup Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install
```

### Code Formatting

```bash
# Format code with ruff
ruff format .

# Lint and fix issues
ruff check --fix .
```

## Project Structure

```
agentic_router/
├── agents/
│   ├── __init__.py
│   ├── difficulty_analyst.py      # Query difficulty analysis agent
│   ├── difficulty_evaluator.py    # Response evaluation agent
│   └── routing_decision_maker.py  # Model routing decision agent
├── core/
│   ├── __init__.py
│   ├── config.py                  # Configuration settings
│   └── logging.py                 # Logging setup
├── database/
│   ├── __init__.py
│   └── difficulty_analysis.py     # FAISS-based difficulty analysis storage
├── models/
│   ├── __init__.py
│   ├── difficulty.py              # Difficulty analysis models
│   └── routing.py                 # Routing decision models
├── scripts/
│   ├── __init__.py
│   ├── generate_dataset.py        # Dataset generation from RouterBench
│   └── generate_difficulty_analysis.py  # Difficulty analysis generation
├── clients.py                     # LLM client utilities
└── embedding.py                   # Embedding model wrapper
```

## License

MIT
