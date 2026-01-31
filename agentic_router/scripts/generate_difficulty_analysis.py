"""Generate difficulty analysis for dataset entries using the evaluator agent.

This script reads the generated dataset, analyzes difficulty using the
DifficultyEvaluatorAgent, and stores the results along with embeddings
using FAISS for efficient similarity search.

Features:
- Supports job continuation with --continue flag
- Records failed analysis entries (analysis=None)
- Supports --redo-failed to reprocess failed entries
- Saves progress periodically to prevent data loss
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm as tqdm_async

from agentic_router.agents.difficulty_evaluator import DifficultyEvaluatorAgent
from agentic_router.core.config import settings
from agentic_router.core.logging import setup_logger
from agentic_router.database import (
    DifficultyAnalysisDatabase,
    DifficultyAnalysisEntry,
)
from agentic_router.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


def get_failed_entry_ids(db: DifficultyAnalysisDatabase) -> set[str]:
    """Get IDs of entries with failed analysis (analysis is None)."""
    return {entry.id for entry in db.entries if entry.analysis is None}


def get_processed_entry_ids(db: DifficultyAnalysisDatabase) -> set[str]:
    """Get IDs of all processed entries."""
    return {entry.id for entry in db.entries}


def remove_entries_by_ids(
    db: DifficultyAnalysisDatabase, ids_to_remove: set[str]
) -> DifficultyAnalysisDatabase:
    """Remove entries by IDs and rebuild FAISS index.

    Note: This creates a new database without the specified entries.
    The FAISS index needs to be rebuilt since we can't remove individual vectors.
    """
    import faiss
    import numpy as np

    # Get indices to keep
    indices_to_keep = [
        i for i, entry in enumerate(db.entries) if entry.id not in ids_to_remove
    ]

    # Create new database
    new_db = DifficultyAnalysisDatabase(
        embedding_model_name=db.embedding_model_name,
        embedding_dim=db.embedding_dim,
    )

    # Copy entries and rebuild index
    if db._embeddings is not None and len(indices_to_keep) > 0:
        new_db.entries = [db.entries[i] for i in indices_to_keep]
        new_embeddings = db._embeddings[indices_to_keep]
        new_db._embeddings = new_embeddings
        new_db.faiss_index = faiss.IndexFlatIP(db.embedding_dim)
        new_db.faiss_index.add(new_embeddings.astype(np.float32))
    elif len(indices_to_keep) > 0:
        # No embeddings stored, need to rebuild without them
        new_db.entries = [db.entries[i] for i in indices_to_keep]

    return new_db


def load_dataset(dataset_path: str) -> list[dict[str, Any]]:
    """Load the JSONL dataset."""
    entries = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def deduplicate_by_prompt(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate entries by prompt to avoid redundant analysis."""
    seen_prompts = set()
    unique_entries = []
    for entry in entries:
        prompt = entry.get("prompt", "")
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            unique_entries.append(entry)
    return unique_entries


def filter_entries_for_processing(
    entries: list[dict[str, Any]],
    existing_db: DifficultyAnalysisDatabase | None,
    redo_failed: bool = False,
) -> tuple[list[dict[str, Any]], DifficultyAnalysisDatabase | None]:
    """Filter entries based on existing database and redo_failed flag.

    Args:
        entries: All entries from the dataset
        existing_db: Existing database (if continuing)
        redo_failed: Whether to redo failed entries

    Returns:
        Tuple of (entries to process, updated database)
    """
    if existing_db is None:
        return entries, None

    processed_ids = get_processed_entry_ids(existing_db)
    failed_ids = get_failed_entry_ids(existing_db)

    logger.info(
        "Existing database has %d entries (%d failed)",
        len(processed_ids),
        len(failed_ids),
    )

    if redo_failed and failed_ids:
        # Remove failed entries from database so they can be reprocessed
        logger.info("Removing %d failed entries for reprocessing...", len(failed_ids))
        existing_db = remove_entries_by_ids(existing_db, failed_ids)
        processed_ids = processed_ids - failed_ids

    # Filter out already successfully processed entries
    entries_to_process = [
        entry for entry in entries if entry.get("id", "") not in processed_ids
    ]

    logger.info("Filtered to %d entries to process", len(entries_to_process))

    return entries_to_process, existing_db


async def generate_analysis(
    entries: list[dict[str, Any]],
    evaluator: DifficultyEvaluatorAgent,
    embedding_model: EmbeddingModel,
    batch_size: int = 10,
    debug: bool = False,
    existing_db: DifficultyAnalysisDatabase | None = None,
    output_path: str | None = None,
    save_interval: int = 50,
    max_concurrency: int = 10,
) -> DifficultyAnalysisDatabase:
    """Generate difficulty analysis for all entries.

    Args:
        entries: List of entries to process
        evaluator: Difficulty evaluator agent
        embedding_model: Embedding model for generating embeddings
        batch_size: Number of entries to process in each batch
        debug: Enable debug logging
        existing_db: Existing database to continue from
        output_path: Path to save progress (for periodic saves)
        save_interval: Save progress every N entries
        max_concurrency: Maximum number of concurrent analysis tasks

    Returns:
        Updated database with all entries
    """
    embedding_dim = await embedding_model.get_embedding_dim()

    # Use existing database or create new one
    if existing_db is not None:
        db = existing_db
        # Ensure embedding dim matches
        if db.embedding_dim != embedding_dim:
            logger.warning(
                "Embedding dimension mismatch: existing=%d, new=%d. Creating new database.",
                db.embedding_dim,
                embedding_dim,
            )
            db = DifficultyAnalysisDatabase(
                embedding_model_name=embedding_model.model_name,
                embedding_dim=embedding_dim,
            )
    else:
        db = DifficultyAnalysisDatabase(
            embedding_model_name=embedding_model.model_name,
            embedding_dim=embedding_dim,
        )

    failed_count = 0
    processed_count = 0
    semaphore = asyncio.Semaphore(max_concurrency)
    lock = asyncio.Lock()  # Lock for thread-safe db operations

    async def process_entry(entry: dict[str, Any], embedding: list[float]) -> None:
        nonlocal failed_count, processed_count

        prompt = entry.get("prompt", "")
        response = entry.get("response", "")

        # Skip if prompt or response is empty
        if not prompt or not response:
            return

        # Determine correctness based on score
        score = entry.get("score", 0.0)
        if score >= 1.0:
            correctness = "correct"
        elif score > 0:
            correctness = "partially_correct"
        else:
            correctness = "incorrect"

        analysis = None
        async with semaphore:
            try:
                # Generate difficulty analysis
                result = await evaluator.evaluate(
                    question=prompt,
                    answer=response,
                    correctness=correctness,
                )
                analysis = result.analysis
                logger.debug(
                    "Entry %s raw analysis:\n%s",
                    entry.get("id"),
                    analysis,
                )
            except Exception as e:
                logger.error("Error analyzing entry %s: %s", entry.get("id"), e)
                async with lock:
                    failed_count += 1

        # Create and add entry (even if analysis failed, record it)
        analysis_entry = DifficultyAnalysisEntry(
            id=entry.get("id", ""),
            prompt=prompt,
            response=response,
            model=entry.get("model", ""),
            openrouter_model=entry.get("openrouter_model", ""),
            score=score,
            analysis=analysis,
        )

        async with lock:
            db.add_entry(analysis_entry, embedding)
            processed_count += 1

            # Periodic save to prevent data loss
            if (
                output_path
                and processed_count > 0
                and processed_count % save_interval == 0
            ):
                logger.info(
                    "Saving progress... (%d processed, %d failed)",
                    processed_count,
                    failed_count,
                )
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                db.save(output_path)

    # Prepare all tasks with embeddings
    all_tasks = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]

        # Batch embed all prompts
        prompts = [entry.get("prompt", "") for entry in batch]
        embeddings = await embedding_model.embed_batch(prompts)

        # Create tasks for all entries in the batch
        for entry, embedding in zip(batch, embeddings):
            all_tasks.append(process_entry(entry, embedding))

    # Run all tasks concurrently with tqdm_async progress bar
    await tqdm_async.gather(*all_tasks, desc="Processing entries", disable=debug)

    # Log summary
    logger.info(
        "Batch processing complete. Processed: %d, Failed in this run: %d, Total failed in DB: %d",
        processed_count,
        failed_count,
        len(get_failed_entry_ids(db)),
    )

    return db


def main():
    parser = argparse.ArgumentParser(
        description="Generate difficulty analysis with embeddings for dataset entries."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=settings.get_data_path("router_bench_dataset.jsonl"),
        help="Path to the input JSONL dataset file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=settings.get_data_path("difficulty_analysis_db.pkl"),
        help="Path to the output pickle file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for difficulty evaluation",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=settings.get_embedding_model_name(),
        help="Embedding model name",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=str,
        default=settings.get_embedding_base_url(),
        help="Base URL for embedding API",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deduplicate entries by prompt before processing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of entries to process (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--continue",
        dest="continue_job",
        action="store_true",
        help="Continue from existing database file (skip already processed entries)",
    )
    parser.add_argument(
        "--redo-failed",
        action="store_true",
        help="Redo failed analysis entries (entries with analysis=None). Implies --continue",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save progress every N entries to prevent data loss",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent analysis tasks (controls asyncio.Semaphore size)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to see raw analysis and other debug info",
    )
    args = parser.parse_args()

    # --redo-failed implies --continue
    if args.redo_failed:
        args.continue_job = True

    # Configure logging
    setup_logger(debug=args.debug)

    # Load dataset
    logger.info("Loading dataset from %s...", args.input)
    entries = load_dataset(args.input)
    logger.info("Loaded %d entries", len(entries))

    # Optionally deduplicate
    if args.deduplicate:
        entries = deduplicate_by_prompt(entries)
        logger.info("After deduplication: %d unique entries", len(entries))

    # Optionally limit entries
    if args.limit:
        entries = entries[: args.limit]
        logger.info("Limited to %d entries", len(entries))

    # Load existing database if continuing
    existing_db = None
    if args.continue_job and Path(args.output).exists():
        logger.info("Loading existing database from %s...", args.output)
        try:
            existing_db = DifficultyAnalysisDatabase.load(args.output)
            logger.info(
                "Loaded existing database with %d entries", len(existing_db.entries)
            )
        except Exception as e:
            logger.warning("Failed to load existing database: %s. Starting fresh.", e)
            existing_db = None

    # Filter entries based on existing database
    entries_to_process, existing_db = filter_entries_for_processing(
        entries, existing_db, args.redo_failed
    )

    if not entries_to_process:
        logger.info(
            "No entries to process. All entries already processed successfully."
        )
        if existing_db:
            failed_count = len(get_failed_entry_ids(existing_db))
            if failed_count > 0:
                logger.info(
                    "Note: %d entries have failed analysis. Use --redo-failed to retry.",
                    failed_count,
                )
        return

    # Initialize embedding model
    logger.info("Initializing embedding model: %s...", args.embedding_model)
    logger.debug("Embedding API base URL: %s", args.embedding_base_url)
    embedding_model = EmbeddingModel(
        model_name=args.embedding_model,
        base_url=args.embedding_base_url,
    )
    logger.info("Embedding model initialized")

    # Initialize evaluator agent
    logger.info("Initializing difficulty evaluator agent...")
    evaluator = DifficultyEvaluatorAgent(
        model_name=args.model,
    )
    logger.info("Evaluator initialized with model: %s", evaluator.model_name)

    # Generate analysis
    logger.info("Running analysis on %d entries...", len(entries_to_process))
    db = asyncio.run(
        generate_analysis(
            entries_to_process,
            evaluator,
            embedding_model,
            args.batch_size,
            args.debug,
            existing_db,
            args.output,
            args.save_interval,
            args.max_concurrency,
        )
    )

    # Save database
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    db.save(args.output)

    # Report statistics
    failed_ids = get_failed_entry_ids(db)
    logger.info("Done! Generated %d difficulty analysis entries.", len(db.entries))
    logger.info("Embedding dimension: %d", db.embedding_dim)
    logger.info("FAISS index size: %d", db.faiss_index.ntotal if db.faiss_index else 0)
    logger.info("Failed entries: %d", len(failed_ids))
    if failed_ids:
        logger.info("Use --redo-failed to retry failed entries.")
    logger.info("Output saved to: %s", args.output)


if __name__ == "__main__":
    main()
