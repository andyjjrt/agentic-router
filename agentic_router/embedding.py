"""Embedding model wrapper using raw OpenAI client."""

import numpy as np

from agentic_router.clients import get_embedding_client
from agentic_router.core.config import settings


class EmbeddingModel:
    """Wrapper for OpenAI embeddings with numpy array output.

    This class provides a convenient interface for generating embeddings
    using raw OpenAI client, returning numpy arrays for compatibility
    with FAISS and other vector databases.
    """

    def __init__(
        self,
        model_name: str = settings.get_embedding_model_name(),
        base_url: str = settings.get_embedding_base_url(),
        api_key: str | None = settings.get_embedding_api_key(),
    ):
        """Initialize the embedding model.

        Args:
            model_name: Name of the embedding model (default: from settings)
            base_url: Base URL for the API (default: from settings)
            api_key: API key (default: from settings)
        """
        self.model_name = model_name
        self._client = get_embedding_client(api_key=api_key, base_url=base_url)
        self._embedding_dim: int | None = None

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding as a numpy array
        """
        response = await self._client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        if self._embedding_dim is None:
            self._embedding_dim = len(embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            Embeddings as a 2D numpy array (num_texts x embedding_dim)
        """
        if not texts:
            return np.array([], dtype=np.float32)

        response = await self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        embeddings = np.array(
            [item.embedding for item in response.data], dtype=np.float32
        )

        if self._embedding_dim is None and len(embeddings) > 0:
            self._embedding_dim = embeddings.shape[1]

        return embeddings

    async def get_embedding_dim(self) -> int:
        """Get the embedding dimension (fetches a sample if not known).

        Returns:
            The embedding dimension
        """
        if self._embedding_dim is None:
            await self.embed("sample")
        return self._embedding_dim or 0

    @property
    def embedding_dim(self) -> int | None:
        """Get the embedding dimension if known (without fetching).

        Returns:
            The embedding dimension or None if not yet determined
        """
        return self._embedding_dim
