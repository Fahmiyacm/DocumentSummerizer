from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document


VECTOR_DB_PATH = Path("vector_db")


def create_vector_store(
    documents: list[Document],
    embeddings: list[list[float]],
    index_name: str = "document_index"
) -> faiss.Index:
    """
    Create a FAISS vector index from document embeddings.

    Args:
        documents: List of document chunks.
        embeddings: Embedding vectors corresponding to the chunks.
        index_name: Name used when saving the index.

    Returns:
        FAISS index.
    """

    try:
        if not documents:
            raise ValueError("No documents provided.")

        if not embeddings:
            raise ValueError("No embeddings provided.")

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings."
            )

        vectors = np.array(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2-dimensional array.")

        dimension = vectors.shape[1]

        index = faiss.IndexFlatIP(dimension)

        index.add(vectors)

        save_vector_store(
            index=index,
            documents=documents,
            index_name=index_name
        )

        return index

    except ValueError as error:
        raise error

    except Exception as error:
        raise RuntimeError(
            f"Failed to create FAISS vector store: {error}"
        )


def save_vector_store(
    index: faiss.Index,
    documents: list[Document],
    index_name: str
) -> None:
    """
    Save FAISS index and document metadata to disk.
    """

    try:
        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

        index_path = VECTOR_DB_PATH / f"{index_name}.faiss"
        metadata_path = VECTOR_DB_PATH / f"{index_name}_metadata.npy"

        faiss.write_index(index, str(index_path))

        metadata = [
            {
                "page_content": document.page_content,
                "metadata": document.metadata
            }
            for document in documents
        ]

        np.save(
            metadata_path,
            metadata,
            allow_pickle=True
        )

        print(f"FAISS index saved: {index_path}")
        print(f"Metadata saved: {metadata_path}")

    except Exception as error:
        raise RuntimeError(
            f"Failed to save vector store: {error}"
        )


def load_vector_store(
    index_name: str = "document_index"
) -> tuple[faiss.Index, list[dict]]:
    """
    Load a FAISS index and its document metadata.
    """

    try:
        index_path = VECTOR_DB_PATH / f"{index_name}.faiss"
        metadata_path = VECTOR_DB_PATH / f"{index_name}_metadata.npy"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        index = faiss.read_index(str(index_path))

        metadata = np.load(
            metadata_path,
            allow_pickle=True
        ).tolist()

        return index, metadata

    except FileNotFoundError as error:
        raise error

    except Exception as error:
        raise RuntimeError(
            f"Failed to load vector store: {error}"
        )