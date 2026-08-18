from pathlib import Path

import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


VECTOR_DB_PATH = Path("vector_db")


def load_vector_store(index_name: str = "document_index"):
    """
    Load the FAISS index and its metadata.
    """

    try:
        index_path = VECTOR_DB_PATH / f"{index_name}.faiss"
        metadata_path = (
            VECTOR_DB_PATH / f"{index_name}_metadata.npy"
        )

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

    except FileNotFoundError:
        raise

    except Exception as error:
        raise RuntimeError(
            f"Failed to load vector store: {error}"
        )


def retrieve_documents(
    question: str,
    embedding_model: HuggingFaceEmbeddings,
    top_k: int = 5
) -> list[Document]:
    """
    Retrieve the most relevant document chunks from FAISS.

    No aggressive similarity threshold is used because short
    queries such as 'skills', 'projects', or 'education'
    can otherwise be incorrectly rejected.
    """

    try:
        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        index, metadata = load_vector_store()

        # Create embedding for user question
        question_embedding = embedding_model.embed_query(
            question.strip()
        )

        query_vector = np.array(
            [question_embedding],
            dtype=np.float32
        )

        # Retrieve top matching chunks
        scores, indices = index.search(
            query_vector,
            top_k
        )

        documents = []

        for score, index_id in zip(
            scores[0],
            indices[0]
        ):

            if index_id < 0:
                continue

            if index_id >= len(metadata):
                continue

            item = metadata[index_id]

            document = Document(
                page_content=item["page_content"],
                metadata={
                    **item["metadata"],
                    "similarity_score": float(score)
                }
            )

            documents.append(document)

        return documents

    except ValueError:
        raise

    except Exception as error:
        raise RuntimeError(
            f"Document retrieval failed: {error}"
        )