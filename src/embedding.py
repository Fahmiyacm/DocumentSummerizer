from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create a multilingual embedding model for English and Arabic documents.

    Returns:
        HuggingFaceEmbeddings: Initialized embedding model.
    """

    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        return embedding_model

    except Exception as error:
        raise RuntimeError(
            f"Failed to initialize embedding model: {error}"
        )


def create_embeddings(
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings
) -> list[list[float]]:
    """
    Convert document chunks into vector embeddings.

    Args:
        documents: List of document chunks.
        embedding_model: Initialized embedding model.

    Returns:
        List of embedding vectors.
    """

    try:
        if not documents:
            raise ValueError("No documents provided for embedding.")

        texts = [document.page_content for document in documents]

        embeddings = embedding_model.embed_documents(texts)

        if not embeddings:
            raise ValueError("Embedding generation returned no vectors.")

        return embeddings

    except ValueError as error:
        raise error

    except Exception as error:
        raise RuntimeError(
            f"Failed to create document embeddings: {error}"
        )