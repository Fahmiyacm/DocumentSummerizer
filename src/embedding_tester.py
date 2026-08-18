from pathlib import Path

from document_loader import load_document
from text_splitter import split_documents
from embedding import create_embedding_model, create_embeddings


def test_document_pipeline(pdf_path: str) -> None:
    """
    Test PDF loading, chunking, and embedding generation.
    """

    try:
        file_path = Path(pdf_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Step 1: Load PDF
        documents = load_document(str(file_path))
        print(f"Total pages loaded: {len(documents)}")

        # Step 2: Split into chunks
        chunks = split_documents(documents)
        print(f"Total chunks created: {len(chunks)}")

        # Step 3: Create multilingual embedding model
        embedding_model = create_embedding_model()
        print("Embedding model loaded successfully.")

        # Step 4: Generate embeddings
        embeddings = create_embeddings(
            chunks,
            embedding_model
        )

        print(f"Total embeddings created: {len(embeddings)}")

        if embeddings:
            print(f"Embedding dimensions: {len(embeddings[0])}")

            print("\nFirst chunk:")
            print(chunks[0].page_content)

            print("\nFirst chunk metadata:")
            print(chunks[0].metadata)

            print("\nFirst embedding (first 10 values):")
            print(embeddings[0][:10])

    except FileNotFoundError as error:
        print(f"File Error: {error}")

    except Exception as error:
        print(f"Pipeline Error: {error}")


if __name__ == "__main__":

    pdf_path = (
        r"C:\Users\fahmi\PycharmProjects\DocumentSummery"
        r"\data\sample_uae_rental_contract.pdf"
    )

    test_document_pipeline(pdf_path)