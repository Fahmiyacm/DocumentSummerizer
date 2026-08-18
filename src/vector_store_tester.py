from pathlib import Path

from document_loader import load_document
from text_splitter import split_documents
from embedding import create_embedding_model, create_embeddings
from vector_store import create_vector_store


def test_vector_store(pdf_path: str):

    try:
        file_path = Path(pdf_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        # 1. Load PDF
        documents = load_document(str(file_path))

        print(f"Documents loaded: {len(documents)}")


        # 2. Split documents
        chunks = split_documents(documents)

        print(f"Chunks created: {len(chunks)}")


        # 3. Create embedding model
        embedding_model = create_embedding_model()

        print("Embedding model loaded")


        # 4. Generate embeddings
        embeddings = create_embeddings(
            chunks,
            embedding_model
        )

        print(
            f"Embeddings created: {len(embeddings)}"
        )


        # 5. Create FAISS vector store
        index = create_vector_store(
            chunks,
            embeddings
        )

        print("\nFAISS Vector Store Created Successfully")

        print(
            f"Total vectors stored: {index.ntotal}"
        )


    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":

    pdf_path = (
        r"C:\Users\fahmi\PycharmProjects\DocumentSummery"
        r"\data\sample_uae_rental_contract.pdf"
    )

    test_vector_store(pdf_path)