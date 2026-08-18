from pathlib import Path

from document_loader import load_document
from text_splitter import split_documents


def test_document_pipeline(pdf_path: str) -> None:
    """
    Test PDF loading and document chunking pipeline.

    Args:
        pdf_path (str): Path to the PDF file.
    """

    try:
        file_path = Path(pdf_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        documents = load_document(str(file_path))

        print(f"Total pages loaded: {len(documents)}")

        chunks = split_documents(documents)

        print(f"Total chunks created: {len(chunks)}")

        if chunks:
            print("\nFirst Chunk Content:")
            print(chunks[0].page_content)

            print("\nFirst Chunk Metadata:")
            print(chunks[0].metadata)

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