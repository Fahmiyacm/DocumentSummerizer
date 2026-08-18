from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document]
) -> list[Document]:
    """
    Split documents into meaningful chunks.
    """

    try:

        if not documents:
            raise ValueError(
                "No documents provided."
            )


        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            separators=[
                "\n\n",
                "\n",
                ".",
                " ",
                ""
            ]
        )


        chunks = splitter.split_documents(
            documents
        )


        # Add chunk id
        for index, chunk in enumerate(chunks):

            chunk.metadata["chunk_id"] = index


        return chunks


    except ValueError as error:
        raise error


    except Exception as error:
        raise RuntimeError(
            f"Chunking failed: {error}"
        )