from embedding import create_embedding_model
from retriever import retrieve_documents


def test_retriever():

    try:

        embedding_model = create_embedding_model()

        question = "What is the annual rent?"

        results = retrieve_documents(
            question,
            embedding_model,
            top_k=3
        )

        print(
            f"Documents retrieved: {len(results)}"
        )


        for i, doc in enumerate(results):

            print("\nResult:", i + 1)

            print(doc.page_content)

            print(doc.metadata)


    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    test_retriever()