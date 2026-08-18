from embedding import create_embedding_model
from rag_pipeline import generate_answer


def test_rag():

    try:

        embedding_model = create_embedding_model()


        question = (
            "What is the annual rent?"
        )

        answer = generate_answer(
            question,
            embedding_model
        )

        print("\nAnswer:")
        print(answer["answer"])

        print("\nSources:")

        for source in answer["sources"]:
            print(source)


    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    test_rag()