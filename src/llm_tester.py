from llm import create_llm


def test_llm():

    try:

        llm = create_llm()

        response = llm.invoke(
            "Explain what RAG is in one sentence."
        )

        print(response.content)


    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    test_llm()