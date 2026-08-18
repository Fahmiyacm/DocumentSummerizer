import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


MODEL_NAME = "openai/gpt-oss-20b"


def create_llm():
    """
    Initialize Groq LLM model.
    """

    try:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to .env file."
            )


        llm = ChatGroq(
            model=MODEL_NAME,
            temperature=0.1,
            api_key=api_key
        )


        return llm


    except ValueError as error:
        raise error


    except Exception as error:
        raise RuntimeError(
            f"Failed to initialize LLM: {error}"
        )