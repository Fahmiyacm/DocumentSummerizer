
from src.retriever import retrieve_documents
from src.llm import create_llm


def generate_answer(
    question,
    embedding_model,
    language="en",
    top_k=5
):
    """
    Generate an answer using retrieved document context.

    language:
        en   -> English
        ar   -> Arabic
        both -> English + Arabic
    """

    try:

        # --------------------------------------------------
        # Validate question
        # --------------------------------------------------

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        # --------------------------------------------------
        # Retrieve relevant documents
        # --------------------------------------------------

        documents = retrieve_documents(
            question=question,
            embedding_model=embedding_model,
            top_k=top_k
        )

        print(
            "\n========== RETRIEVED DOCUMENTS =========="
        )

        for i, document in enumerate(documents):

            print(
                f"\n--- CHUNK {i + 1} ---"
            )

            print(
                "Page:",
                document.metadata.get(
                    "page",
                    "Unknown"
                )
            )

            print(
                "Score:",
                document.metadata.get(
                    "similarity_score",
                    "Unknown"
                )
            )

            print(
                document.page_content
            )

        print(
            "\n=========================================\n"
        )

        # --------------------------------------------------
        # No documents found
        # --------------------------------------------------

        if not documents:

            return {
                "answer": (
                    "Information not found "
                    "in the document."
                ),
                "sources": []
            }

        # --------------------------------------------------
        # Build context
        # --------------------------------------------------

        context_parts = []

        for document in documents:

            text = document.page_content.strip()

            if text:

                page = document.metadata.get(
                    "page",
                    "Unknown"
                )

                context_parts.append(
                    f"[Page {page}]\n{text}"
                )

        context = "\n\n---\n\n".join(
            context_parts
        )

        # --------------------------------------------------
        # Language instruction
        # --------------------------------------------------

        if language == "ar":

            language_instruction = """
Answer ONLY in Arabic.

Use clear and professional Modern Standard Arabic.

Do not provide an English translation.

All explanations and bullet points must be in Arabic.

Keep names, company names, technical terms,
document numbers, currency codes, and dates
unchanged when appropriate.
"""

        elif language == "both":

            language_instruction = """
Provide the answer in BOTH English and Arabic.

First provide the English answer.

Then provide the Arabic answer.

The Arabic answer must contain the same
information as the English answer.

Do not add information in either language.
"""

        else:

            language_instruction = """
Answer ONLY in English.

Use clear and professional business English.
"""

        # --------------------------------------------------
        # RAG Prompt
        # --------------------------------------------------

        prompt = f"""
You are a professional Business Document
Intelligence Assistant.

Your task is to answer the user's question
using ONLY the provided document context.

The document may be any type of business document,
including:

- Resume / CV
- Contract
- Agreement
- Invoice
- Report
- Policy
- Proposal
- Financial document
- Legal document
- Business correspondence
- Other business documents

IMPORTANT RULES:

1. Use ONLY information contained in the
   provided document context.

2. Never use outside knowledge.

3. Never guess or invent information.

4. Understand short keyword questions intelligently.

Examples:

"skills"
→ Find relevant skills, technologies,
  tools, or capabilities.

"education"
→ Find educational qualifications
  or academic background.

"experience"
→ Find work history, professional
  experience, or previous roles.

"country"
→ Find country, location, city,
  address, or geographical information.

"amount"
→ Find prices, salaries, payments,
  costs, fees, or financial values.

"date"
→ Find important dates, periods,
  deadlines, contract dates, or timelines.

"people"
→ Find names, parties, companies,
  organizations, or stakeholders.

"responsibility"
→ Find duties, obligations,
  roles, or assigned tasks.

"risk"
→ Find risks, penalties, liabilities,
  restrictions, or important conditions
  explicitly mentioned in the document.

"summary"
→ Give a short overview of the document
  based only on the available context.

5. If the requested information exists
   in the context, answer directly.

6. If multiple pieces of information exist,
   use bullet points.

7. Keep the answer concise.

8. Do not repeat unnecessary information.

9. If the requested information is NOT present
   in the provided context, answer exactly:

Information not found in the document.

10. Do not say information is missing merely
    because the question is short.

    For example, if the user asks:

    country

    search the context for country,
    location, city, address, nationality,
    or other geographical information.

11. Language requirement:

{language_instruction}

12. When answering in Arabic, translate
    the answer accurately while preserving
    the original meaning.

13. When answering in Both languages,
    provide the same factual information
    in both languages.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

        # --------------------------------------------------
        # Call LLM
        # --------------------------------------------------

        llm = create_llm()

        response = llm.invoke(
            prompt
        )

        answer = response.content.strip()

        # --------------------------------------------------
        # Remove accidental markdown fences
        # --------------------------------------------------

        if answer.startswith("```"):

            lines = answer.splitlines()

            if len(lines) >= 3:

                answer = "\n".join(
                    lines[1:-1]
                ).strip()

        # --------------------------------------------------
        # Create unique sources
        # --------------------------------------------------

        unique_sources = []

        seen_sources = set()

        for document in documents:

            source = document.metadata.get(
                "source",
                "Unknown document"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            source_key = (
                str(source),
                str(page)
            )

            if source_key not in seen_sources:

                seen_sources.add(
                    source_key
                )

                unique_sources.append(
                    {
                        "source": source,
                        "page": page
                    }
                )

        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return {
            "answer": answer,
            "sources": unique_sources
        }

    except ValueError:
        raise

    except Exception as error:

        raise RuntimeError(
            f"RAG generation failed: {error}"
        )
