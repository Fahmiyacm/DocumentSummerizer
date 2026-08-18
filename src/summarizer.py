from src.llm import create_llm


def generate_summary(documents, language="en"):
    """
    Generate a concise business document summary.

    language:
        en   -> English
        ar   -> Arabic
        both -> English + Arabic
    """

    try:

        if not documents:
            raise ValueError(
                "No documents available for summarization."
            )

        # -----------------------------------------
        # Combine document text
        # -----------------------------------------

        full_text = "\n\n".join(
            document.page_content
            for document in documents
            if document.page_content.strip()
        )

        if not full_text.strip():
            raise ValueError(
                "Document contains no readable text."
            )

        # -----------------------------------------
        # Language instruction
        # -----------------------------------------

        if language == "ar":

            language_instruction = """
Write the entire summary in Arabic.

Use clear, professional Modern Standard Arabic.

ALL headings, explanations, and bullet points
must be written in Arabic.

Do NOT provide an English translation.

Keep names, company names, technical terms,
currency codes, document numbers, and dates
unchanged when appropriate.
"""

        elif language == "both":

            language_instruction = """
Provide the summary 

First provide the complete English summary.

Then provide the complete Arabic summary.

The Arabic version must contain the same
information as the English version.

Do not add information that is not present
in the document.
"""

        else:

            language_instruction = """
Write the entire summary in English.

Use clear and professional business English.
"""

        # -----------------------------------------
        # Prompt
        # -----------------------------------------

        prompt = f"""
You are a professional Business Document
Intelligence Assistant.

Create a VERY SHORT and CLEAR summary
of the provided document.

First identify what type of document it is.

Then explain what the document is mainly about.

Extract ONLY the most important information.

The document may be:

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
- Other business document

For a CV:
Focus on professional profile, experience,
education, certifications, and important projects.

For a Contract:
Focus on parties, purpose, contract period,
important financial terms, responsibilities,
and important conditions.

For a Report:
Focus on purpose, major findings,
important numbers, and conclusions.

For an Invoice:
Focus on supplier/customer, invoice purpose,
total amount, dates, and payment information.

For a Policy:
Focus on purpose, scope, major requirements,
and important obligations.

For other documents:
Identify the information most important
for understanding the document.

IMPORTANT:

- Do not repeat the document.
- Do not list every date.
- Do not list every responsibility.
- Do not include irrelevant information.
- Do not invent information.
- Do not use outside knowledge.
- Do not create unnecessary sections.
- Keep the summary concise.

Maximum 80-100 words for EACH language version.

LANGUAGE:

{language_instruction}

FORMAT:

For English:

### Document Type

[One short line]

### Summary

[2-3 concise sentences]

### Key Information

- [Important point]
- [Important point]
- [Important point]
- [Important point]

For Arabic, use Arabic headings.

For Both:
provide the complete English version first,
then the complete Arabic version.

DOCUMENT:

{full_text}

SUMMARY:
"""

        # -----------------------------------------
        # LLM
        # -----------------------------------------

        llm = create_llm()

        response = llm.invoke(prompt)

        return response.content.strip()

    except ValueError:
        raise

    except Exception as error:

        raise RuntimeError(
            f"Summary generation failed: {error}"
        )