import json
import re

from src.llm import create_llm


def _extract_json(text):
    """
    Extract the first valid JSON object from LLM output.

    Handles:
    - ```json ... ```
    - Extra text before JSON
    - Extra text after JSON
    - Multiple JSON objects
    """

    if not text:
        raise ValueError("LLM returned an empty response.")

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*$",
        "",
        text
    )

    text = text.strip()

    # First attempt: entire response is JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object boundaries
    start = text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found in LLM response."
        )

    # Find balanced JSON object
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):

        char = text[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:

                json_text = text[
                    start:i + 1
                ]

                return json.loads(json_text)

    raise ValueError(
        "Incomplete JSON object returned by LLM."
    )


def analyze_risks(
    documents,
    language="en"
):
    """
    Analyze a business document for genuine risks
    and important clauses.

    language:
        en   -> English
        ar   -> Arabic
        both -> English + Arabic
    """

    try:

        if not documents:
            raise ValueError(
                "No documents available for risk analysis."
            )

        # --------------------------------------------------
        # Build document context
        # --------------------------------------------------

        context_parts = []

        for document in documents:

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            text = document.page_content.strip()

            if text:

                context_parts.append(
                    f"===== PAGE {page} =====\n{text}"
                )

        context = "\n\n".join(
            context_parts
        )

        if not context.strip():
            raise ValueError(
                "No text available for risk analysis."
            )

        # --------------------------------------------------
        # Language instruction
        # --------------------------------------------------

        if language == "ar":

            language_instruction = """
Return all human-readable fields in Arabic:

- title
- level
- description
- evidence
- important clause title
- important clause value

Use professional Modern Standard Arabic.

Keep names, company names, document numbers,
currency codes, and dates unchanged when appropriate.

Do NOT provide an English explanation outside the JSON.
"""

        elif language == "both":

            language_instruction = """
Return both English and Arabic for every
human-readable field.

For example:

"title": "Late Payment Penalty | غرامة التأخر في الدفع"

"description": "Late payment may result in a penalty. | قد يؤدي التأخير في الدفع إلى غرامة."

"evidence": "A late payment may result in a penalty. | قد يؤدي التأخير في الدفع إلى غرامة."

"level": "Medium | متوسط"

Important clause titles and values should also
contain both English and Arabic where appropriate.

Do NOT provide explanations outside the JSON.
"""

        else:

            language_instruction = """
Return all human-readable fields in English.

Use clear and professional business English.
"""

        # --------------------------------------------------
        # Prompt
        # --------------------------------------------------

        prompt = f"""
You are a professional Business Document
Risk and Compliance Analyst.

Analyze ONLY the document provided below.

Your job is NOT to treat every contract condition
as a risk.

Identify ONLY genuine risks that are explicitly
supported by the document.

IMPORTANT:

A risk should represent something that could
reasonably cause:

- financial loss
- penalty
- liability
- contractual exposure
- operational problem
- compliance issue
- dispute
- missed obligation
- termination consequence

Do NOT automatically classify normal document
information as a risk.

For example, these are normally IMPORTANT CLAUSES,
not risks:

- rent amount
- security deposit
- contract dates
- payment schedule
- renewal terms
- normal maintenance responsibilities
- ordinary notice periods

Only classify them as risks if the document
explicitly describes a consequence, penalty,
liability, failure, dispute, or obligation
that creates meaningful exposure.

Never use outside knowledge.

Never invent a risk.

Every risk MUST have evidence from the document.

Every risk MUST include the correct page number
when available.

IMPORTANT:

If the document contains only normal contractual
conditions and no genuine risks, return:

"risks": []

Do not create artificial risks just to fill the list.

{language_instruction}

Return ONLY ONE valid JSON object.

DO NOT return:

- Markdown
- ```json
- explanations
- comments
- multiple JSON objects
- text before JSON
- text after JSON

Use EXACTLY this structure:

{{
    "risks": [
        {{
            "title": "Late Payment Penalty",
            "level": "Medium",
            "description": "Late payment may result in a penalty.",
            "evidence": "A late payment may result in a penalty.",
            "page": 1
        }}
    ],
    "important_clauses": [
        {{
            "title": "Annual Rent",
            "value": "AED 120,000",
            "page": 1
        }}
    ]
}}

Risk level MUST be one of:

- High
- Medium
- Low

If there are no genuine risks:

{{
    "risks": [],
    "important_clauses": []
}}

DOCUMENT:

{context}
"""

        # --------------------------------------------------
        # Call LLM
        # --------------------------------------------------

        llm = create_llm()

        response = llm.invoke(prompt)

        content = response.content

        if not content:
            raise ValueError(
                "LLM returned an empty response."
            )

        # --------------------------------------------------
        # Extract JSON safely
        # --------------------------------------------------

        result = _extract_json(
            content
        )

        # --------------------------------------------------
        # Validate top-level object
        # --------------------------------------------------

        if not isinstance(result, dict):

            raise ValueError(
                "LLM response is not a JSON object."
            )

        risks = result.get(
            "risks",
            []
        )

        important_clauses = result.get(
            "important_clauses",
            []
        )

        # --------------------------------------------------
        # Validate lists
        # --------------------------------------------------

        if not isinstance(risks, list):

            risks = []

        if not isinstance(
            important_clauses,
            list
        ):

            important_clauses = []

        # --------------------------------------------------
        # Clean risk objects
        # --------------------------------------------------

        clean_risks = []

        for risk in risks:

            if not isinstance(
                risk,
                dict
            ):
                continue

            clean_risks.append(
                {
                    "title": str(
                        risk.get(
                            "title",
                            "Unknown Risk"
                        )
                    ),

                    "level": str(
                        risk.get(
                            "level",
                            "Medium"
                        )
                    ),

                    "description": str(
                        risk.get(
                            "description",
                            ""
                        )
                    ),

                    "evidence": str(
                        risk.get(
                            "evidence",
                            ""
                        )
                    ),

                    "page": risk.get(
                        "page",
                        "Unknown"
                    )
                }
            )

        # --------------------------------------------------
        # Clean important clauses
        # --------------------------------------------------

        clean_clauses = []

        for clause in important_clauses:

            if not isinstance(
                clause,
                dict
            ):
                continue

            clean_clauses.append(
                {
                    "title": str(
                        clause.get(
                            "title",
                            ""
                        )
                    ),

                    "value": str(
                        clause.get(
                            "value",
                            ""
                        )
                    ),

                    "page": clause.get(
                        "page",
                        "Unknown"
                    )
                }
            )

        # --------------------------------------------------
        # Final result
        # --------------------------------------------------

        return {
            "risks": clean_risks,
            "important_clauses": clean_clauses
        }

    except ValueError as error:

        return {
            "risks": [],
            "important_clauses": [],
            "error": str(error)
        }

    except json.JSONDecodeError as error:

        return {
            "risks": [],
            "important_clauses": [],
            "error": (
                f"Invalid JSON returned by LLM: {error}"
            )
        }

    except Exception as error:

        return {
            "risks": [],
            "important_clauses": [],
            "error": (
                f"Risk analysis failed: {error}"
            )
        }