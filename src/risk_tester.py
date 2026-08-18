from src.document_loader import load_document
from src.risk_analyzer import analyze_risks


PDF_PATH = r"C:\Users\fahmi\PycharmProjects\DocumentSummery\data\sample_uae_rental_contract.pdf"


documents = load_document(PDF_PATH)

result = analyze_risks(
    documents,
    language="English"
)

print("=" * 60)
print("RESULT TYPE:")
print(type(result))

print("=" * 60)
print("RESULT:")
print(result)

print("=" * 60)

if isinstance(result, dict):

    print("RISKS:")
    print(result.get("risks", []))

    print()

    print("IMPORTANT CLAUSES:")
    print(
        result.get(
            "important_clauses",
            []
        )
    )

else:

    print("ERROR: analyze_risks() returned a string.")

print(analyze_risks)
print(analyze_risks.__module__)
print(analyze_risks.__code__.co_varnames)