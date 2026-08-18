import os
import tempfile

import streamlit as st

from src.document_loader import load_document
from src.text_splitter import split_documents
from src.embedding import (
    create_embedding_model,
    create_embeddings
)
from src.vector_store import create_vector_store
from src.rag_pipeline import generate_answer
from src.summarizer import generate_summary
from src.risk_analyzer import analyze_risks


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Business Document Intelligence Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e6e6e6;
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .sidebar-description {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 20px;
    }

    /* Application header */
    .app-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 25px;
    }

    /* Result cards */
    .result-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        background: white;
    }

    .risk-high {
        border-left: 5px solid #dc2626;
    }

    .risk-medium {
        border-left: 5px solid #f59e0b;
    }

    .risk-low {
        border-left: 5px solid #16a34a;
    }

    /* Small labels */
    .section-label {
        font-size: 13px;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "processed" not in st.session_state:
    st.session_state.processed = False


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():
    return create_embedding_model()


embedding_model = get_embedding_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">📄 Document Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-description">'
        'Upload and analyze your business documents.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">Document</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=[
            "pdf",
            "docx",
            "txt"
        ],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.caption(
            f"📄 {uploaded_file.name} "
            f"· {uploaded_file.size / 1024:.1f} KB"
        )

    # --------------------------------------------------------
    # Process document
    # --------------------------------------------------------

    process_button = st.button(
        "⚙️ Process Document",
        use_container_width=True,
        type="primary"
    )

    # --------------------------------------------------------
    # Sidebar status
    # --------------------------------------------------------

    st.divider()

    if st.session_state.processed:

        st.success(
            "Document ready"
        )

    else:

        st.info(
            "Upload a document to begin."
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <h1 style="
        font-size: 2rem;
        white-space: nowrap;
        margin-bottom: 0.5rem;
    ">
        📄 Business Document Intelligence Platform
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'AI-powered summarization, document Q&A, and risk analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if process_button:

    if uploaded_file is None:

        st.warning(
            "Please upload a document first."
        )

    else:

        try:

            file_extension = os.path.splitext(
                uploaded_file.name
            )[1].lower()

            # ------------------------------------------------
            # Save with correct extension
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                document_path = temp_file.name

            # ------------------------------------------------
            # Process
            # ------------------------------------------------

            with st.spinner(
                "Processing document and creating knowledge base..."
            ):

                documents = load_document(
                    document_path
                )

                if not documents:

                    st.error(
                        "No readable text was found in the document."
                    )

                    st.stop()

                # ------------------------------------------------
                # Source metadata
                # ------------------------------------------------

                for doc in documents:

                    doc.metadata["source"] = (
                        uploaded_file.name
                    )

                # ------------------------------------------------
                # Split
                # ------------------------------------------------

                chunks = split_documents(
                    documents
                )

                if not chunks:

                    st.error(
                        "No document chunks were created."
                    )

                    st.stop()

                # ------------------------------------------------
                # Embeddings
                # ------------------------------------------------

                embeddings = create_embeddings(
                    chunks,
                    embedding_model
                )

                # ------------------------------------------------
                # FAISS
                # ------------------------------------------------

                create_vector_store(
                    chunks,
                    embeddings
                )

                # ------------------------------------------------
                # Store
                # ------------------------------------------------

                st.session_state.documents = documents

                st.session_state.uploaded_filename = (
                    uploaded_file.name
                )

                st.session_state.processed = True

            st.success(
                "Document processed successfully."
            )

        except Exception as error:

            st.error(
                f"Document processing failed: {error}"
            )


# ============================================================
# MAIN WORKSPACE
# ============================================================

if st.session_state.documents:

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    tab1, tab2, tab3 = st.tabs(
        [
            "📄 Summary",
            "💬 Ask Questions",
            "⚠ Risk Analysis"
        ]
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    with tab1:

        st.subheader("Document Summary")

        st.caption(
            f"📄 {st.session_state.uploaded_filename}"
        )

        if st.button(
            "Generate Summary",
            key="summary_button",
            type="primary"
        ):

            try:

                with st.spinner("Generating summary..."):

                    summary = generate_summary(
                        st.session_state.documents,
                        language="en"
                    )

                # Clean result — no box
                st.markdown(summary)

            except Exception as error:

                st.error(
                    f"Summary generation failed: {error}"
                )


    # ========================================================
    # Q&A
    # ========================================================

    with tab2:

        st.subheader("Ask Questions")

        st.caption(
            "Ask anything about the uploaded document."
        )

        question = st.text_input(
            "Question",
            placeholder="e.g. What is the annual rent?",
            label_visibility="collapsed"
        )

        if question.strip():

            try:

                with st.spinner("Searching the document..."):

                    result = generate_answer(
                        question=question,
                        embedding_model=embedding_model,
                        top_k=5,
                        language="en"
                    )

                # ------------------------------------------------
                # Validate result
                # ------------------------------------------------

                if not isinstance(result, dict):

                    st.error(
                        "The Q&A system returned an invalid response."
                    )

                else:

                    answer = result.get(
                        "answer",
                        "Information not found in the document."
                    )

                    sources = result.get(
                        "sources",
                        []
                    )

                    # ------------------------------------------------
                    # Answer
                    # ------------------------------------------------

                    st.markdown("### Answer")

                    st.write(answer)

                    # ------------------------------------------------
                    # Sources
                    # ------------------------------------------------

                    if sources:

                        source_pages = {}

                        for source in sources:

                            if not isinstance(source, dict):
                                continue

                            filename = source.get(
                                "source",
                                "Unknown document"
                            )

                            page = source.get(
                                "page",
                                "Unknown"
                            )

                            if filename not in source_pages:

                                source_pages[filename] = []

                            if page not in source_pages[filename]:

                                source_pages[filename].append(page)

                        if source_pages:

                            st.markdown("### Sources")

                            for filename, pages in source_pages.items():

                                pages = sorted(
                                    pages,
                                    key=lambda value: (
                                        int(value)
                                        if str(value).isdigit()
                                        else str(value)
                                    )
                                )

                                pages_text = ", ".join(
                                    str(page)
                                    for page in pages
                                )

                                st.caption(
                                    f"📄 {filename} · Page(s): {pages_text}"
                                )

            except Exception as error:

                st.error(
                    f"Question answering failed: {error}"
                )


    # ========================================================
    # RISK ANALYSIS
    # ========================================================

    with tab3:

        st.subheader("⚠ Risk Analysis")

        if st.button(
            "Analyze Risks",
            type="primary"
        ):

            try:

                with st.spinner("Analyzing document..."):

                    result = analyze_risks(
                        st.session_state.documents,
                        language="en"
                    )

                # --------------------------------------------------
                # Validate result
                # --------------------------------------------------

                if not isinstance(result, dict):

                    st.error(
                        "Risk analyzer returned an unexpected response."
                    )

                    st.stop()

                if result.get("error"):

                    st.error(
                        result["error"]
                    )

                    st.stop()

                # --------------------------------------------------
                # Get data
                # --------------------------------------------------

                risks = result.get(
                    "risks",
                    []
                )

                clauses = result.get(
                    "important_clauses",
                    []
                )

                if not isinstance(risks, list):
                    risks = []

                if not isinstance(clauses, list):
                    clauses = []

                risks = [
                    risk
                    for risk in risks
                    if isinstance(risk, dict)
                ]

                clauses = [
                    clause
                    for clause in clauses
                    if isinstance(clause, dict)
                ]

                # ==================================================
                # SUMMARY
                # ==================================================

                high = 0
                medium = 0
                low = 0

                for risk in risks:

                    level = str(
                        risk.get(
                            "level",
                            ""
                        )
                    ).lower().strip()

                    if level in [
                        "high",
                        "مرتفع",
                        "عالي"
                    ]:

                        high += 1

                    elif level in [
                        "medium",
                        "متوسط"
                    ]:

                        medium += 1

                    elif level in [
                        "low",
                        "منخفض"
                    ]:

                        low += 1

                st.markdown(
                    f"**{len(risks)} risks identified**  "
                    f"· 🔴 High: {high}  "
                    f"· 🟠 Medium: {medium}  "
                    f"· 🟢 Low: {low}"
                )

                st.divider()

                # ==================================================
                # RISKS
                # ==================================================

                if not risks:

                    st.success(
                        "No significant risks identified."
                    )

                else:

                    for risk in risks:

                        title = str(
                            risk.get(
                                "title",
                                "Unknown Risk"
                            )
                        )

                        level = str(
                            risk.get(
                                "level",
                                "Unknown"
                            )
                        )

                        description = str(
                            risk.get(
                                "description",
                                ""
                            )
                        )

                        evidence = str(
                            risk.get(
                                "evidence",
                                ""
                            )
                        )

                        page = risk.get(
                            "page",
                            "Unknown"
                        )

                        # ------------------------------------------
                        # Risk title
                        # ------------------------------------------

                        st.markdown(
                            f"**⚠ {title}**"
                        )

                        # ------------------------------------------
                        # Risk level
                        # ------------------------------------------

                        level_lower = level.lower().strip()

                        if level_lower in [
                            "high",
                            "مرتفع",
                            "عالي"
                        ]:

                            st.markdown(
                                "🔴 **High Risk**"
                            )

                        elif level_lower in [
                            "medium",
                            "متوسط"
                        ]:

                            st.markdown(
                                "🟠 **Medium Risk**"
                            )

                        elif level_lower in [
                            "low",
                            "منخفض"
                        ]:

                            st.markdown(
                                "🟢 **Low Risk**"
                            )

                        else:

                            st.markdown(
                                f"**Risk Level:** {level}"
                            )

                        # ------------------------------------------
                        # Description
                        # ------------------------------------------

                        if description:

                            st.write(
                                description
                            )

                        # ------------------------------------------
                        # Evidence
                        # ------------------------------------------

                        if evidence:

                            st.caption(
                                f"Evidence: {evidence}"
                            )

                        st.caption(
                            f"Page {page}"
                        )

                        st.divider()

                # ==================================================
                # IMPORTANT CLAUSES
                # ==================================================

                if clauses:

                    st.markdown(
                        "**Important Clauses**"
                    )

                    clause_data = []

                    for clause in clauses:

                        title = str(
                            clause.get(
                                "title",
                                ""
                            )
                        )

                        value = str(
                            clause.get(
                                "value",
                                ""
                            )
                        )

                        page = clause.get(
                            "page",
                            "Unknown"
                        )

                        clause_data.append(
                            {
                                "Clause": title,
                                "Value": value,
                                "Page": page
                            }
                        )

                    if clause_data:

                        st.dataframe(
                            clause_data,
                            use_container_width=True,
                            hide_index=True
                        )

            except Exception as error:

                st.error(
                    f"Risk analysis failed: {error}"
                )