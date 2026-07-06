from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "data" / "fia_2026_formula_1_technical_regulations_issue_8_-_2024-06-24-5.pdf"
DEFAULT_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

WEB_URLS = [
    "https://en.wikipedia.org/wiki/2023_Monaco_Grand_Prix",
    "https://en.wikipedia.org/wiki/Max_Verstappen",
    "https://en.wikipedia.org/wiki/Lewis_Hamilton",
    "https://en.wikipedia.org/wiki/Michael_Schumacher",
    "https://en.wikipedia.org/wiki/Pit_stop",
    "https://en.wikipedia.org/wiki/Formula_One",
    "https://en.wikipedia.org/wiki/Formula_One_car",
]

LOCAL_FALLBACK_TEXT = """
Formula 1 is the highest class of international single-seater racing. Race outcomes
depend on qualifying position, tire management, pit strategy, car performance,
driver execution, weather, safety cars, and reliability.

Max Verstappen won the 2023 Monaco Grand Prix after taking pole position and
controlling track position on a circuit where overtaking is extremely difficult.
Late rain made tire choice important, and Verstappen switched to intermediate
tires while avoiding major mistakes.

A Formula 1 pit stop is a strategic stop to change tires or repair damage.
Pit strategy depends on tire degradation, pit-lane time loss, traffic, safety-car
risk, and the performance difference between tire compounds.

Formula 1 technical regulations define the permitted dimensions, bodywork,
aerodynamics, safety structures, power-unit requirements, and other design limits
for cars.
"""


def load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _import_langchain() -> dict[str, Any]:
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_groq import ChatGroq
        from langchain_core.prompts import PromptTemplate
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Module 5 dependencies are missing. Install: faiss-cpu groq "
            "langchain-core langchain-community langchain-text-splitters "
            "langchain-groq langchain-huggingface sentence-transformers pypdf "
            "requests beautifulsoup4"
        ) from exc

    return {
        "Document": Document,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
        "FAISS": FAISS,
        "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
        "ChatGroq": ChatGroq,
        "PromptTemplate": PromptTemplate,
    }


def load_website_document(url: str):
    libs = _import_langchain()
    Document = libs["Document"]
    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError:
        return None

    headers = {"User-Agent": "Mozilla/5.0 F1 Capstone RAG API"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "sup"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else url
    paragraphs = [
        item.get_text(" ", strip=True)
        for item in soup.find_all(["p", "li", "h1", "h2", "h3"])
    ]
    text = clean_text(" ".join(paragraphs))
    return Document(
        page_content=text,
        metadata={"source": url, "title": title, "type": "website"},
    )


def load_pdf_documents(pdf_path: Path, max_pages: int | None = None):
    libs = _import_langchain()
    Document = libs["Document"]
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Module 5 PDF dependency is missing. Install: pypdf") from exc

    if not pdf_path.exists():
        return []

    reader = PdfReader(str(pdf_path))
    documents = []
    pages_to_read = len(reader.pages) if max_pages is None else min(len(reader.pages), max_pages)
    for page_number in range(pages_to_read):
        text = clean_text(reader.pages[page_number].extract_text() or "")
        if len(text) < 50:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "title": "FIA 2026 Formula 1 Technical Regulations",
                    "type": "pdf",
                    "page": page_number + 1,
                },
            )
        )
    return documents


def _fallback_document():
    libs = _import_langchain()
    Document = libs["Document"]
    return Document(
        page_content=clean_text(LOCAL_FALLBACK_TEXT),
        metadata={"source": "curated_notes", "title": "Formula 1 race knowledge base", "type": "knowledge"},
    )


def load_documents():
    documents = []
    for url in WEB_URLS:
        try:
            doc = load_website_document(url)
            if doc is not None and len(doc.page_content) > 200:
                documents.append(doc)
        except Exception:
            continue

    try:
        documents.extend(load_pdf_documents(PDF_PATH))
    except RuntimeError:
        pass
    if not documents:
        documents.append(_fallback_document())
    return documents


def format_source(doc) -> str:
    source_type = doc.metadata.get("type", "source")
    title = doc.metadata.get("title", "Untitled")
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page")
    if page:
        return f"{title}, page {page}"
    if source_type == "knowledge":
        return title
    return f"{title} | {source}"


def format_docs(docs) -> str:
    formatted = []
    for doc in docs:
        source = format_source(doc)
        formatted.append(f"Source: {source}\nContent: {doc.page_content}")
    return "\n\n---\n\n".join(formatted)


@lru_cache(maxsize=1)
def load_rag_components():
    libs = _import_langchain()
    splitter = libs["RecursiveCharacterTextSplitter"](
        chunk_size=1000,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = load_documents()
    chunks = splitter.split_documents(documents)
    embeddings = libs["HuggingFaceEmbeddings"](model_name=EMBEDDING_MODEL)
    vector_db = libs["FAISS"].from_documents(chunks, embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 6})


def _extractive_answer(question: str, retrieved_docs) -> str:
    best_context = clean_text(" ".join(doc.page_content for doc in retrieved_docs[:2]))
    if len(best_context) > 1200:
        best_context = best_context[:1200].rsplit(" ", 1)[0] + "..."
    return (
        "Groq is not configured, so this is an extractive answer from the retrieved Module 5 context.\n\n"
        f"{best_context}"
    )


def ask_f1_chatbot(question: str, model_name: str = DEFAULT_MODEL, show_sources: bool = True) -> dict[str, Any]:
    load_env_file()
    retriever = load_rag_components()
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        libs = _import_langchain()
        llm = libs["ChatGroq"](model=model_name, temperature=0.2)
        prompt = libs["PromptTemplate"].from_template(
            """
You are a Formula 1 expert assistant for a capstone project.
Use the retrieved context to answer the user's question.

Answer style:
- Start with a direct 1-2 sentence answer.
- Then give 3-5 neat bullet points only when reasons, steps, or factors are useful.
- Use a compact table only when the user explicitly asks to compare two or more things.
- Never write filler such as "Comparison table not applicable" or "not applicable in this case".
- Do not add a separate "Data source:" line inside the answer.
- If the retrieved source is the Formula 1 race knowledge base, call it "F1 knowledge base".
- Do not print raw source labels such as LOCAL, local_fallback, PDF, or pipe-separated metadata.
- Do not invent facts that are not supported by the context.

Context:
{context}

Question:
{question}

Neat answer:
"""
        )
        response = llm.invoke(prompt.format(context=context, question=question))
        answer = response.content
        mode = "rag_groq"
    else:
        answer = _extractive_answer(question, retrieved_docs)
        mode = "rag_extractive"

    sources = []
    if show_sources:
        seen = set()
        for doc in retrieved_docs:
            source = format_source(doc)
            if source not in seen:
                sources.append(source)
                seen.add(source)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "mode": mode,
    }
