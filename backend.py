"""PDF ingestion and LangGraph RAG pipeline."""

import os
import re
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import pdfplumber
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Ebook-Agentic-AI.pdf"
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pdf_docs_huggingface_bge_small"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def get_embeddings():
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not api_token:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN missing. Put it in your .env file."
        )

    return HuggingFaceEndpointEmbeddings(
        model=EMBED_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=api_token,
    )


def clean_extracted_text(text: str) -> str:
    """Normalize PDF layout whitespace and repair words split across lines."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_documents(pdf_path: str | Path) -> list[Document]:
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"{pdf.resolve()} not found")

    documents: list[Document] = []
    with pdfplumber.open(str(pdf)) as pdf_file:
        for page_num, page in enumerate(pdf_file.pages, start=1):
            text = clean_extracted_text(page.extract_text() or "")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf.name, "page": page_num},
                    )
                )
    return documents


def ingest_pdf() -> None:
    pdf = Path(PDF_PATH)
    if not pdf.exists():
        raise FileNotFoundError(f"{pdf.resolve()} not found")

    docs = extract_pdf_documents(pdf)
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = pdf.name
        chunk.metadata["page"] = chunk.metadata.get("page", 1)

    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    existing_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )
    existing_store.delete_collection()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Stored in ./{PERSIST_DIR} (collection: {COLLECTION_NAME})")


RUN_INGEST = __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ingest"

TOP_K = 4
# Chunks below this cosine relevance are treated as noise.
SCORE_THRESHOLD = 0.30
NOT_FOUND = "NOT_FOUND"

llm = None
vector_store = None

SYSTEM_PROMPT = f"""You are a question-answering assistant for a single PDF document.

Rules:
- Answer ONLY from the CONTEXT below. Never use outside knowledge.
- If the context does not contain the answer, reply with exactly: {NOT_FOUND}
- Do not guess, extrapolate, or fill gaps.
- Cite the page number(s) you used, like [p. 12], after the claims they support.
- Keep the answer focused and well structured.

CONTEXT:
{{context}}"""


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    context: list[dict]
    confidence: float


def _format_context(context: list[dict]) -> str:
    if not context:
        return "(no relevant passages found)"
    return "\n\n---\n\n".join(
        f"[chunk {c['chunk_id']} | page {c['page']} | score {c['score']:.3f}]\n{c['text']}"
        for c in context
    )


def retrieve_node(state: ChatState) -> dict:
    """Pull the top-k chunks and their relevance scores."""
    question = state["messages"][-1].content

    hits = vector_store.similarity_search_with_relevance_scores(question, k=TOP_K)

    context = [
        {
            "chunk_id": doc.metadata.get("chunk_id"),
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("source"),
            "score": round(float(score), 4),
            "text": doc.page_content,
        }
        for doc, score in hits
        if score >= SCORE_THRESHOLD
    ]

    scores = [c["score"] for c in context[:3]]
    retrieval_confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

    return {
        "question": question,
        "context": context,
        "confidence": retrieval_confidence,
    }


def generate_node(state: ChatState) -> dict:
    """Answer strictly from the retrieved context."""
    context = state["context"]

    prompt = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=_format_context(context))),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(prompt)

    # The model signalled the context was insufficient -> confidence is zero.
    if NOT_FOUND in response.content:
        response.content = (
            "I couldn't find an answer to that in the PDF."
        )
        return {"messages": [response], "confidence": 0.0}

    return {"messages": [response]}


def no_context_node(state: ChatState) -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "I couldn't find anything relevant to that in the PDF.",
            }
        ],
        "confidence": 0.0,
    }


def route(state: ChatState) -> str:
    return "generate" if state["context"] else "no_context"


def create_chatbot():
    global llm, vector_store

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY missing. Put it in your .env file.")

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )

    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_context", no_context_node)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve", route, {"generate": "generate", "no_context": "no_context"}
    )
    graph.add_edge("generate", END)
    graph.add_edge("no_context", END)

    return graph.compile(checkpointer=MemorySaver())


chatbot = None if RUN_INGEST or __name__ == "__main__" else create_chatbot()


if __name__ == "__main__":
    if RUN_INGEST:
        ingest_pdf()
    else:
        ingest_pdf()
        chatbot = create_chatbot()
        out = chatbot.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=input("Enter your query?")
                    )
                ]
            },
            config={"configurable": {"thread_id": "cli"}},
        )
        print(out["messages"][-1].content)
        print("\nconfidence:", out["confidence"])
        for c in out["context"]:
            print(f"  page {c['page']} | score {c['score']}")