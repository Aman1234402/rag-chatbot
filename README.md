
# PDF RAG Chatbot

A Streamlit chatbot that answers questions using content extracted from `Ebook-Agentic-AI.pdf`. The application cleans the PDF text, splits it into chunks, creates embeddings, stores them in ChromaDB, and retrieves relevant passages before generating an answer.


## Features


- PDF text extraction and whitespace cleanup with `pdfplumber`
- Hugging Face embedding model: `BAAI/bge-small-en-v1.5`
- Persistent ChromaDB vector store
- Groq-powered answer generation
- Page citations and retrieval confidence for answers grounded in the PDF
- Out-of-scope questions return a clear not-found response without showing retrieval details
- Streamlit chat interface with conversation reset
- Automatic PDF indexing when the frontend starts


## Requirements

- Python 3.13 or newer
- A Groq API key
- A Hugging Face API token
- The supplied PDF file, `Ebook-Agentic-AI.pdf`


## Setup

Create and activate a virtual environment:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```
## Run the Frontend

Start the Streamlit application:

```powershell
.\env\Scripts\python.exe -m streamlit run frontend.py
```

When Streamlit starts, it automatically extracts the PDF, refreshes the ChromaDB collection, and initializes the chatbot. Open the local URL shown by Streamlit, usually `http://localhost:8501`.


## Run the backend Directly

To rebuild the vector store and ask a CLI question:

```powershell
.\env\Scripts\python.exe backend.py
```


## Project Structure

```text
backend.py                 PDF ingestion, embeddings, ChromaDB, and RAG pipeline
frontend.py                Streamlit chat interface
Ebook-Agentic-AI.pdf       Source PDF
requirements.txt           Python dependencies
chroma_db/                 Generated persistent vector database
```

The `chroma_db/` directory is generated locally and can be rebuilt from the PDF at any time.
## Sample Queries

Try these questions in the Streamlit chat:

 1. What is agentic AI?
 2. How is agentic AI different from traditional AI systems?
 3. What are the main capabilities of an AI agent?
 4. How do AI agents perceive and interact with their environment?
 5. What challenges are associated with building     reliable agentic AI systems?
 6. What industries can benefit from agentic AI?

Questions unrelated to the PDF return a not-found response instead of using outside knowledge.
## Architecture

1. `pdfplumber` extracts text from each PDF page.
2. The ingestion layer cleans whitespace and repairs words split across lines.
3. `RecursiveCharacterTextSplitter` divides the cleaned text into overlapping chunks and preserves page metadata.
4. Hugging Face `BAAI/bge-small-en-v1.5` creates an embedding for each chunk.
5. ChromaDB stores the embeddings and chunk metadata in the local `chroma_db/` directory.
6. A user question is embedded and matched against the most relevant chunks.
7. LangGraph sends the retrieved context to the Groq chat model, which answers only from that context and cites PDF pages.
8. Streamlit provides the chat interface and displays answers, confidence, and source chunks for in-document questions.

When `frontend.py` starts, it runs the ingestion pipeline once through Streamlit's cached resource initializer, then creates the chatbot over the refreshed Chroma collection.