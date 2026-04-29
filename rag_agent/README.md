# Local Context-Aware RAG Agent

A hybrid Retrieval-Augmented Generation (RAG) system combining local document understanding with real-time external knowledge via tool-augmented agents.

---

## Overview

This project implements a locally deployed AI agent capable of answering queries using both **ingested documents (PDFs)** and **live web data**.

The system uses a vector database (ChromaDB) for semantic retrieval and extends its capabilities through **tool-based augmentation (MCP)**, enabling dynamic information retrieval when local context is insufficient.

Built using LangGraph, the agent supports multi-step reasoning, tool calling, and persistent contextual memory.

---

## Key Features

* **Local LLM Execution** (via Ollama – LLaMA 3.2)
* **Semantic Retrieval (RAG)** using ChromaDB
* **Agentic Workflow** with LangGraph (multi-step reasoning + tool use)
* **Hybrid Knowledge System**:

  * Local PDFs (static knowledge)
  * Web search + content extraction (dynamic knowledge)
* **Dual Interface**:

  * CLI-based querying
  * Web UI via FastAPI
* **Persistent Memory** from tool outputs

---

## Pipeline Overview

1. Documents are loaded from `/data`
2. Text is chunked and embedded into ChromaDB
3. User query is processed by a LangGraph agent
4. Relevant context is retrieved from the vector store
5. If insufficient context:

   * Agent triggers external tools (web search / content extraction)
6. Combined context is passed to the LLM
7. Final response is generated and returned via CLI or API

---

## Project Structure

```
main.py          # CLI execution + ingestion
api.py           # FastAPI server + chat endpoint

src/
  ingestion/     # PDF loading and preprocessing
  retrieval/     # Vector DB and similarity search
  generation/    # RAG chain / response generation
  agents/        # LangGraph agent logic
  tools/         # MCP tools (web search, scraping)
  utils/         # helper functions

frontend/        # Chat UI

data/            # Input documents (ignored)
chroma_db/       # Vector DB storage (ignored)
```

---

## External Tool Integration

The system integrates external tools via MCP (Model Context Protocol):

* **Web Search (Exa API)**
* **Page Content Extraction**

### Purpose

* Extend knowledge beyond local documents
* Enable real-time information retrieval
* Improve response relevance and coverage

---

## Example

**Query:**
"What are the key findings in the document?"

**Response:**
"The document focuses on X, highlighting Y and concluding Z..."

*(Example shortened for demonstration)*

---

## Getting Started

### Requirements

* Python 3.9+
* Ollama installed

```bash
ollama run llama3.2
```

### Installation

```bash
pip install -r requirements.txt
```

### Run (CLI)

```bash
python main.py --query "Your question here"
```

### Run (Web)

```bash
python api.py
```

Open: http://localhost:8000

---

## Tech Stack

* LangGraph (agent orchestration)
* Ollama (local LLM execution)
* ChromaDB (vector storage)
* FastAPI (backend API)

---

## Key Learnings

* Designing hybrid RAG + agent systems
* Integrating external tools into LLM workflows
* Managing trade-offs between local and dynamic knowledge
* Handling retrieval limitations and fallback strategies

---

## Disclaimer

For educational purposes only. Not intended for medical or professional advice.
