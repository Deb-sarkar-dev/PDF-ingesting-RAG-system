# Hybrid RAG Agent with Web Search Fallback

A modular Retrieval-Augmented Generation (RAG) system that combines **local document understanding** with **real-time web search**, designed to improve factual accuracy and reduce hallucinations in Large Language Models.

---

## Why This Project Exists

Large Language Models (LLMs) are powerful but fundamentally limited:
- They hallucinate when context is missing
- They lack access to private or domain-specific documents
- They cannot reliably access real-time information

This project was built to address those limitations by creating a **hybrid knowledge system** that dynamically selects between local and external sources of information.

---

## Overview

This system is a locally deployed AI agent capable of answering queries using:

- **Ingested documents (PDFs stored locally)**
- **Semantic retrieval via vector database (ChromaDB)**
- **Real-time web search and content extraction**
- **Agent-based reasoning and tool selection (LangGraph)**

The system intelligently decides when local knowledge is sufficient and when external tools are required.

---

## Key Features

- Local LLM execution using **Ollama (LLaMA 3.2)**
- Semantic search using **ChromaDB vector database**
- Hybrid retrieval (local + web-based knowledge)
- Agentic workflow using **LangGraph**
- Tool-augmented reasoning with dynamic decision making
- Persistent context handling across multi-step queries
- Dual interface:
  - CLI-based interaction
  - FastAPI backend for web usage

---

## System Architecture

User Query
   ↓
LangGraph Agent (Decision Layer)
   ↓
Retrieval Layer
   ├── Vector DB (ChromaDB)
   └── Web Search + Extraction
   ↓
Context Aggregation
   ↓
LLM Processing (LLaMA via Ollama)
   ↓
Final Response (API / CLI)

---

## Project Structure

main.py        → CLI execution + ingestion pipeline  
api.py         → FastAPI server + chat endpoint  

src/
  agent/       → LangGraph orchestration  
  core/        → shared logic  
  llm/         → LLM interface (Ollama)  
  nodes/       → processing nodes  
  retrieval/   → vector DB + retrieval  

frontend/      → chat UI  
data/          → input docs  
chroma_db/     → vector storage  

---

## External Tool Integration

The system extends its capabilities using external tools via MCP (Model Context Protocol):

### Web Search (Exa API)
Used when local documents do not contain sufficient information.

### Content Extraction
Extracts and cleans relevant content from web pages.

### Purpose of Tools
- Extend knowledge beyond static documents
- Enable real-time information access
- Improve response completeness and reliability

---

## Example Workflow

**User Query:**  
"What are the key findings in the document?"

**System Behavior:**
- Retrieves relevant chunks from ChromaDB
- Evaluates whether context is sufficient
- Falls back to web search if needed
- Combines retrieved sources into a final prompt

**Response:**  
A context-aware answer generated using both local and external knowledge sources.

---

## Design Decisions

- **Local LLM (Ollama)**  
  Enables offline execution and privacy-focused inference.

- **Hybrid Retrieval System**  
  Combines semantic vector search with keyword/web search for maximum coverage.

- **Agent-Based Architecture (LangGraph)**  
  Allows multi-step reasoning, conditional tool use, and structured workflows.

- **Modular Design**  
  Each component (retrieval, LLM, tools) is independent and extensible.

- **Fallback Strategy**  
  Ensures the system can recover when local context is insufficient.

---

## Technical Highlights

- Built a hybrid RAG + agent-based AI system from scratch
- Integrated tool-calling for dynamic external knowledge retrieval
- Designed modular architecture for extensibility
- Implemented fallback mechanisms for missing or incomplete context
- Combined offline LLM execution with real-time web augmentation

---

## Example Use Case

- Querying private PDFs (notes, research papers, documents)
- Answering general knowledge questions with real-time accuracy
- Combining multiple knowledge sources into a single coherent response

---

## Tech Stack

- Python
- Ollama (LLaMA 3.2)
- LangGraph (agent orchestration)
- ChromaDB (vector database)
- FastAPI (backend API)
- Exa API (web search tool)

---

## Future Improvements

- Advanced reranking models for better retrieval quality
- Evaluation metrics (faithfulness, context relevance)
- Streaming responses for real-time interaction
- Improved caching and latency optimization
- Multi-agent collaboration system

---

## Disclaimer

This project is for educational and research purposes only. It is not intended for medical, legal, or professional decision-making use.
