# Local Context-Aware RAG Agent

A robust, multi-tenant Local Retrieval-Augmented Generation (RAG) system built with Python, LangGraph, Ollama, and FastAPI.

## Overview
This project provides a complete environment for running a local RAG agent. It ingests data (such as PDF files) into a local Chroma vector database and uses Llama 3.2 via Ollama to answer queries based on the contextual data. The agent is built using LangGraph, allowing for multi-agent capabilities, sophisticated tool calling (MCP), and external fallback searches (like Wikipedia or live web data) when local context falls short.

## Features
- **Local LLM Execution**: Uses [Ollama](https://ollama.com/) running the `llama3.2` model to ensure complete privacy and offline capabilities.
- **RAG via ChromaDB**: Embeds and stores document data locally using ChromaDB for semantic retrieval.
- **Agentic Workflow**: Powered by LangGraph to support tool calling, state management, and multi-step reasoning.
- **CLI & Web Interfaces**:
  - Run asynchronous queries directly from the command line (`main.py`).
  - Spin up a local FastAPI server with a built-in React/HTML chat widget frontend (`api.py`).
- **Persistent Memory**: Retains and caches insights from fallback tools for future requests.

## Project Structure
- `main.py`: Entry point for standard command-line inference and PDF data ingestion.
- `api.py`: FastAPI server that mounts the web frontend and exposes the `/api/chat` API endpoint.
- `src/`: Core logic containing LangGraph agent setup, LLM provider wrappers, and Vector Store configurations.
- `data/`: The directory where you can drop your source PDF files or text documents for ingestion.
- `chroma_db/`: Local persistent storage for the Chroma vector database.
- `frontend/`: Contains the static files for the chat web UI (e.g., `index.html`, `chat-widget.js`).

## Getting Started

### Prerequisites
1. Python 3.9+
2. [Ollama](https://ollama.com/) installed and running on your local machine.
3. Make sure to pull the Llama model:
   ```bash
   ollama run llama3.2
   ```

### Installation
1. Clone the repository and navigate to the root directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**1. Command Line Execution:**
You can directly query the agent from the terminal. By default, it looks for PDFs in the `rag_agent/data` folder:
```bash
python main.py --query "What are the main topics in the documents?" --pdf_dir "rag_agent/data"
```

**2. Web Server & API:**
Start the FastAPI backend and serve the Web UI frontend:
```bash
python api.py
```
Then, open your browser and navigate to `http://localhost:8000` to interact with the agent via the chat widget.

## Built With
- [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/) - Framework for stateful agents.
- [Ollama](https://ollama.com/) - Run local LLMs.
- [Chroma](https://www.trychroma.com/) - The AI-native open-source embedding database.
- [FastAPI](https://fastapi.tiangolo.com/) - Web API framework.
