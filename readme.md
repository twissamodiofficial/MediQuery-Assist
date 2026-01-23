---
title: MedQuery-Assist
app_file: main.py
sdk: gradio
sdk_version: 6.4.0
---
# Medical Assistant Chatbot

A conversational AI medical assistant that supports text, voice, and document-based interactions. Built with LangGraph, RAG, and Gradio.

## Live at: https://huggingface.co/spaces/twissamodi/MedQuery-Assist

## Features

- **Multi-modal Input**: Text, voice (Whisper), and PDF document upload
- **RAG System**: Store and retrieve patient medical records from PDF documents
- **Web Search**: Access latest medical information via Google Serper API
- **Conversational Memory**: Maintains context across conversation using LangGraph checkpointing
- **ReAct Framework**: Step-by-step reasoning with tool usage
- **Auto-transcription**: Voice messages automatically transcribed and sent

## Architecture

```
├── rag_setup.py          # Document processing and vector store
├── tools.py              # Medical history search and web search tools
├── graph_setup.py        # LangGraph workflow configuration
├── prompts.py            # System prompts
├── chat_handler.py       # Chat logic and session management
├── audio_handler.py      # Audio transcription
├── main.py               # Gradio interface
└── data/
    ├── patient_record_db/    # Chroma vector store
    └── long_term_memory.db   # SQLite conversation checkpoints
```

## Installation

```bash
pip install langgraph langchain-huggingface langchain-community langchain-chroma langgraph-checkpoint-sqlite langchain
pip install gradio transformers torch
pip install sentence-transformers
```

## Environment Setup

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=your_hf_token
SERPER_API_KEY=your_serper_key
```

## Usage

```bash
python main.py
```

Access the interface at `http://127.0.0.1:7860`

## How It Works

### 1. Document Upload
- Upload PDF medical records
- Documents are chunked, embedded, and stored in Chroma vector database
- Duplicate detection via file hashing

### 2. Query Processing
- User queries are processed through LangGraph workflow
- LLM decides which tools to use (medical history search or web search)
- Multi-step reasoning follows ReAct pattern

### 3. Voice Input
- Record audio via microphone
- Automatic transcription using Whisper-small
- Auto-send to chat after transcription

### 4. Response Generation
- DeepSeek-V3 model generates responses
- Can make multiple tool calls per query
- Maintains conversation context via SQLite checkpointing

## Components

### RAG_Setup
- Embeddings: `sentence-transformers/all-mpnet-base-v2`
- Vector Store: Chroma with persistence
- Chunk size: 1000 characters
- Similarity search returns top 5 results

### GraphSetup
- LLM: DeepSeek-V3 via HuggingFace Inference
- Max tokens: 1024
- Recursion limit: 25
- Memory: SQLite checkpointing

### Tools
- `check_medical_history`: Searches patient records
- `web_search`: Google Serper API for medical information

### AudioHandler
- Model: `openai/whisper-small`
- Auto-send after transcription
- Clears audio input after processing

## Session Management

- Each application instance generates a unique session ID
- All users in the same instance share conversation history
- Restart application to create new session

## File Structure

```
data/
├── patient_record_db/           # Vector embeddings
│   └── chroma.sqlite3
└── long_term_memory.db          # Conversation checkpoints
```

## Limitations

- Single global session (all users share history)
- SQLite connection with `check_same_thread=False` (thread safety concern)
- No user authentication
- File uploads not validated beyond extension
- No cleanup of uploaded temporary files

## Example Queries

**Simple Query:**
```
What medications am I taking?
```

**Complex Query:**
```
Can I take ibuprofen with my current medications?
```

**Upload Flow:**
1. Upload PDF medical record
2. System confirms upload success
3. Ask questions about the uploaded document

## Dependencies

- langgraph
- langchain-huggingface
- langchain-community
- langchain-chroma
- gradio
- transformers
- sentence-transformers
- pypdf
- google-serper-api
- python-dotenv

## Notes

- Requires active internet for HuggingFace Inference API
- Requires Serper API key for web search
- First run downloads embedding model (~400MB)
- Whisper model downloads on first audio transcription (~500MB)
