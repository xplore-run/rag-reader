# RagReader

A AI-powered chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on PDF documentation. Built with modern Python frameworks and designed for scalability and ease of use.

Note - The current status is ALPHA.

## Objective
The objective is to provide ready to use production grade RAG powered ChatBot.

### Use cases - 
1. Tech documenation once fed to chatbot can help onboard new developers.
2. Product Documentation once fed to chatbot can help onboard new product person help CSM, Clients with their queries.
3. Manual of any appliances can be fed and end users can ask any question and get answers easily.
4. The usecases are many and can be customised to make wonders.

## 🌟 Features

### Core Capabilities
- **📄 Advanced PDF Processing**: 
  - Multi-pdf-library support (PyPDF2, pdfplumber, PyMuPDF)
  - Table extraction and formatting
  - Automatic text cleaning and preprocessing
  - Page-level metadata preservation

- **🔤 Intelligent Text Chunking**:
  - Multiple chunking strategies (recursive, token-based, section-based)
  - Configurable chunk size and overlap
  - Context-aware splitting that preserves meaning
  - Neighbor chunk retrieval for enhanced context

- **🔍 Semantic Search**:
  - Vector similarity search using embeddings
  - Support for multiple embedding models
  - Relevance scoring and reranking
  - Metadata filtering capabilities

- **🤖 Multi-LLM Support**:
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude 3)
  - Local models via Ollama (todo - coming soon)
  - Easy switching between providers

- **💾 Vector Database Options**:
  - ChromaDB (default, local)
  - Pinecone - Todo Add support
  - FAISS - Todo Add support

### User Interfaces

1. **🖥️ Command Line Interface (CLI)**

2. **🌐 Web Interface (React App)**

3. **🚀 REST API (FastAPI)**

## 📋 Prerequisites

- Python 3.11 or higher
- 4GB+ RAM recommended
- API keys for LLM providers (OpenAI/Anthropic)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd chatbot-tech-rag

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY or ANTHROPIC_API_KEY
# Optional: PINECONE_API_KEY (if using Pinecone)
```

### 3. Add Documents

```bash
# Place PDF files in the train-data directory
cp /path/to/your/docs/*.pdf train-data/
```

### 4. Index Documents

```bash
# Run the indexing script
python scripts/index_documents.py

# With options
python scripts/index_documents.py --clear  # Clear existing index
python scripts/index_documents.py --config config/dev.yaml  --clear   --verify # Use specific config
python scripts/index_documents.py --verify  # Verify after indexing
```

### 5. Run the Chatbot

**Option A: Command Line Interface**
```bash
# Interactive mode
python -m src.main

# Single question mode
python -m src.main --config config/dev.yaml --question "what is XYZ?"

# With custom configuration
python -m src.main --config config/prod.yaml
```

## 🔧 Configuration

### Environment Variables (.env)

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_MODEL=claude-3-haiku-20240307

# Vector Store Configuration
VECTOR_STORE_TYPE=chromadb
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=tech_docs

# Embedding Configuration
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/chatbot.log
```

### YAML Configuration Files

The system supports YAML configuration files for different environments:

```yaml
# config/dev.yaml
embedding:
  type: sentence_transformer
  model_name: all-MiniLM-L6-v2

vector_store:
  type: chromadb
  persist_directory: ./chroma_db_dev

llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.7

rag:
  top_k: 5
  rerank: true
  include_neighbors: true
```

## 📁 Project Structure

```
chatbot-tech-rag/
├── src/
│   ├── pdf_processor/     # PDF extraction and text processing
│   │   ├── pdf_extractor.py    # Multi-backend PDF text extraction
│   │   └── text_chunker.py     # Intelligent text chunking
│   ├── embeddings/        # Embedding generation
│   │   ├── base.py            # Abstract base class
│   │   ├── sentence_transformer.py  # Local embeddings
│   │   ├── openai_embedding.py     # OpenAI embeddings
│   │   └── factory.py         # Embedding model factory
│   ├── vector_store/      # Vector database operations
│   │   ├── base.py           # Abstract base class
│   │   ├── chromadb_store.py # ChromaDB implementation
│   │   └── factory.py        # Vector store factory
│   ├── rag/              # RAG implementation
│   │   ├── retriever.py      # Document retrieval
│   │   └── generator.py      # Answer generation
│   ├── llm/              # LLM integrations
│   │   ├── base.py          # Abstract base class
│   │   ├── openai_client.py # OpenAI implementation
│   │   ├── anthropic_client.py # Anthropic implementation
│   │   └── factory.py       # LLM client factory
│   ├── cli/              # Command-line interface
│   │   └── chatbot_cli.py   # Rich CLI implementation
│   ├── api/              # REST API
│   │   └── app.py          # FastAPI application
│   └── utils/            # Utilities
│       ├── config.py       # Configuration management
│       └── logger.py       # Logging setup
├── scripts/              # Utility scripts
│   └── index_documents.py  # Document indexing script
├── config/               # Configuration files
│   ├── dev.yaml           # Development config
│   └── prod.yaml          # Production config
├── train-data/           # PDF documents directory
├── tests/                # Test suite
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
└── README.md            # This file
```

## Todo -

1. **Embedding Caching**: Cache frequently used embeddings
2. **Batch Processing**: Process multiple documents in parallel
3. **Async Operations**: Use async endpoints for better concurrency
4. **Model Selection**: Choose appropriate models for your use case
5. **Index Optimization**: Regularly optimize vector indices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support
- Create an issue for bug reports or feature requests

---
