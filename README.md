# RAG System for Technical PDF Documents

A Retrieval-Augmented Generation (RAG) system built with LangChain and Qdrant to make technical manuals and annual reports searchable.

## Overview

This RAG system enhances LLM responses by retrieving relevant information from a knowledge base before generating answers. The system is specifically optimized for technical PDF documents and annual reports, allowing for efficient search and retrieval of specific financial and technical information.

## Features

- PDF document loading and preprocessing
- Intelligent text chunking with RecursiveCharacterTextSplitter
- Embedding generation using OpenAI (text-embedding-ada-002)
- Vector storage in Qdrant (using Docker)
- Semantic search with metadata filtering capabilities
- Context-aware response generation
- Financial summary generation
- Interactive command-line interface
- Streamlit chatbot interface with conversation memory

## Requirements

- Python 3.8+
- Docker and Docker Compose (for running Qdrant)
- OpenAI API key (for embeddings and generation)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Basic_RAG.git
   cd Basic_RAG
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `env_template` to `.env` and fill in your API keys
   
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   QDRANT_URL=http://localhost:6333  # Default Docker port mapping
   QDRANT_API_KEY=  # Only needed for secure Qdrant instances
   COLLECTION_NAME=company_reports
   EMBEDDING_MODEL=text-embedding-ada-002
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

## Setting up Qdrant with Docker

### Option 1: Using Docker Compose (Recommended)

A `docker-compose.yml` file is provided for easy setup. Run:

```bash
docker-compose up -d
```

This will start Qdrant in a container with proper volume mapping for data persistence.

### Option 2: Using Docker directly

If you prefer to use Docker commands directly:

```bash
docker pull qdrant/qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Verify Qdrant is running:
```bash
docker ps
```

You can access the Qdrant dashboard at: http://localhost:6334/dashboard

## Usage

### Quick Start

Use the provided startup scripts to automatically check Docker status, start Qdrant if needed, and launch the chatbot:

**On Windows:**
```
start_chatbot.bat
```

**On Linux/Mac:**
```
./start_chatbot.sh
```

The script will:
1. Check if Docker is running (and try to start it if it's not)
2. Verify the Qdrant container is running (and start it if needed)
3. Launch the Streamlit chatbot interface

### Ingesting Documents

Place your PDF documents in the `data` directory, then run:

```bash
python src/rag_app.py ingest --dir data
```

Additional options:
- `--no-metadata`: Skip metadata extraction

### Querying the System

```bash
python src/rag_app.py query --text "What was the revenue in Q2 2022?"
```

Additional options:
- `--k`: Number of documents to retrieve (default: 5)
- `--year`: Filter by specific year
- `--financial`: Filter for financial information only

### Generating Financial Summaries

```bash
python src/rag_app.py summary --year 2022
```

Additional options:
- `--k`: Number of documents to include (default: 10)

### Interactive Demo

For an interactive command-line experience:

```bash
python src/interactive_demo.py --data-dir data
```

If you've already ingested documents:

```bash
python src/interactive_demo.py --skip-ingestion
```

### Chatbot Interface

For a user-friendly web-based chatbot interface:

```bash
streamlit run src/chatbot_app.py
```

Features of the chatbot:
- Conversational memory that maintains context
- Document ingestion through the UI
- Advanced filtering options (year, financial information)
- Clear, chat-like interface
- Mobile-friendly responsive design

## System Architecture

The system consists of the following components:

1. **Document Processor**: Loads and splits PDF documents into manageable chunks using RecursiveCharacterTextSplitter.
2. **Embeddings Generator**: Converts text chunks into vector embeddings using OpenAI's text-embedding-ada-002 model.
3. **Vector Store**: Manages storage and indexing of document embeddings in Qdrant, with Docker for easy deployment.
4. **Retriever**: Fetches relevant documents based on query similarity and metadata filters.
5. **Generator**: Produces high-quality responses using retrieved documents as context.
6. **Conversation Memory**: Maintains chat history for contextual conversation.
7. **RAG Application**: Integrates all components into a cohesive system.
8. **Streamlit Interface**: Provides a web-based chatbot experience.

## Optimizations for Technical Manuals

- Special metadata extraction for financial information
- Year-based filtering for temporal queries
- Financial summary generation functionality
- Customized chunking for complex technical content
- Source tracking for accurate citation
- Configurable vector dimensions (1536 for OpenAI embeddings)
- Conversation memory for more natural interactions

## Performance Considerations

- The system uses batched processing to handle large document collections
- Document chunks are processed in parallel where possible
- Metadata indexing improves filter-based queries
- Configurable chunk sizes to balance context retention and processing efficiency
- Direct Qdrant client API usage for better performance
- Streamlit caching for improved UI responsiveness

## Extending the System

- Add support for additional document formats (DOCX, HTML, etc.)
- Implement custom embedding models for domain-specific content
- Add evaluation metrics to measure retrieval and generation quality
- Integrate with other communication platforms (Slack, Discord, etc.)
- Expand the UI with document visualization and analytics

## License

[MIT License](LICENSE) 