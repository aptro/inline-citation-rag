# Inline Citation Demo

A CLI tool for question-answering with inline citations using Qdrant vector database and FastEmbed embeddings.

## Setup

1. Install dependencies:
   ```bash
   uv pip install -e .
   ```

2. Start Qdrant:
   ```bash
   docker-compose up -d
   ```

3. Set OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Usage

```bash
python citation_cli.py
```