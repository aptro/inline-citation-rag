#!/bin/bash

echo "Setting up Inline Citation Demo..."

# Start Qdrant with Docker
echo "Starting Qdrant vector database..."
docker compose up -d

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to be ready..."
sleep 5

# Check if Qdrant is running
if curl -s http://localhost:6333/health > /dev/null; then
    echo "✓ Qdrant is running"
else
    echo "✗ Qdrant failed to start. Please check Docker."
    exit 1
fi

echo ""
echo "Setup complete! You can now run:"
echo "  python citation_cli.py"
echo ""
echo "Make sure to set your OpenAI API key:"
echo "  export OPENAI_API_KEY='your-api-key'"