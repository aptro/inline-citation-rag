#!/usr/bin/env python3
import json
import os
import sys
import argparse
from typing import List
from dataclasses import dataclass, field
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from fastembed import TextEmbedding
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn


@dataclass
class Article:
    title: str
    url: str
    text: str
    index: int
    chunks: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    article: Article
    chunk: str
    score: float
    chunk_index: int


class CitationAssistant:
    def __init__(
        self,
        articles_path: str,
        api_key: str = None,
        qdrant_url: str = "localhost:6333",
    ):
        self.console = Console()
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.collection_name = "articles"

        self.articles = self._load_articles(articles_path)
        self._setup_vector_db()

    def _load_articles(self, path: str) -> List[Article]:
        with open(path, "r") as f:
            data = json.load(f)

        articles = []
        for idx, article in enumerate(data["articles"]):
            articles.append(
                Article(
                    title=article["title"],
                    url=article["url"],
                    text=article["text"],
                    index=idx + 1,
                )
            )
        return articles

    def _chunk_text(
        self, text: str, max_chars: int = 2000, overlap: int = 200
    ) -> List[str]:
        chunks = []

        for i in range(0, len(text), max_chars - overlap):
            chunk = text[i : i + max_chars]
            chunks.append(chunk)

            if i + max_chars >= len(text):
                break

        return chunks

    def _setup_vector_db(self):
        try:
            self.qdrant_client.delete_collection(self.collection_name)
        except Exception:
            pass

        embedding_dim = 384  # BGE-small dimension
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

        self._index_articles()

    def _index_articles(self):
        points = []
        point_id = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Indexing articles...", total=len(self.articles))

            for article in self.articles:
                chunks = self._chunk_text(article.text)
                article.chunks = chunks

                chunk_embeddings = list(self.embedding_model.embed(chunks))

                for chunk_idx, (chunk, embedding) in enumerate(
                    zip(chunks, chunk_embeddings)
                ):
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding.tolist(),
                            payload={
                                "article_index": article.index,
                                "article_title": article.title,
                                "article_url": article.url,
                                "chunk_index": chunk_idx,
                                "chunk_text": chunk,
                            },
                        )
                    )
                    point_id += 1

                progress.update(task, advance=1)

        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )
            self.console.print(
                f"[green]✓ Indexed {len(points)} chunks from {len(self.articles)} articles[/green]"
            )

    def reindex_articles(self):
        self.console.print("\n[bold yellow]Reindexing all articles...[/bold yellow]")
        self._setup_vector_db()
        self.console.print("[green]✓ Reindexing complete![/green]")

    def _search_similar_chunks(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_embedding = list(self.embedding_model.embed([query]))[0]

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
        )

        results = []
        for result in search_results:
            article_idx = result.payload["article_index"] - 1
            results.append(
                SearchResult(
                    article=self.articles[article_idx],
                    chunk=result.payload["chunk_text"],
                    score=result.score,
                    chunk_index=result.payload["chunk_index"],
                )
            )

        return results

    def answer_question(self, query: str) -> str:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            progress.add_task("Searching for relevant information...", total=None)
            search_results = self._search_similar_chunks(query, top_k=8)

        if not search_results:
            return "I couldn't find relevant information in the articles to answer your question."

        # Filter results with score > 0.7
        relevant_results = [r for r in search_results if r.score > 0.7]

        if not relevant_results:
            relevant_results = search_results[
                :3
            ]  # Take top 3 if no high-scoring results

        context = "\n\n".join(
            [
                f"[{result.article.index}] From '{result.article.title}':\n{result.chunk}"
                for result in relevant_results
            ]
        )

        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided articles. 
When you use information from an article, you MUST include an inline citation in the format [1], [2], etc.
The citation number corresponds to the article index provided in the context.
If you use multiple pieces of information from the same article in one sentence, cite it once at the end.
If information comes from multiple articles, include all relevant citations.
Do not make up information not present in the provided passages.
Be concise but complete in your answers."""

        user_prompt = f"""Context from articles:
{context}

Question: {query}

Answer the question using ONLY the information from the context above. Include inline citations [n] for each piece of information you use."""

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            progress.add_task("Generating answer...", total=None)

            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

        answer = response.choices[0].message.content

        # Extract used citations
        used_citations = set()
        for result in relevant_results:
            if f"[{result.article.index}]" in answer:
                used_citations.add(result.article.index)

        if used_citations:
            answer += "\n\n**References:**"
            for idx in sorted(used_citations):
                article = self.articles[idx - 1]
                answer += f"\n- [{idx}] {article.title} - {article.url}"

        return answer

    def list_articles(self):
        self.console.print("\n[bold cyan]Available Articles:[/bold cyan]")
        self.console.print("─" * 80)

        for article in self.articles:
            panel_content = f"""[bold]Title:[/bold] {article.title}
[bold]URL:[/bold] {article.url}
[bold]Length:[/bold] {len(article.text.split())} words
[bold]Chunks:[/bold] {len(article.chunks) if article.chunks else 'Not indexed'}"""

            self.console.print(
                Panel(panel_content, title=f"[{article.index}]", border_style="blue")
            )


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about articles with inline citations"
    )
    parser.add_argument(
        "--articles", default="articles.json", help="Path to articles JSON file"
    )
    parser.add_argument(
        "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--qdrant-url", default="localhost:6333", help="Qdrant server URL"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available articles"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex all articles in the vector database",
    )

    args = parser.parse_args()

    console = Console()

    if not os.path.exists(args.articles):
        console.print(f"[red]Error: Articles file '{args.articles}' not found[/red]")
        sys.exit(1)

    try:
        with console.status("[bold green]Initializing Citation Assistant..."):
            assistant = CitationAssistant(args.articles, args.api_key, args.qdrant_url)
    except Exception as e:
        console.print(f"[red]Error initializing assistant: {e}[/red]")
        sys.exit(1)

    if args.list:
        assistant.list_articles()
        return

    if args.reindex:
        assistant.reindex_articles()
        return

    console.print(
        Panel.fit(
            "[bold cyan]Citation CLI[/bold cyan]\nAsk questions about the articles and get answers with inline citations",
            border_style="cyan",
        )
    )
    console.print(
        "Type [bold green]'quit'[/bold green] or [bold green]'exit'[/bold green] to end the session"
    )
    console.print("Type [bold green]'list'[/bold green] to see available articles")
    console.print(
        "Type [bold green]'reindex'[/bold green] to reindex all articles in the vector database"
    )
    console.print("─" * 80)

    while True:
        try:
            query = console.input(
                "\n[bold yellow]Your question:[/bold yellow] "
            ).strip()

            if query.lower() in ["quit", "exit"]:
                console.print("[green]Goodbye![/green]")
                break

            if query.lower() == "list":
                assistant.list_articles()
                continue

            if query.lower() == "reindex":
                assistant.reindex_articles()
                continue

            if not query:
                continue

            answer = assistant.answer_question(query)
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(answer))

        except KeyboardInterrupt:
            console.print("\n\n[green]Goodbye![/green]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
