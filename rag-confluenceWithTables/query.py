"""
Confluence RAG Query Engine — Hybrid Search (BM25 + Vector + Semantic Reranking).

Architecture:
    User question
        → AzureHybridRetriever  (LlamaIndex BaseRetriever)
            ├─ Azure AI Search:  BM25 full-text  +  HNSW vector  +  semantic reranker
            └─ Returns List[NodeWithScore]  (text nodes AND table nodes)
        → table-aware prompt builder
        → Azure OpenAI chat completion
        → answer + sources

Run:
    python query.py
    (reads from .env)
"""

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Custom LlamaIndex retriever ────────────────────────────────────────────────

class AzureHybridRetriever(BaseRetriever):
    """
    LlamaIndex-compatible retriever that issues a single Azure AI Search call
    combining three signals:
        1. BM25 full-text search  (search_text parameter)
        2. HNSW vector similarity  (vector_queries parameter)
        3. Azure semantic reranker  (query_type=SEMANTIC)

    Why not use LlamaIndex's built-in AzureAISearchVectorStore?
    ─────────────────────────────────────────────────────────────
    The built-in store issues pure vector-only searches.  Getting all three
    signals requires building the SearchClient call directly.  BaseRetriever
    is the clean hook: implement _retrieve(), return List[NodeWithScore], and
    the rest of the LlamaIndex pipeline (response synthesis, streaming, etc.)
    works unmodified.
    """

    def __init__(
        self,
        search_client: SearchClient,
        oai_client: AzureOpenAI,
        embedding_deployment: str,
        top_k: int = 6,
        chunk_type_filter: Optional[str] = None,  # "text" | "table" | None
    ):
        self._search_client = search_client
        self._oai_client = oai_client
        self._embedding_deployment = embedding_deployment
        self._top_k = top_k
        self._chunk_type_filter = chunk_type_filter
        super().__init__()

    def _embed(self, text: str) -> List[float]:
        resp = self._oai_client.embeddings.create(
            model=self._embedding_deployment,
            input=[text],
            dimensions=1536,
        )
        return resp.data[0].embedding

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str
        query_vector = self._embed(query_str)

        # Over-fetch for the vector leg so the semantic reranker has a richer
        # candidate pool.  The final result count is controlled by `top`.
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=self._top_k * 2,
            fields="content_vector",
        )

        filter_expr = None
        if self._chunk_type_filter:
            filter_expr = f"chunk_type eq '{self._chunk_type_filter}'"

        raw_results = self._search_client.search(
            search_text=query_str,
            vector_queries=[vector_query],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="confluence-semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=self._top_k,
            filter=filter_expr,
            select=[
                "id", "page_id", "title", "url",
                "content", "section_header", "chunk_type",
                "space_key", "author",
            ],
        )

        nodes: List[NodeWithScore] = []
        for result in raw_results:
            # Prefer semantic reranker score; fall back to fusion score.
            score = (
                result.get("@search.reranker_score")
                or result.get("@search.score", 0.0)
            )
            node = TextNode(
                text=result["content"],
                id_=result["id"],
                metadata={
                    "page_id": result.get("page_id", ""),
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "space_key": result.get("space_key", ""),
                    "author": result.get("author", ""),
                    "section_header": result.get("section_header", ""),
                    "chunk_type": result.get("chunk_type", "text"),
                },
            )
            nodes.append(NodeWithScore(node=node, score=float(score)))

        return nodes


# ── Table-aware prompt builder ─────────────────────────────────────────────────

def build_context(nodes: List[NodeWithScore]) -> str:
    """
    Render retrieved nodes into an LLM prompt context.

    Table chunks get a labelled Markdown code fence so the LLM understands
    it is reading structured data and should not paraphrase cell values.
    Text chunks are rendered plainly.
    """
    parts = []
    for nws in nodes:
        meta = nws.node.metadata
        chunk_type = meta.get("chunk_type", "text")
        title = meta.get("title", "")
        section = meta.get("section_header", "")
        url = meta.get("url", "")

        parts.append(f"**Source:** {title}  |  **Section:** {section}  |  {url}")

        if chunk_type == "table":
            parts.append(
                f"*The following is a table from section '{section}' of '{title}':*"
            )
            parts.append(f"```\n{nws.node.text}\n```")
        else:
            parts.append(nws.node.text)

        parts.append("---")

    return "\n\n".join(parts)


# ── Main query engine ──────────────────────────────────────────────────────────

class ConfluenceQueryEngine:
    """
    High-level query interface.

    query()             — hybrid search over all chunk types.
    query_tables_only() — hybrid search restricted to table chunks.
    """

    def __init__(
        self,
        azure_search_endpoint: str,
        azure_search_api_key: str,
        azure_openai_endpoint: str,
        azure_openai_api_key: str,
        embedding_deployment: str = "text-embedding-3-small",
        chat_deployment: str = "gpt-4o",
        index_name: str = "confluence-rag",
        top_k: int = 6,
        api_version: str = "2024-02-01",
    ):
        self._search_client = SearchClient(
            endpoint=azure_search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(azure_search_api_key),
        )
        self._oai_client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_api_key,
            api_version=api_version,
        )
        self._embedding_deployment = embedding_deployment
        self._chat_deployment = chat_deployment
        self._top_k = top_k

    def _make_retriever(self, chunk_type_filter: Optional[str] = None) -> AzureHybridRetriever:
        return AzureHybridRetriever(
            search_client=self._search_client,
            oai_client=self._oai_client,
            embedding_deployment=self._embedding_deployment,
            top_k=self._top_k,
            chunk_type_filter=chunk_type_filter,
        )

    def _generate(self, question: str, nodes: List[NodeWithScore]) -> str:
        context = build_context(nodes)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions from Confluence "
                    "documentation. When the context contains a Markdown table, "
                    "preserve all cell values accurately in your answer. "
                    "Always cite the source page title and URL."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer using only the provided context. "
                    "If a table is relevant, reproduce it in your answer."
                ),
            },
        ]
        resp = self._oai_client.chat.completions.create(
            model=self._chat_deployment,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        return resp.choices[0].message.content

    def _format_response(self, answer: str, nodes: List[NodeWithScore]) -> dict:
        sources = [
            {
                "title": n.node.metadata.get("title"),
                "url": n.node.metadata.get("url"),
                "section_header": n.node.metadata.get("section_header"),
                "chunk_type": n.node.metadata.get("chunk_type"),
                "score": round(n.score or 0.0, 4),
                "snippet": n.node.text[:300],
            }
            for n in nodes
        ]
        return {"answer": answer, "sources": sources}

    # ── Public methods ─────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """Hybrid search over all chunk types (text + tables)."""
        retriever = self._make_retriever()
        nodes = retriever.retrieve(QueryBundle(query_str=question))
        answer = self._generate(question, nodes)
        return self._format_response(answer, nodes)

    def query_tables_only(self, question: str) -> dict:
        """Hybrid search restricted to table chunks only."""
        retriever = self._make_retriever(chunk_type_filter="table")
        nodes = retriever.retrieve(QueryBundle(query_str=question))
        answer = self._generate(question, nodes)
        return self._format_response(answer, nodes)

    def query_text_only(self, question: str) -> dict:
        """Hybrid search restricted to text chunks only."""
        retriever = self._make_retriever(chunk_type_filter="text")
        nodes = retriever.retrieve(QueryBundle(query_str=question))
        answer = self._generate(question, nodes)
        return self._format_response(answer, nodes)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = ConfluenceQueryEngine(
        azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
        azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        embedding_deployment=os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        ),
        chat_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
        index_name=os.environ.get("AZURE_SEARCH_INDEX_NAME", "confluence-rag"),
    )

    # General query (text + tables)
    result = engine.query("What are the system requirements listed in the documentation?")
    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    for src in result["sources"]:
        print(f"  [{src['chunk_type']}] {src['title']} / {src['section_header']}  (score={src['score']})")

    print("\n" + "=" * 60)

    # Table-only query
    result = engine.query_tables_only("Show me the pricing table for enterprise plans.")
    print("\n=== Table-focused Answer ===")
    print(result["answer"])
