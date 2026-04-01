"""
Azure AI Search index creation for Confluence RAG (text + tables).
Run this once before ingestion.

Usage:
    python index_create.py
    (reads from .env or environment variables)
"""

import os
from dotenv import load_dotenv

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

VECTOR_DIMS = 1536  # text-embedding-3-small


def _build_fields() -> list:
    return [
        # ── Primary key ────────────────────────────────────────────────────────
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),

        # ── Confluence page metadata ───────────────────────────────────────────
        SimpleField(
            name="page_id",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="space_key",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="url",
            type=SearchFieldDataType.String,
            retrievable=True,
        ),
        SearchableField(
            name="author",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="last_modified",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),

        # ── Content fields ─────────────────────────────────────────────────────
        # The nearest ancestor section heading — improves semantic reranking.
        SearchableField(
            name="section_header",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
            filterable=True,
            retrievable=True,
        ),
        # For text chunks: plain paragraph text.
        # For table chunks: Markdown-serialised table (preserves structure for BM25).
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
            retrievable=True,
        ),

        # ── Chunk classification ───────────────────────────────────────────────
        # "text" or "table" — enables filtered retrieval and prompt rendering.
        SimpleField(
            name="chunk_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),

        # ── Vector field ───────────────────────────────────────────────────────
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]


def _build_vector_search() -> VectorSearch:
    return VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algo",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine",
                },
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algo",
            )
        ],
    )


def _build_semantic_search() -> SemanticSearch:
    """
    Semantic reranker config.
    The reranker sees title + content + section_header for richer context,
    especially important for table chunks where the heading carries meaning.
    """
    return SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="confluence-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[
                        SemanticField(field_name="content"),
                        SemanticField(field_name="section_header"),
                    ],
                    keywords_fields=[
                        SemanticField(field_name="space_key"),
                        SemanticField(field_name="chunk_type"),
                    ],
                ),
            )
        ]
    )


def create_index(
    search_endpoint: str,
    search_api_key: str,
    index_name: str = "confluence-rag",
) -> None:
    """Create (or update) the Azure AI Search index."""
    client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=AzureKeyCredential(search_api_key),
    )

    index = SearchIndex(
        name=index_name,
        fields=_build_fields(),
        vector_search=_build_vector_search(),
        semantic_search=_build_semantic_search(),
    )

    result = client.create_or_update_index(index)
    print(f"Index '{result.name}' created/updated successfully.")


if __name__ == "__main__":
    create_index(
        search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
        index_name=os.environ.get("AZURE_SEARCH_INDEX_NAME", "confluence-rag"),
    )
