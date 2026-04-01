"""
Confluence RAG Ingestion Pipeline.

Flow:
    atlassian-python-api (Confluence)
        → Docling HTMLDocumentBackend
            → Structure-aware chunking  (text paragraphs + Markdown tables)
                → Azure OpenAI embeddings
                    → Azure AI Search (upsert)

Run:
    python ingest.py
    (reads from .env)
"""

import io
import logging
import os
from typing import Iterator, List, Optional

import tiktoken
from atlassian import Confluence
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DocumentStream
from docling.document_converter import DocumentConverter, FormatOption
from docling_core.types.doc import DocItemLabel

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Approximate token limit per text chunk (tables are never split).
MAX_TEXT_TOKENS = 400
# Chunks per embedding API call.
EMBED_BATCH_SIZE = 16
# Documents per Azure Search upload call.
UPLOAD_BATCH_SIZE = 100

# Fields expanded on every page fetch — body.view gives clean rendered HTML
# (no Confluence storage macros), which Docling's HTMLDocumentBackend handles well.
_PAGE_EXPAND = "body.view,history.lastUpdated,space"


# ── Confluence helpers ─────────────────────────────────────────────────────────

def iter_space_pages(
    confluence: Confluence,
    space_key: str,
    max_pages: Optional[int] = None,
) -> Iterator[dict]:
    """
    Yield page dicts for every page in a space using atlassian-python-api.
    get_all_pages_from_space() handles offset pagination internally.
    """
    pages = confluence.get_all_pages_from_space(
        space=space_key,
        start=0,
        limit=50,
        expand=_PAGE_EXPAND,
        content_type="page",
    )
    for i, page in enumerate(pages):
        if max_pages and i >= max_pages:
            break
        yield page


def iter_cql_pages(
    confluence: Confluence,
    cql: str,
    max_pages: Optional[int] = None,
) -> Iterator[dict]:
    """
    Yield page dicts matching an arbitrary CQL query.
    Useful for incremental updates, e.g.:
        cql = 'space="MYKEY" AND lastModified >= now("-7d")'
    """
    start = 0
    limit = 50
    fetched = 0
    while True:
        results = confluence.cql(
            cql,
            start=start,
            limit=limit,
            expand=_PAGE_EXPAND,
        )
        pages = results.get("results", [])
        if not pages:
            break
        for page in pages:
            yield page
            fetched += 1
            if max_pages and fetched >= max_pages:
                return
        if len(pages) < limit:
            break
        start += limit


def extract_page_metadata(page: dict, confluence_base_url: str) -> dict:
    history = page.get("history", {})
    last_updated = history.get("lastUpdated", {})
    space_key = page.get("space", {}).get("key", "")
    page_id = page["id"]
    webui_path = page.get("_links", {}).get("webui", "")
    return {
        "page_id": page_id,
        "title": page.get("title", ""),
        "space_key": space_key,
        "url": f"{confluence_base_url.rstrip('/')}{webui_path}",
        "author": last_updated.get("by", {}).get("displayName", "unknown"),
        "last_modified": last_updated.get("when", ""),
    }


def get_rendered_html(page: dict) -> str:
    """Extract the rendered HTML body (body.view) from a page dict."""
    return page.get("body", {}).get("view", {}).get("value", "")


# ── BeautifulSoup pre-processor ───────────────────────────────────────────────

_BLOCK_TAGS = frozenset({
    "div", "p", "ul", "ol", "table",
    "h1", "h2", "h3", "h4", "h5", "h6", "pre",
})


def _is_layout_table(table: Tag) -> bool:
    """
    Heuristic: a <table> is a layout table (used for page structure, not data)
    when it has NO <th> header cells AND every <td> cell contains at least one
    block-level element (div, p, table, heading, list, pre).

    Why: Confluence uses <table> for side-by-side column layouts as well as for
    real data tables. Docling cannot tell them apart — it emits every <table> as
    a TableItem, producing fake "table" chunks from layout wrappers. Identifying
    and unwrapping layout tables prevents these spurious chunks and lets Docling
    process the actual prose content inside the cells correctly.

    The heuristic is conservative: if even one cell has direct text or inline
    content (not wrapped in a block element), we treat it as a data table and
    leave it untouched.
    """
    # Data tables always have at least one header cell.
    if table.find("th"):
        return False

    cells = table.find_all("td")
    if not cells:
        # Empty table — no data value, safe to unwrap.
        return True

    for cell in cells:
        has_block = any(
            isinstance(child, Tag) and child.name in _BLOCK_TAGS
            for child in cell.children
        )
        if not has_block:
            # This cell holds direct text/inline content → treat as data table.
            return False

    return True


def preprocess_confluence_html(html: str) -> str:
    """
    Pre-process Confluence rendered HTML (body.view) before passing to Docling.

    Three transformations are applied in order:

    1. Promote expand macro titles → <h5>
       Confluence expand macros render their title inside a <span
       class="expand-control-text">, not as an <h> tag. Docling only recognises
       <h1>–<h6> as SECTION_HEADER items. Without this promotion, every chunk
       inside an expand macro is tagged with the last heading that appeared
       *before* the macro — silently propagating incorrect section_header values.
       Injecting <h5> as the first child of the expand content div means Docling
       detects it as SECTION_HEADER and updates the heading stack correctly.

    2. Promote tab macro titles → <h5>
       Confluence tab macros (built-in AUI tabs) render tab titles as anchor
       links inside a <ul class="tabs-menu"> list, not as <h> tags. Without
       promotion, content inside every tab pane inherits the last heading from
       *before* the entire tab block — all panes share the same wrong
       section_header. The fix mirrors transformation 1: inject <h5> at the
       start of each pane so Docling updates the heading stack per tab.

    3. Unwrap layout table wrappers
       Confluence uses <table> for visual page layout (two-column panels,
       side-by-side content blocks) as well as for real data tables. Docling
       emits every <table> as a TableItem and calls export_to_markdown() on it,
       producing meaningless "table" chunks for what is just structural markup.
       Layout tables (no <th>, all cells contain block elements) are replaced
       with a plain <div> containing their cell contents so Docling treats the
       inner content as normal text paragraphs.
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── 1. Promote expand macro titles to <h5> ────────────────────────────────
    # Confluence rendered structure:
    #   <div class="expand-container">
    #     <div class="expand-control">
    #       <span class="expand-control-text">Title here</span>
    #     </div>
    #     <div class="expand-content"> … child content … </div>
    #   </div>
    expand_promoted = 0
    for container in soup.find_all("div", class_="expand-container"):
        title_span = container.find("span", class_="expand-control-text")
        content_div = container.find("div", class_="expand-content")
        if title_span and content_div:
            h5 = soup.new_tag("h5")
            h5.string = title_span.get_text(strip=True)
            # Insert as first child of expand-content so Docling encounters
            # the heading before any child text or table items.
            content_div.insert(0, h5)
            expand_promoted += 1
    # A count of zero on a page known to contain expand macros means the
    # CSS class names have changed — check body.view HTML and update the
    # selector above.
    logger.debug(f"Promoted {expand_promoted} expand macro title(s) to <h5>")

    # ── 2. Promote tab macro titles to <h5> ───────────────────────────────────
    # Confluence built-in AUI tab macro rendered structure:
    #   <div class="aui-tabs horizontal-tabs">
    #     <ul class="tabs-menu">
    #       <li class="menu-item first-tab active-tab">
    #         <a href="#pane-id-1">Tab Title 1</a>
    #       </li>
    #       <li class="menu-item">
    #         <a href="#pane-id-2">Tab Title 2</a>
    #       </li>
    #     </ul>
    #     <div class="tabs-pane active-pane" id="pane-id-1"> … </div>
    #     <div class="tabs-pane"             id="pane-id-2"> … </div>
    #   </div>
    #
    # The anchor href="#pane-id" and the pane id="pane-id" are the only link
    # between a tab title and its content.  We use this to build a lookup map
    # so each pane gets its own heading injected before its content.
    #
    # Note: third-party tab apps (e.g. Tabs Pro by Adaptavist) render different
    # HTML and are not covered here.  Add a separate block below if needed.
    tabs_promoted = 0
    tabs_unmatched = 0
    for tab_container in soup.find_all("div", class_="aui-tabs"):
        # Build {pane_id → title_text} from the menu list.
        # lstrip("#") converts href="#pane-id-1" to the bare id "pane-id-1".
        pane_titles = {}
        for li in tab_container.select("ul.tabs-menu li.menu-item"):
            anchor = li.find("a", href=True)
            if anchor:
                pane_id = anchor["href"].lstrip("#")
                pane_titles[pane_id] = anchor.get_text(strip=True)

        # Inject <h5> as the first child of each pane whose id has a matching
        # title.  Panes without a menu entry (unusual but defensive) are left
        # untouched so their content is not silently misattributed.
        for pane in tab_container.find_all("div", class_="tabs-pane"):
            pane_id = pane.get("id", "")
            title = pane_titles.get(pane_id)
            if title:
                h5 = soup.new_tag("h5")
                h5.string = title
                # Insert before pane content so Docling sees the heading first.
                pane.insert(0, h5)
                tabs_promoted += 1
            else:
                # The pane has no matching menu title — the href/id linkage is
                # broken for this pane.  This can happen when Confluence generates
                # dynamic pane IDs that go out of sync with the menu href, or when
                # a nested tab macro causes ID collisions.  Without an injected
                # heading, this pane's content will inherit the wrong section_header.
                tabs_unmatched += 1

    # tabs_promoted == 0 on a page known to have tab macros → the aui-tabs /
    # tabs-menu / tabs-pane selectors no longer match the rendered HTML; check
    # body.view and update the selectors above.
    logger.debug(f"Promoted {tabs_promoted} tab pane title(s) to <h5>")
    # tabs_unmatched > 0 means one or more panes had no matching menu title and
    # were left without an injected heading — those panes will carry an incorrect
    # section_header.  Inspect the raw body.view HTML to diagnose the mismatch.
    if tabs_unmatched:
        logger.warning(
            f"{tabs_unmatched} tab pane(s) had no matching menu title — "
            "their section_header will be incorrect. "
            "Check the href/id linkage in body.view HTML."
        )

    # ── 3. Unwrap layout table wrappers ───────────────────────────────────────
    # Iterate over a list() snapshot because we mutate the tree in the loop.
    for table in list(soup.find_all("table")):
        if not _is_layout_table(table):
            continue
        wrapper = soup.new_tag("div")
        # Extract each cell's children in DOM order and move them to the wrapper.
        # .extract() detaches the node from its current position, so it is safe
        # to re-append it without creating duplicate references in the tree.
        for td in table.find_all("td"):
            for child in list(td.children):
                wrapper.append(child.extract())
        table.replace_with(wrapper)

    return str(soup)


# ── Docling HTML parser ────────────────────────────────────────────────────────

# Labels we treat as prose text (accumulate into a buffer, then chunk by tokens).
_TEXT_LABELS = frozenset({
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CAPTION,
    DocItemLabel.FOOTNOTE,
})


class DoclingHTMLParser:
    """
    Parses Confluence HTML with Docling and returns typed chunks.

    Strategy
    ─────────
    • Walk doc.iterate_items() in reading order.
    • SECTION_HEADER items update a heading_stack dict keyed by heading level
      (1–6).  When a heading at level N is seen, all entries with keys > N are
      cleared — so the stack always reflects the current ancestor hierarchy
      without stale deeper headings bleeding into unrelated sections.
    • TABLE items are emitted immediately as a single "table" chunk
      (Markdown-serialised via export_to_markdown()).  Tables are NEVER split.
    • TEXT / LIST_ITEM / CAPTION / FOOTNOTE items are accumulated in a buffer
      and flushed as a "text" chunk whenever the buffer exceeds MAX_TEXT_TOKENS.
    • Every chunk records:
        - section_header: the innermost (deepest) active heading text.
        - heading_hierarchy: dict of all active headings, e.g.
          {"h1": "Overview", "h2": "Installation", "h3": "Linux"}.
          Used by _build_docs() to build enriched embedding text.
    """

    def __init__(self):
        self._converter = DocumentConverter(
            format_options={
                InputFormat.HTML: FormatOption(backend=HTMLDocumentBackend)
            }
        )
        self._enc = tiktoken.get_encoding("cl100k_base")

    def parse(self, html: str) -> List[dict]:
        """
        Parse raw HTML and return a list of chunk dicts.
        Each dict has: content (str), chunk_type ("text"|"table"), section_header (str).
        """
        # Pre-process before Docling sees the HTML:
        #   · expand macro titles → <h5>  (fixes section_header propagation)
        #   · layout tables unwrapped      (prevents fake table chunks)
        html = preprocess_confluence_html(html)

        stream = DocumentStream(
            name="page.html",
            stream=io.BytesIO(html.encode("utf-8")),
        )
        result = self._converter.convert(stream)
        doc = result.document

        chunks: List[dict] = []
        # heading_stack maps heading level (int 1–6) → heading text.
        # When a heading at level N is encountered, all entries with level > N
        # are removed so that sibling or ancestor headings do not bleed into
        # unrelated sections.  Example after seeing h1 "A", h2 "B", h3 "C":
        #   {1: "A", 2: "B", 3: "C"}
        # After seeing a new h2 "D" the stack becomes:
        #   {1: "A", 2: "D"}   ← h3 "C" cleared because 3 > 2
        heading_stack: dict = {}
        text_buffer: List[str] = []

        def update_heading_stack(level: int, text: str) -> None:
            """Record heading at `level`; remove all deeper levels."""
            # Clear any previously recorded headings at deeper levels so
            # they don't appear as ancestors of the new, shallower heading.
            for k in list(heading_stack.keys()):
                if k > level:
                    del heading_stack[k]
            heading_stack[level] = text

        def current_section_header() -> str:
            """Return the deepest (most specific) active heading text, or ''."""
            if not heading_stack:
                return ""
            # max key = deepest heading level recorded so far.
            return heading_stack[max(heading_stack)]

        def current_hierarchy() -> dict:
            """
            Return a snapshot of the heading stack as human-readable keys.
            {"h1": "Overview", "h2": "Setup", "h3": "Linux"}
            Used to build enriched embedding text in _build_docs().
            """
            return {f"h{lvl}": txt for lvl, txt in sorted(heading_stack.items())}

        def flush_text_buffer():
            nonlocal text_buffer
            body = "\n".join(text_buffer).strip()
            if body:
                chunks.append({
                    "content": body,
                    "chunk_type": "text",
                    "section_header": current_section_header(),
                    # Snapshot taken at flush time so it reflects the headings
                    # active when this paragraph group was written, not later.
                    "heading_hierarchy": current_hierarchy(),
                })
            text_buffer = []

        for item, _level in doc.iterate_items():
            label = item.label

            if label == DocItemLabel.SECTION_HEADER:
                flush_text_buffer()
                text = getattr(item, "text", "").strip()
                # SectionHeaderItem.level is the h-tag depth (1=h1 … 6=h6).
                # Fall back to 1 if the attribute is absent (older docling).
                heading_level = getattr(item, "level", 1)
                update_heading_stack(heading_level, text)

            elif label == DocItemLabel.TABLE:
                flush_text_buffer()
                md = item.export_to_markdown()
                if md.strip():
                    chunks.append({
                        "content": md,
                        "chunk_type": "table",
                        "section_header": current_section_header(),
                        "heading_hierarchy": current_hierarchy(),
                    })

            elif label in _TEXT_LABELS:
                text = getattr(item, "text", "").strip()
                if text:
                    text_buffer.append(text)
                    token_count = len(self._enc.encode("\n".join(text_buffer)))
                    if token_count >= MAX_TEXT_TOKENS:
                        flush_text_buffer()

        flush_text_buffer()  # emit any trailing text
        return chunks




# ── Main ingestion orchestrator ────────────────────────────────────────────────

class ConfluenceRAGIngester:
    """
    Orchestrates: Confluence fetch → Docling parse → embed → Azure AI Search upsert.

    Re-running over the same space is idempotent: chunk IDs are deterministic
    (f"{page_id}_{chunk_index}") so upload_documents performs create-or-update.
    To handle page deletions or structure changes, call delete_page_chunks()
    before re-ingesting a specific page.
    """

    def __init__(
        self,
        confluence_base_url: str,
        confluence_email: str,
        confluence_api_token: str,
        azure_search_endpoint: str,
        azure_search_api_key: str,
        azure_openai_endpoint: str,
        azure_openai_api_key: str,
        embedding_deployment: str,
        index_name: str = "confluence-rag",
        embed_batch_size: int = EMBED_BATCH_SIZE,
        upload_batch_size: int = UPLOAD_BATCH_SIZE,
    ):
        self._confluence_base_url = confluence_base_url
        self._confluence = Confluence(
            url=confluence_base_url,
            username=confluence_email,
            password=confluence_api_token,
            cloud=True,
        )
        self._parser = DoclingHTMLParser()
        self._embedder = AzureOpenAIEmbedding(
            model=embedding_deployment,
            deployment_name=embedding_deployment,
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_api_key,
            api_version="2024-02-01",
            embed_batch_size=embed_batch_size,
        )
        self._search = SearchClient(
            endpoint=azure_search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(azure_search_api_key),
        )
        self._upload_batch = upload_batch_size

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_docs(
        self,
        chunks: List[dict],
        page_meta: dict,
        start_idx: int,
    ) -> List[dict]:
        """
        Merge chunk content with page metadata into Azure Search documents.

        Each document gets a transient `_embed_text` field (NOT an index field)
        containing an enriched string used only for embedding generation:

            "Page: <title> | Section: <h1> > <h2> > <h3>\\n\\n<content>"

        Prepending the page title and ancestor headings means the embedding
        vector carries location context that the raw content alone would lack.
        This enriches the vector leg of hybrid search without altering the
        stored `content` field that users see in query results.

        `_embed_text` is stripped by `_attach_embeddings()` before upload so
        it never reaches Azure Search.
        """
        docs = []
        for i, chunk in enumerate(chunks):
            idx = start_idx + i

            # Build the breadcrumb from the heading hierarchy snapshot stored
            # on each chunk (e.g. {"h1": "Overview", "h2": "Setup"}).
            # Sorted by level so the breadcrumb reads h1 > h2 > h3.
            hierarchy: dict = chunk.get("heading_hierarchy", {})
            breadcrumb = " > ".join(
                hierarchy[k] for k in sorted(hierarchy)
            )
            # Enriched text: prefix with page title + full heading path so the
            # embedding captures *where* in the document this chunk lives.
            embed_text = (
                f"Page: {page_meta['title']} | Section: {breadcrumb}\n\n"
                f"{chunk['content']}"
            )

            # Fallback: chunks that appear before the first heading on a page
            # have an empty section_header.  An empty string is unhelpful to
            # the Azure semantic reranker, which uses section_header as a
            # keyword signal alongside title and content.  Substituting the
            # page title ensures those chunks still carry meaningful context
            # and are not ranked lower than they deserve simply because they
            # sit before the first heading in the document.
            section_header = chunk["section_header"] or page_meta["title"]

            docs.append({
                "id": f"{page_meta['page_id']}_{idx}",
                "page_id": page_meta["page_id"],
                "title": page_meta["title"],
                "space_key": page_meta["space_key"],
                "url": page_meta["url"],
                "author": page_meta["author"],
                "last_modified": page_meta["last_modified"],
                "content": chunk["content"],
                "section_header": section_header,
                "chunk_type": chunk["chunk_type"],
                "chunk_index": idx,
                # Transient — consumed by _attach_embeddings(), not uploaded.
                "_embed_text": embed_text,
            })
        return docs

    def _attach_embeddings(self, docs: List[dict]) -> None:
        """
        Generate and attach content_vector to each document in-place.

        Uses the transient `_embed_text` field (page title + heading breadcrumb
        + content) for richer vector representations, then pops it from the dict
        so it is not present when the document is uploaded to Azure Search.

        AzureOpenAIEmbedding handles batching internally via embed_batch_size.
        """
        # pop() both retrieves the enriched text AND removes it from the doc
        # in one step — no separate cleanup pass needed.
        texts = [d.pop("_embed_text") for d in docs]
        vectors = self._embedder.get_text_embedding_batch(texts)
        for doc, vec in zip(docs, vectors):
            doc["content_vector"] = vec

    def _upload(self, docs: List[dict]) -> int:
        results = self._search.upload_documents(docs)
        return sum(1 for r in results if r.succeeded)

    # ── Public API ─────────────────────────────────────────────────────────────

    def delete_page_chunks(self, page_id: str) -> int:
        """
        Delete all indexed chunks for a given page_id.
        Call this before re-ingesting a page to avoid stale chunks.
        """
        results = self._search.search(
            search_text="*",
            filter=f"page_id eq '{page_id}'",
            select=["id"],
        )
        ids = [{"id": r["id"]} for r in results]
        if ids:
            self._search.delete_documents(ids)
        return len(ids)

    def ingest_space(
        self,
        space_key: str,
        max_pages: Optional[int] = None,
    ) -> int:
        """
        Ingest all pages from a Confluence space.
        Returns the total number of chunks indexed.
        """
        total_chunks = 0
        buffer: List[dict] = []

        for page in iter_space_pages(self._confluence, space_key, max_pages=max_pages):
            html = get_rendered_html(page)
            if not html.strip():
                logger.warning(f"Skipping empty page: {page.get('id')}")
                continue

            page_meta = extract_page_metadata(page, self._confluence_base_url)

            try:
                chunks = self._parser.parse(html)
            except Exception as exc:
                logger.error(f"Docling parse failed for page {page_meta['page_id']}: {exc}")
                continue

            if not chunks:
                logger.warning(f"No chunks extracted from page '{page_meta['title']}'")
                continue

            text_count = sum(1 for c in chunks if c["chunk_type"] == "text")
            table_count = sum(1 for c in chunks if c["chunk_type"] == "table")
            logger.info(
                f"Page '{page_meta['title']}': "
                f"{text_count} text chunks, {table_count} table chunks"
            )

            docs = self._build_docs(chunks, page_meta, start_idx=total_chunks)
            buffer.extend(docs)
            total_chunks += len(docs)

            if len(buffer) >= self._upload_batch:
                self._attach_embeddings(buffer)
                uploaded = self._upload(buffer)
                logger.info(f"Uploaded {uploaded}/{len(buffer)} documents to Azure Search.")
                buffer = []

        # Flush remaining
        if buffer:
            self._attach_embeddings(buffer)
            uploaded = self._upload(buffer)
            logger.info(f"Final flush: uploaded {uploaded}/{len(buffer)} documents.")

        logger.info(f"Ingestion complete. Total chunks: {total_chunks}")
        return total_chunks


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingester = ConfluenceRAGIngester(
        confluence_base_url=os.environ["CONFLUENCE_BASE_URL"],
        confluence_email=os.environ["CONFLUENCE_EMAIL"],
        confluence_api_token=os.environ["CONFLUENCE_API_TOKEN"],
        azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
        azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        embedding_deployment=os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        ),
        index_name=os.environ.get("AZURE_SEARCH_INDEX_NAME", "confluence-rag"),
    )

    ingester.ingest_space(
        space_key=os.environ["CONFLUENCE_SPACE_KEY"],
        max_pages=int(os.environ["MAX_PAGES"]) if os.environ.get("MAX_PAGES") else None,
    )
