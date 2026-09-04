"""RAG tool that retrieves travel-tip context from a local city guide corpus.

The corpus lives in data/city_guides/*.md. On first run it's chunked and
embedded into a persistent Chroma store; subsequent runs reuse that store.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from pydantic import BaseModel, Field

from config import settings

_GUIDES_DIR = Path(__file__).parent.parent / "data" / "city_guides"
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    """Lazily build (or load) the persistent Chroma vector store."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    store = Chroma(
        collection_name=settings.rag_collection_name,
        embedding_function=_embeddings,
        persist_directory=settings.rag_persist_dir,
    )

    if store._collection.count() == 0:
        _ingest_guides(store)

    _vectorstore = store
    return store


def _ingest_guides(store: Chroma) -> None:
    """Chunk and embed every markdown file in data/city_guides into `store`."""
    loader = DirectoryLoader(
        str(_GUIDES_DIR), glob="*.md", loader_cls=TextLoader
    )
    documents = loader.load()
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    if chunks:
        store.add_documents(chunks)


class GuideSearchInput(BaseModel):
    query: str = Field(
        description="What to look up, e.g. 'best time to visit Kyoto' "
        "or 'getting around Lisbon'"
    )


@tool("search_travel_guide", args_schema=GuideSearchInput)
def search_travel_guide(query: str) -> str:
    """Search curated local travel guides for practical tips: neighborhoods,
    transit, currency, tipping, best times to visit, and day trips. Use this
    for 'how do I get around' or 'what's it like' style questions, as a
    complement to live places/weather lookups.
    """
    retriever = _get_vectorstore().as_retriever(
        search_kwargs={"k": settings.rag_top_k}
    )
    results = retriever.invoke(query)
    if not results:
        return "No relevant guide content found."

    return "\n---\n".join(doc.page_content for doc in results)
