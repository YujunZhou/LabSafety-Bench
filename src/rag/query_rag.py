import os
import argparse
from typing import List


def _lazy_import_llama_index():
    from llama_index.core import load_index_from_storage, StorageContext
    return load_index_from_storage, StorageContext


def load_query_engine(persist_dir: str, similarity_top_k: int = 5):
    load_index_from_storage, StorageContext = _lazy_import_llama_index()
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
    return query_engine


def format_sources(response) -> List[str]:
    sources = []
    try:
        for n in getattr(response, "source_nodes", []) or []:
            meta = n.node.metadata or {}
            file_path = meta.get("file_path") or meta.get("filename") or "unknown"
            score = getattr(n, "score", None)
            snippet = (n.node.get_content() or "").strip().replace("\n", " ")
            snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
            if score is not None:
                sources.append(f"{file_path} (score={score:.3f}) :: {snippet}")
            else:
                sources.append(f"{file_path} :: {snippet}")
    except Exception:
        pass
    return sources


def main():
    parser = argparse.ArgumentParser(description="Query an existing LlamaIndex RAG index and print sources")
    parser.add_argument("--persist_dir", type=str, required=True, help="Index persistence directory")
    parser.add_argument("--question", type=str, required=True, help="Question to query")
    parser.add_argument("--top_k", type=int, default=2, help="Number of similar documents to return")
    parser.add_argument("--embed_backend", type=str, default="hf", choices=["hf", "openai"], help="Embedding backend: hf(default) or openai")
    parser.add_argument("--hf_model_name", type=str, default="BAAI/bge-small-en-v1.5", help="HuggingFace embedding model name")
    args = parser.parse_args()

    # Ensure embed backend is set BEFORE loading the index to avoid defaulting to OpenAI
    try:
        from llama_index.core import Settings
        # Disable default LLM to avoid falling back to OpenAI when OPENAI_API_KEY is not set
        Settings.llm = None
        if args.embed_backend == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            print("Using OpenAI embeddings (text-embedding-3-small)")
        else:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            Settings.embed_model = HuggingFaceEmbedding(model_name=args.hf_model_name)
            print(f"Using HuggingFace embeddings ({args.hf_model_name})")
    except Exception as e:
        print(f"[Warning] Failed to set embedding backend or disable LLM: {e}")

    qe = load_query_engine(args.persist_dir, similarity_top_k=args.top_k)
    resp = qe.query(args.question)
    print("\n--- RAG-Powered Answer ---\n")
    print(str(resp))
    print("\n--- Sources ---\n")
    for s in format_sources(resp):
        print("- ", s)


if __name__ == "__main__":
    main()


