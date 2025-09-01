import os
import argparse
from typing import Optional


def _lazy_import_llama_index():
    # Defer heavy imports to runtime to avoid hard dependency for users not running RAG
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.core.storage import StorageContext as StorageContextType
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core import load_index_from_storage
    return VectorStoreIndex, SimpleDirectoryReader, StorageContext, StorageContextType, BaseIndex, load_index_from_storage


def build_or_reload_index(input_dir: str, persist_dir: str, chunk_size: int = 1024, chunk_overlap: int = 128):
    VectorStoreIndex, SimpleDirectoryReader, StorageContext, StorageContextType, BaseIndex, load_index_from_storage = _lazy_import_llama_index()

    os.makedirs(persist_dir, exist_ok=True)

    # If an index already exists, reload it rather than rebuilding
    if any(os.scandir(persist_dir)):
        storage_context: StorageContextType = StorageContext.from_defaults(persist_dir=persist_dir)
        index: BaseIndex = load_index_from_storage(storage_context)
        return index

    # Read documents
    reader = SimpleDirectoryReader(input_dir, recursive=True)
    documents = reader.load_data()

    # Optionally set global chunking (LlamaIndex handles defaults; we keep parameters for future customization)
    index = VectorStoreIndex.from_documents(documents)

    # Persist index to disk for later reload
    index.storage_context.persist(persist_dir=persist_dir)
    return index


def main():
    parser = argparse.ArgumentParser(description="Build or reload a LlamaIndex vector index from safety documents directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory path containing safety documents (PDF/TXT etc.)")
    parser.add_argument("--persist_dir", type=str, default="./storage/safety_index", help="Index persistence directory")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=64, help="Chunk overlap")
    parser.add_argument("--embed_backend", type=str, default="hf", choices=["hf", "openai"], help="Embedding backend: hf(default) or openai")
    parser.add_argument("--hf_model_name", type=str, default="BAAI/bge-small-en-v1.5", help="HuggingFace embedding model name")
    args = parser.parse_args()

    # Configure embedding backend
    try:
        from llama_index.core import Settings
        if args.embed_backend == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            print("Using OpenAI embeddings (text-embedding-3-small)")
        else:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            Settings.embed_model = HuggingFaceEmbedding(model_name=args.hf_model_name)
            print(f"Using HuggingFace embeddings ({args.hf_model_name})")
    except Exception as e:
        print(f"[Warning] Failed to set embedding backend: {e}")

    print(f"Loading documents from: {args.input_dir}")
    index = build_or_reload_index(args.input_dir, args.persist_dir, args.chunk_size, args.chunk_overlap)
    print(f"Index ready. Persisted at: {args.persist_dir}")


if __name__ == "__main__":
    main()


