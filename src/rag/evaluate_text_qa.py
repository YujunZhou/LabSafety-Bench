import os
import json
import argparse
from typing import Dict, Any


def _lazy_import():
    from datasets import load_dataset
    from llama_index.core import Settings, load_index_from_storage, StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return load_dataset, Settings, load_index_from_storage, StorageContext, HuggingFaceEmbedding


def load_query_engine(persist_dir: str, hf_model_name: str, top_k: int):
    load_dataset, Settings, load_index_from_storage, StorageContext, HuggingFaceEmbedding = _lazy_import()
    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbedding(model_name=hf_model_name)
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=top_k)


def format_sources(response) -> Any:
    items = []
    for n in getattr(response, "source_nodes", []) or []:
        meta = n.node.metadata or {}
        file_path = meta.get("file_path") or meta.get("filename") or "unknown"
        score = getattr(n, "score", None)
        snippet = (n.node.get_content() or "").strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        items.append({
            "file": file_path,
            "score": float(score) if score is not None else None,
            "snippet": snippet,
        })
    return items


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate Text-QA using RAG query engine")
    parser.add_argument("--persist_dir", type=str, required=True, help="Index persistence directory")
    parser.add_argument("--hf_model", type=str, default="BAAI/bge-m3", help="HF embedding model name")
    parser.add_argument("--top_k", type=int, default=2, help="Number of similar documents")
    parser.add_argument("--split", type=str, default="QA", choices=["QA", "sampledQA"], help="Dataset split")
    parser.add_argument("--out", type=str, default="./rag_text_qa_results.json", help="Output JSON file path")
    args = parser.parse_args()

    qe = load_query_engine(args.persist_dir, args.hf_model, args.top_k)

    load_dataset, *_ = _lazy_import()
    dataset = load_dataset('yujunzhou/LabSafety_Bench', name='MCQ', split=args.split)
    questions = [sample['Question'] for sample in dataset]
    gts = [sample['Correct Answer'] for sample in dataset]

    results = []
    for i, q in enumerate(questions):
        resp = qe.query(f"Question: {q}\nAnswer:")
        results.append({
            "index": i,
            "question": q,
            "ground_truth": gts[i],
            "answer": str(resp),
            "sources": format_sources(resp),
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()


