import os
import json
import argparse


def _lazy_import():
    from datasets import load_dataset
    from llama_index.core import Settings, load_index_from_storage, StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return load_dataset, Settings, load_index_from_storage, StorageContext, HuggingFaceEmbedding


def load_retriever(persist_dir: str, hf_model_name: str, top_k: int):
    load_dataset, Settings, load_index_from_storage, StorageContext, HuggingFaceEmbedding = _lazy_import()
    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbedding(model_name=hf_model_name)
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index.as_retriever(similarity_top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="Precompute RAG contexts for a dataset and save to JSON")
    parser.add_argument("--persist_dir", type=str, required=True, help="Index persistence directory")
    parser.add_argument("--hf_model", type=str, default="BAAI/bge-m3", help="HF embedding model name")
    parser.add_argument("--top_k", type=int, default=2, help="Number of similar documents")
    parser.add_argument("--dataset", type=str, default="text_qa", choices=["text_qa", "scenario", "decision"], help="Target dataset type")
    parser.add_argument("--scenario_variant", type=int, default=1, choices=[1, 2, 3, 4], help="Question variant for scenario dataset (1-4)")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    retriever = load_retriever(args.persist_dir, args.hf_model, args.top_k)
    load_dataset, *_ = _lazy_import()

    if args.dataset == "text_qa":
        ds = load_dataset('yujunzhou/LabSafety_Bench', name='MCQ', split='QA')
        items = [f"Question: {s['Question']}\nFinal Choice:" for s in ds]
    elif args.dataset == "scenario":
        ds = load_dataset('yujunzhou/LabSafety_Bench', name='scenario', split='scenario')
        from prompts import generate_lab_safety_prompts
        idx_map = {1: 0, 2: 1, 3: 2, 4: 3}
        sel = idx_map[args.scenario_variant]
        items = [generate_lab_safety_prompts(s['Scenario'])[sel] for s in ds]
    else:  # decision
        ds = load_dataset('yujunzhou/LabSafety_Bench', name='scenario', split='scenario')
        items = [f"Scenario: {s['Scenario']}\nDecision: {d['Decision']}" for s in ds for d in s['Decisions']]

    contexts = []
    for q in items:
        nodes = retriever.retrieve(q)
        parts = []
        for n in nodes:
            snippet = (n.node.get_content() or '').strip()
            if len(snippet) > 600:
                snippet = snippet[:600] + '...'
            parts.append(snippet)
        contexts.append("\n\n".join(parts))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(contexts, f, ensure_ascii=False)
    print(f"Saved contexts to {args.out} ({len(contexts)} entries)")


if __name__ == "__main__":
    main()


