# MuSiQue.py: the MuSiQue dataset in the shared schema. Every dataset file provides DATASET and load_pool(seed).
#   Shared schema: id, dataset, question, answer, gold_answers, hops, decomposition, paragraphs, source
import random

from datasets import load_dataset

DATASET = "MuSiQue"


def to_schema(ex):
    return {
        "id": ex["id"],
        "dataset": DATASET,
        "question": ex["question"],
        "answer": ex["answer"],
        "gold_answers": [ex["answer"]] + list(ex["answer_aliases"] or []),
        "hops": len(ex["question_decomposition"]),
        "decomposition": [{"question": s["question"], "answer": s["answer"], "support_idx": s["paragraph_support_idx"]}
                          for s in ex["question_decomposition"]],
        "paragraphs": [{"idx": p["idx"], "title": p["title"], "text": p["paragraph_text"],
                        "is_supporting": p["is_supporting"]} for p in ex["paragraphs"]],
        "source": "Salesforce/ContextualBench (MuSiQue, validation)",
    }


def load_pool(seed=0, split="validation", full=False):
    """All answerable questions (2-, 3- and 4-hop), shuffled. split="train" reads the official MuSiQue release
    (ContextualBench has only validation; same ids and fields)."""
    if split == "train":
        data = load_dataset("dgslibisey/MuSiQue", split="train")
    else:
        data = load_dataset("Salesforce/ContextualBench", "MuSiQue", split="validation")
    pool = [{**to_schema(ex), "source": f"MuSiQue ({split})"} for ex in data if ex["answerable"]]
    random.Random(seed).shuffle(pool)
    return pool


if __name__ == "__main__":
    from collections import Counter
    pool = load_pool()
    print(f"{len(pool)} answerable {DATASET} questions, by hops: {dict(sorted(Counter(r['hops'] for r in pool).items()))}")
