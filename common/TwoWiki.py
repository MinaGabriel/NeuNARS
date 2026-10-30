# TwoWiki.py: the 2WikiMultihopQA dataset in the shared schema, like MuSiQue.py and HotpotQA.py.
#   Shared schema: id, dataset, question, answer, gold_answers, hops, decomposition, paragraphs, source
# Only "compositional" and "inference" questions are kept: chains such as "Who is the mother of the director
# of film X?". "comparison" and "bridge_comparison" questions need comparing values or yes/no, which NARS
# cannot do here, so they are set aside as a NARS limitation, like in HotpotQA.
# Usage: python pipeline.py --dataset TwoWiki --per-hop 150   ->  runs/2Wiki_test.pkl
import random

from datasets import load_dataset

DATASET = "2Wiki"
POOL_SIZE = 1000  # enough to keep 150 after the closed-book and evidence filters


def to_schema(ex):
    supporting = {title for title, _ in ex["supporting_facts"]}
    return {
        "id": ex["_id"],
        "dataset": DATASET,
        "question": ex["question"],
        "answer": ex["answer"],
        "gold_answers": [ex["answer"]],
        "hops": len(ex["evidences"]),  # one Wikidata triple per hop
        "decomposition": [{"question": f"{s} -> {r}", "answer": o, "support_idx": None} for s, r, o in ex["evidences"]],
        "paragraphs": [{"idx": i, "title": title, "text": " ".join(sentences).strip(), "is_supporting": title in supporting}
                       for i, (title, sentences) in enumerate(ex["context"])],
        "type": ex["type"],
        "source": "voidful/2WikiMultihopQA (validation)",
    }


def load_pool(seed=0, split="validation", full=False):
    """Random chain questions (compositional and inference) with a non-yes/no answer, shuffled: POOL_SIZE of
    validation, or all of train."""
    data = load_dataset("voidful/2WikiMultihopQA", split=split)
    pool = [{**to_schema(ex), "source": f"voidful/2WikiMultihopQA ({split})"} for ex in data
            if ex["type"] in ("compositional", "inference") and ex["answer"].lower() not in ("yes", "no")]
    random.Random(seed).shuffle(pool)
    return pool[:POOL_SIZE] if split == "validation" and not full else pool   # full: every question (build.py)


if __name__ == "__main__":
    from collections import Counter
    pool = load_pool()
    print(f"{len(pool)} {DATASET} questions, by hops: {dict(Counter(r['hops'] for r in pool))}, "
          f"supporting paragraphs: {dict(Counter(sum(p['is_supporting'] for p in r['paragraphs']) for r in pool))}")
    print(pool[0]["question"], "->", pool[0]["answer"])
