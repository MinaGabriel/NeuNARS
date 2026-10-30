# HotpotQA.py: the HotpotQA dataset (distractor setting) in the shared schema, like MuSiQue.py.
#   Shared schema: id, dataset, question, answer, gold_answers, hops, decomposition, paragraphs, source
# Only "bridge" questions are kept: a 2-hop chain through one entity (like MuSiQue's 2-hop questions).
# "comparison" questions ("Who was born first, X or Y?", yes/no) need comparing values, which NARS
# cannot do here, so they are set aside as a NARS limitation, like the how/why questions in MuSiQue.
import random

from datasets import load_dataset

DATASET = "HotpotQA"
POOL_SIZE = 1500  # enough to keep 150 after the closed-book and evidence filters; all 5,918 would just cost more


def to_schema(ex):
    supporting = set(ex["supporting_facts"]["title"])
    return {
        "id": ex["id"],
        "dataset": DATASET,
        "question": ex["question"],
        "answer": ex["answer"],
        "gold_answers": [ex["answer"]],  # HotpotQA has no answer aliases
        "hops": 2,
        "decomposition": [],  # HotpotQA has no sub-questions
        "paragraphs": [{"idx": i, "title": title, "text": "".join(sentences).strip(),
                        "is_supporting": title in supporting}
                       for i, (title, sentences) in enumerate(zip(ex["context"]["title"], ex["context"]["sentences"]))],
        "type": ex["type"],
        "level": ex["level"],
        "source": "hotpotqa/hotpot_qa (distractor, validation)",
    }


def load_pool(seed=0, split="validation", full=False):
    """Random bridge questions with a non-yes/no answer, shuffled: POOL_SIZE of validation, or all of train."""
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    pool = [{**to_schema(ex), "source": f"hotpotqa/hotpot_qa (distractor, {split})"}
            for ex in data if ex["type"] == "bridge" and ex["answer"].lower() not in ("yes", "no")]
    random.Random(seed).shuffle(pool)
    return pool[:POOL_SIZE] if split == "validation" and not full else pool   # full: every question (build.py)


if __name__ == "__main__":
    pool = load_pool()
    print(f"{len(pool)} {DATASET} bridge questions; supporting paragraphs per question: "
          f"{sorted(set(sum(p['is_supporting'] for p in r['paragraphs']) for r in pool))}")
    print(pool[0]["question"], "->", pool[0]["answer"])
