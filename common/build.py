"""Builds a dataset's question set: <dataset>/questions.pkl, read by every notebook of that dataset.

The same filters as every earlier set (pipeline.py), with Llama-3.1-70B:
1. Pool: 2-hop questions, shuffled with seed 0; the validation split first, then (if needed) the train split. Excluded:
   every old test and dev question, any question whose supporting paragraph is in those sets or in the LoRA training
   data, and every train question tried during training-data generation.
2. Closed book: drop the questions the 70B answers without context (the facts must matter).
3. Evidence: drop the questions the 70B cannot answer from the 2 supporting paragraphs (they must suffice).
4. Select: drop time-sensitive questions and answers that are not one entity, date or number; keep the questions
   independent: no two share a supporting paragraph or a gold answer.
5. Facts: the 70B lists the facts of the supporting paragraphs and of k_max distractor paragraphs per question.
6. Layout: the distractors get a fixed order (k distractors = the first k: the sets nest) and every paragraph a fixed
   reading position, so adding distractors only inserts paragraphs.
"""
import importlib
import json
import os
import pickle
import random
from collections import Counter

from llm import DeepInfra
from pipeline import (CLOSED_BOOK, EVIDENCE, EXTRACT, answer_and_grade, normalize, parse_facts, removal_reason,
                      select, supporting)

COMMON = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(COMMON)
MODULES = {"MuSiQue": "MuSiQue", "HotpotQA": "HotpotQA", "2Wiki": "TwoWiki"}


def exclusions(dataset):
    """Ids and supporting-paragraph texts no new question may share, and the train ids tried for the LoRA data."""
    held = []
    for split in ("dev", "test"):
        path = f"{COMMON}/excluded/{dataset}_{split}.pkl"
        if os.path.exists(path):
            held += pickle.load(open(path, "rb"))
    ids = {q["id"] for q in held}
    texts = {p["text"] for q in held for p in q["supporting"]}
    tried = set()
    training = f"{ROOT}/data/train/{dataset}"           # local only (git-ignored); absent in Colab, where we never build
    if os.path.exists(f"{training}/{dataset}_train.jsonl"):
        with open(f"{training}/{dataset}_train.jsonl") as f:
            texts |= {p["text"] for line in f for p in json.loads(line)["paragraphs"]}
        with open(f"{training}/log.jsonl") as f:
            tried = {json.loads(line)["id"] for line in f if line.strip()}
    elif dataset != "2Wiki":
        raise SystemExit(f"{training} is missing: it is needed to exclude the LoRA training questions")
    return ids, texts, tried


def independent(records, taken):
    """Keep a question only if it shares no supporting paragraph and no gold answer with one already taken."""
    out = []
    for r in records:
        texts, answer = {p["text"] for p in r["supporting"]}, normalize(r["answer"])
        if texts & taken["texts"] or answer in taken["answers"]:
            continue
        taken["texts"] |= texts
        taken["answers"].add(answer)
        out.append(r)
    return out


def build(dataset, n_questions=200, k_max=16, out="questions.pkl", seed=0, batch=400, extend=False):
    """extend=True keeps the questions already in `out` unchanged (same facts, same layout) and adds new ones up to
    n_questions; the new ones share no id, supporting paragraph or gold answer with the old ones."""
    module = importlib.import_module(MODULES[dataset])
    ids, texts, tried = exclusions(dataset)
    old, taken = [], {"texts": set(), "answers": set()}
    if extend:
        old = pickle.load(open(out, "rb"))
        assert all(sum(not p["is_supporting"] for p in q["layout"]) >= k_max for q in old), f"{out} has fewer than {k_max} distractors"
        ids |= {q["id"] for q in old}
        independent([{"supporting": [p for p in q["layout"] if p["is_supporting"]], "answer": q["answer"]} for q in old], taken)
        print(f"Extending {out}: {len(old)} questions kept, {max(n_questions - len(old), 0)} to build")
        n_questions -= len(old)
    llm = DeepInfra()
    llm.check()

    def fresh(split):
        pool = [r for r in module.load_pool(seed, split=split, full=True) if r["hops"] == 2]
        kept = [r for r in pool if r["id"] not in ids and r["id"] not in tried
                and not texts & {p["text"] for p in r["paragraphs"] if p["is_supporting"]}
                and sum(not p["is_supporting"] for p in r["paragraphs"]) >= k_max]
        for r in kept:
            r["supporting"], r["source_split"] = supporting(r), split
        print(f"1. {split}: {len(pool)} 2-hop questions, {len(kept)} fresh with at least {k_max} distractors")
        return kept

    def extract(paragraph_list):
        replies = llm.ask_all([EXTRACT.format(title=p["title"], text=p["text"]) for p in paragraph_list],
                              max_tokens=1024, desc="Extracting facts")   # 384 cut long paragraphs mid-sentence
        return [parse_facts(reply or "") for reply in replies]

    def screen(candidates, need, counts, taken):
        answer_and_grade(llm, candidates, "closed_book", lambda r: CLOSED_BOOK.format(question=r["question"]))
        unknown = [r for r in candidates if r["closed_book"]["correct"] is False]
        paragraphs = lambda r: "\n\n".join(f"{p['title']}: {p['text']}" for p in r["supporting"])
        answer_and_grade(llm, unknown, "evidence", lambda r: EVIDENCE.format(paragraphs=paragraphs(r), question=r["question"]))
        answerable = [r for r in unknown if r["evidence"]["correct"] is True]
        trial = {key: set(value) for key, value in taken.items()}
        selected = independent([r for r in select(answerable, per_hop=10 ** 6) if not removal_reason(r)], trial)[:need + 10]
        paragraph_list = [p for r in selected for p in r["paragraphs"]]
        for p, found in zip(paragraph_list, extract(paragraph_list)):
            p["facts"] = found
        complete = []
        for r in selected:
            for s in r["supporting"]:
                s["facts"] = next(p["facts"] for p in r["paragraphs"] if p["text"] == s["text"])
            if all(s["facts"] for s in r["supporting"]) and \
                    sum(not p["is_supporting"] and bool(p["facts"]) for p in r["paragraphs"]) >= k_max:
                complete.append(r)
        complete = complete[:need]
        independent(complete, taken)
        for key, value in [("screened", len(candidates)), ("unknown", len(unknown)), ("answerable", len(answerable)),
                           ("selected", len(selected)), ("complete", len(complete))]:
            counts[key] += value
        return complete

    kept, stats = [], {}
    for split in ("validation", "train"):
        if len(kept) >= n_questions:
            break
        pool, counts, start = fresh(split), Counter(), 0
        while len(kept) < n_questions and start < len(pool):
            chunk = pool[start:start + batch]
            start += len(chunk)
            kept += screen(chunk, n_questions - len(kept), counts, taken)
            print(f"   {split}: {dict(counts)} → {len(kept)} questions so far")
        stats[split] = dict(counts)

    for r in kept:                                      # 6. layout
        rng = random.Random(r["id"])
        distractors = [p for p in r["paragraphs"] if not p["is_supporting"] and p["facts"]]
        rng.shuffle(distractors)
        layout = [{"title": p["title"], "text": p["text"], "facts": p["facts"], "is_supporting": True, "rank": None}
                  for p in r["supporting"]]
        layout += [{"title": p["title"], "text": p["text"], "facts": p["facts"], "is_supporting": False, "rank": rank}
                   for rank, p in enumerate(distractors[:k_max])]
        for p in layout:
            p["position"] = rng.random()
        r["layout"] = sorted(layout, key=lambda p: p["position"])
        r.pop("paragraphs")

    if old:
        previous = json.load(open(out.replace(".pkl", "_stats.json")))
        stats["extended_from"] = {"questions": len(old), "stats": previous}
        kept = old + kept
    assert len({q["id"] for q in kept}) == len(kept)
    assert len({normalize(q["answer"]) for q in kept}) == len(kept), "two questions share a gold answer"
    assert len({p["text"] for q in kept for p in q["layout"] if p["is_supporting"]}) == 2 * len(kept)
    pickle.dump(kept, open(out, "wb"))
    stats.update(dataset=dataset, questions=len(kept), k_max=k_max, by_source=dict(Counter(r["source_split"] for r in kept)),
                 api=llm.summary())
    json.dump(stats, open(out.replace(".pkl", "_stats.json"), "w"), indent=2)
    print(f"Saved {len(kept)} questions to {out} {stats['by_source']}. API: {llm.summary()}")
    if len(kept) - len(old) < n_questions:
        print(f"Only {len(kept) - len(old)} fresh questions pass every filter.")
    return kept
