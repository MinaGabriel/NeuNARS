# pipeline.py: builds the test set with ONE model (Llama-3.1-70B on DeepInfra), in five steps:
#   1. Pool         every answerable question of the dataset (2-, 3- and 4-hop), shuffled.
#   2. Closed book  the 70B answers with no context. A string match, or else the 70B judge, decides whether it
#                   knew the answer. Questions it knew are dropped: the test needs facts the model does not memorize.
#   3. Evidence     the 70B answers again using ONLY the supporting paragraphs (graded the same way). Questions it
#                   still gets wrong are dropped: the paragraphs must be enough to answer.
#   4. Select       drop time-sensitive questions and repeated chains (same supporting paragraphs); keep up to
#                   --per-hop questions per hop count.
#   5. Facts        the 70B lists the facts in each supporting paragraph. NARS and the baselines answer from these.
#   6. Clean        set aside questions whose gold answer is not one entity, date or number (how/why questions,
#                   lists, sentences, phrases): NARS answers with a single term. Reads only questions and gold answers.
#   7. Distractors  (optional, --distractors) facts of the other paragraphs too, so every system can be run on the
#                   full context (supporting + distractor paragraphs, in the dataset's order) instead of the evidence.
#
# --keep known flips step 2: it keeps the questions the 70B DOES know, for the counterfactual experiment
# (does a system follow edited facts, or answer from memory?). Everything else is the same.
#
# Usage:  DEEPINFRA_API_KEY=... python pipeline.py --dataset MuSiQue --per-hop 150 [--keep known]
#         python pipeline.py --dataset MuSiQue --clean-only     (step 6 again on an existing runs/<dataset>_test.pkl)
#         ... --split dev --distractors 10   step 7 for the 10 questions the notebook samples (a pilot); no N = all
#         ... --split dev --hops 2   a development set for tuning ONA.py: same steps, but never a test question
#                                    or a test paragraph, so tuning on it cannot leak into the test results
# Output: runs/<dataset>_test.pkl          questions + facts (record["supporting"][i]["facts"])
#         runs/<dataset>_test_entity.pkl   after step 6: the questions the experiment notebook reads
#         runs/<dataset>_test_removed.json the set-aside questions, with the reason
#         runs/<dataset>_test_stats.json   counts per step and hop, judge flips, cost, prompts
#         (with --keep known: runs/<dataset>_known.pkl and runs/<dataset>_known_stats.json)
import argparse
import importlib
import json
import os
import pickle
import random
import re
import unicodedata
from collections import Counter

from llm import MODEL, DeepInfra

CLOSED_BOOK = """Answer with only the answer, in as few words as possible. No sentences, no explanations.

Question: Who directed Jaws?
Answer: Steven Spielberg
Question: What is the capital of the country where the Eiffel Tower is?
Answer: Paris

Question: {question}
Answer:"""

EVIDENCE = """Answer the question using only the paragraphs below. Answer with only the answer, in as few words as
possible. No sentences, no explanations.

{paragraphs}

Question: {question}
Answer:"""

JUDGE = """Question: {question}
Gold answer: {gold}
Model answer: {answer}
Does the model answer refer to the same thing as the gold answer (e.g. "Prince Philip" = "Philip Mountbatten")?
Reply with only yes or no."""

EXTRACT = """Extract every fact stated in the paragraph below.
Rules:
- One fact per line, starting with "- ".
- Each fact is one short, self-contained sentence with a single claim.
- Use full names; never use pronouns ("he", "she", "it", "the film").
- Only facts stated in the paragraph; do not add anything you know.

Title: {title}
Paragraph: {text}

Facts:"""

QUANTITY = r"how (many|much|long|old|far|tall|big|large|high|deep)\b"  # "How many ..." wants a number: keep
CONNECTORS = {"of", "for", "the", "and", "in", "de", "la", "von", "van", "on", "at", "-", "&"}
PREPOSITIONS = {"in", "at", "on", "for", "along", "by", "from", "to", "with", "as", "near", "past",
                "during", "after", "before", "since", "until", "around", "into", "through"}

TIME_SENSITIVE = re.compile(r"\b(this season|current|currently|now|this year|latest|recent|recently|today)\b", re.I)


# ---------- helpers ----------

def normalize(text):
    """Lowercase, no accents, no punctuation or articles, no spaces: 'The Fiorello La Guardia.' -> 'fiorellolaguardia'"""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^a-z0-9 ]", " ", text))
    return text.replace(" ", "")


def is_correct(prediction, golds):
    """Correct if the prediction and any gold answer contain each other after normalization."""
    p = normalize(prediction)
    return bool(p) and any(normalize(g) and (normalize(g) in p or p in normalize(g)) for g in golds)


def parse_facts(text):
    """'- Rumi was born in Balkh.\\n- ...' -> ['Rumi was born in Balkh.', ...] (no duplicates, >= 3 words)."""
    facts = [line.strip().lstrip("-*• ").strip() for line in text.splitlines() if line.strip().startswith(("-", "*", "•"))]
    return list(dict.fromkeys(f for f in facts if len(f.split()) >= 3))


def supporting(record):
    """The record's supporting paragraphs, in the layout the experiments read (facts are filled in step 5)."""
    return [{"paragraph_idx": p["idx"], "title": p["title"], "text": p["text"], "facts": []}
            for p in record["paragraphs"] if p["is_supporting"]]


def by_hop(records):
    return dict(sorted(Counter(r["hops"] for r in records).items()))


# ---------- steps ----------

def answer_and_grade(llm, records, key, prompt_of):
    """The 70B answers every record (stored in record[key]); a string match or the 70B judge grades the answer.
    record[key] = {"answer", "correct": True/False/None (None = the API call failed), "graded_by"}."""
    replies = llm.ask_all([prompt_of(r) for r in records], max_tokens=32, desc=f"{key}: answering")
    unsure = []
    for r, reply in zip(records, replies):
        answer = None if reply is None else reply.split("\n")[0].strip()
        r[key] = {"answer": answer, "correct": None, "graded_by": None}
        if answer is not None and is_correct(answer, r["gold_answers"]):
            r[key].update(correct=True, graded_by="string")
        elif answer is not None:
            unsure.append(r)
    verdicts = llm.ask_all([JUDGE.format(question=r["question"], gold=" / ".join(r["gold_answers"]),
                                         answer=r[key]["answer"]) for r in unsure],
                           max_tokens=3, desc=f"{key}: judging")
    for r, verdict in zip(unsure, verdicts):
        if verdict is not None:
            r[key].update(correct=verdict.lower().startswith("yes"), graded_by="judge")
    return {"string_correct": sum(r[key]["graded_by"] == "string" for r in records),
            "judge_correct": sum(r[key]["graded_by"] == "judge" and r[key]["correct"] for r in records),
            "failed_calls": sum(r[key]["correct"] is None for r in records)}


def select(records, per_hop):
    """Drop time-sensitive questions and reworded duplicates, then keep up to per_hop questions per hop count.
    A duplicate is a question over the same supporting paragraphs (the same chain). Sharing only the answer is
    fine: MuSiQue's 3- and 4-hop questions often end at the same answer through different chains."""
    seen, kept, count = set(), [], Counter()
    for r in records:
        key = frozenset(p["text"] for p in r["supporting"])
        if TIME_SENSITIVE.search(r["question"]) or key in seen or count[r["hops"]] >= per_hop:
            continue
        seen.add(key)
        count[r["hops"]] += 1
        kept.append(r)
    return kept


def extract_facts(llm, records):
    """Fill record["supporting"][i]["facts"]. Each unique paragraph is sent once. Returns paragraphs with no facts."""
    unique = list({p["text"]: p for r in records for p in r["supporting"]}.values())
    replies = llm.ask_all([EXTRACT.format(title=p["title"], text=p["text"]) for p in unique],
                          max_tokens=384, desc="Extracting facts")
    facts_of = {p["text"]: parse_facts(reply or "") for p, reply in zip(unique, replies)}
    for r in records:
        for p in r["supporting"]:
            p["facts"] = facts_of[p["text"]]
    return sum(not facts for facts in facts_of.values())


def is_name(text):
    """Mostly capitalized words: "Academy Award for Best Live Action Short Film", "ATS - 6 (...)"."""
    words = [w for w in text.split() if w.lower() not in CONNECTORS and re.search(r"\w", w)]
    return bool(words) and sum(w[0].isupper() or w[0].isdigit() or w[0] == "(" for w in words) >= 0.75 * len(words)


def removal_reason(q):
    """Why a question cannot have a single-term answer, or None to keep it."""
    question = q["question"].strip().lower()
    gold = re.sub(r"^the\s+", "", q["answer"].strip().rstrip("."), flags=re.I)  # "the Coast Range" -> "Coast Range"
    words = gold.split()
    if re.match(r"(why|how)\b", question) and not re.match(QUANTITY, question):
        return "asks how/why"           # "How were ... expelled?" -> a description
    separators = re.sub(r"(?<=\d),(?=\d)", "", gold)  # "1,335,907" is one number, not a list
    if separators.count(",") >= 2 or re.search(r",[^,]*\band\b", separators):
        return "answer is a list"       # "Suzhou pingtan, spoken theatre and puppet theatre"
    if len(words) > 6 and not is_name(gold) and not gold[0].isdigit():  # "3,771.9 people per square mile" is a value
        return "answer is a sentence"   # "It is not possible for foreigners to go to/from Myanmar ..."
    if words[0].lower() in PREPOSITIONS or words[0].endswith("ing") and words[0].islower() \
            or re.search(r"\b(is|was|were|are)\b", gold):
        return "answer is a phrase"     # "in Northern Florida", "trying to find ...", "palace was seldom used"
    return None


def clean(dataset, name="test"):
    """Step 6: runs/<dataset>_<name>.pkl -> <name>_entity.pkl (kept) and <name>_removed.json (set aside, with reason)."""
    questions = pickle.load(open(f"runs/{dataset}_{name}.pkl", "rb"))
    kept, removed = [], []
    for i, q in enumerate(questions):
        reason = removal_reason(q)
        if reason:
            removed.append({"index": i, "id": q["id"], "hops": q["hops"], "reason": reason,
                            "question": q["question"], "gold": q["answer"]})
        else:
            kept.append(q)
    pickle.dump(kept, open(f"runs/{dataset}_{name}_entity.pkl", "wb"))
    json.dump(removed, open(f"runs/{dataset}_{name}_removed.json", "w"), indent=1, ensure_ascii=False)
    print(f"6. Clean: kept {len(kept)} of {len(questions)} {by_hop(kept)}; set aside "
          f"{dict(Counter(r['reason'] for r in removed))} -> runs/{dataset}_{name}_entity.pkl")
    return kept


def add_distractors(llm, dataset, name, hops, sample, seed):
    """Step 7 (optional): record["context"] = every paragraph of the question, supporting and distractor, in the
    dataset's order, each with its facts (supporting paragraphs keep the facts of step 5). With --sample N only the
    questions the notebook picks for N_QUESTIONS = N (same hops and seed) are done, for a cheap pilot."""
    path = f"runs/{dataset}_{name}_entity.pkl"
    questions = pickle.load(open(path, "rb"))
    pool = [i for i, q in enumerate(questions) if q["hops"] == hops]
    picked = sorted(random.Random(seed).sample(pool, min(sample, len(pool)))) if sample else pool
    todo = [questions[i] for i in picked if "context" not in questions[i]]
    unique = list({p["text"]: p for q in todo for p in q["paragraphs"] if not p["is_supporting"]}.values())
    replies = llm.ask_all([EXTRACT.format(title=p["title"], text=p["text"]) for p in unique],
                          max_tokens=384, desc="Extracting distractor facts")
    facts_of = {p["text"]: parse_facts(reply or "") for p, reply in zip(unique, replies)}
    for q in todo:
        facts_of.update({p["text"]: p["facts"] for p in q["supporting"]})
        q["context"] = [{"title": p["title"], "is_supporting": p["is_supporting"], "facts": facts_of[p["text"]]}
                        for p in q["paragraphs"]]
    pickle.dump(questions, open(path, "wb"))
    print(f"7. Distractors: {len(todo)} questions, {len(unique)} distractor paragraphs -> {path}. API: {llm.summary()}")


# ---------- run ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MuSiQue", help="module name of the dataset file, e.g. MuSiQue")
    parser.add_argument("--per-hop", type=int, default=150, help="questions kept per hop count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=16, help="parallel API requests")
    parser.add_argument("--keep", choices=["unknown", "known"], default="unknown",
                        help="unknown: the test set; known: questions the 70B already knows (counterfactual set)")
    parser.add_argument("--split", choices=["test", "dev"], default="test",
                        help="dev: questions for tuning ONA.py, excluding every test question and test paragraph")
    parser.add_argument("--hops", type=int, nargs="*", help="keep only these hop counts, e.g. --hops 2")
    parser.add_argument("--clean-only", action="store_true", help="only run step 6 on an existing runs/<dataset>_<split>.pkl")
    parser.add_argument("--distractors", type=int, nargs="?", const=0, metavar="N",
                        help="only step 7 on an existing <split>_entity.pkl: facts of the distractor paragraphs too; "
                             "with N, just the questions the notebook samples for N_QUESTIONS = N")
    args = parser.parse_args()
    name = args.split if args.keep == "unknown" else "known"
    if args.clean_only:
        clean(importlib.import_module(args.dataset).DATASET, name)
        return

    ds = importlib.import_module(args.dataset)
    llm = DeepInfra(args.model, args.workers)
    llm.check()
    if args.distractors is not None:
        add_distractors(llm, ds.DATASET, name, (args.hops or [2])[0], args.distractors, args.seed)
        return

    # 1. Pool
    records = ds.load_pool(args.seed)
    for r in records:
        r["supporting"] = supporting(r)
    if args.hops:
        records = [r for r in records if r["hops"] in args.hops]
    if args.split == "dev":  # no test question, and no question sharing a paragraph with one
        test_file = f"runs/{ds.DATASET}_test.pkl"
        test_file = test_file if os.path.exists(test_file) else f"runs/{ds.DATASET}_test_entity.pkl"
        test = pickle.load(open(test_file, "rb"))
        test_ids, test_texts = {r["id"] for r in test}, {p["text"] for r in test for p in r["supporting"]}
        records = [r for r in records if r["id"] not in test_ids and not test_texts & {p["text"] for p in r["supporting"]}]
    stats = {"model": args.model, "dataset": ds.DATASET, "seed": args.seed, "per_hop": args.per_hop,
             "pool": by_hop(records)}
    print(f"1. Pool: {len(records)} questions {by_hop(records)}")

    # 2. Closed book: keep the questions the 70B does NOT know (--keep known: the ones it DOES know)
    stats["closed_book"] = answer_and_grade(llm, records, "closed_book",
                                            lambda r: CLOSED_BOOK.format(question=r["question"]))
    wanted = args.keep == "known"
    kept = [r for r in records if r["closed_book"]["correct"] is wanted]
    stats["closed_book"][f"kept_{args.keep}"] = by_hop(kept)
    print(f"2. Closed book: {len(kept)} {args.keep} to the model {by_hop(kept)}")

    # 3. Evidence: keep the questions the supporting paragraphs answer
    paragraphs = lambda r: "\n\n".join(f"{p['title']}: {p['text']}" for p in r["supporting"])
    stats["evidence"] = answer_and_grade(llm, kept, "evidence",
                                         lambda r: EVIDENCE.format(paragraphs=paragraphs(r), question=r["question"]))
    answerable = [r for r in kept if r["evidence"]["correct"] is True]
    stats["evidence"]["kept_answerable"] = by_hop(answerable)
    print(f"3. Evidence: {len(answerable)} answerable from the paragraphs {by_hop(answerable)}")

    # 4. Select
    final = select(answerable, args.per_hop)
    stats["final"] = by_hop(final)
    print(f"4. Selected: {len(final)} questions {by_hop(final)}")

    # 5. Facts
    stats["paragraphs_without_facts"] = extract_facts(llm, final)
    stats["facts_per_question"] = round(sum(len(p["facts"]) for r in final for p in r["supporting"]) / max(1, len(final)), 1)
    print(f"5. Facts: {stats['facts_per_question']} per question")

    out = f"runs/{ds.DATASET}_{name}.pkl"
    pickle.dump(final, open(out, "wb"))
    stats["keep"], stats["split"], stats["hops"] = args.keep, args.split, args.hops
    stats["api"] = llm.summary()
    stats["prompts"] = {"closed_book": CLOSED_BOOK, "evidence": EVIDENCE, "judge": JUDGE, "extract": EXTRACT}
    json.dump(stats, open(f"runs/{ds.DATASET}_{name}_stats.json", "w"), indent=2, ensure_ascii=False)
    print(f"Saved {out}: {len(final)} questions. API: {llm.summary()}")
    clean(ds.DATASET, name)


if __name__ == "__main__":
    main()
