"""Where NARS fails: sorts every wrong or missing NARS answer into one pipeline stage, per k.

    python common/error_analysis.py MuSiQue "NARS_8B_LoRA_v2"            # counts per k
    python common/error_analysis.py MuSiQue "NARS_8B_LoRA_v2" --examples 3

Stages, checked in this order (the first that applies wins):
  timeout          ONA ran out of time (or another crash)
  rules: no join   every rule condition matches some triple, but no entity links the hops
  rules: other     the rules were rejected for another reason (not chained, unknown relation, ...)
  gold not read    the gold answer is not in any fact of the supporting paragraphs
  gold not in KB   the gold answer is in the facts but in no triple (triple extraction)
  bridge not in KB the hop-1 answer (from the MuSiQue decomposition) is in no triple
  near miss        the answer and the gold overlap, but the judge said no
  distractor       the answer only comes from distractor paragraphs
  wrong rule       the answer comes from a supporting paragraph: right facts, wrong chain
  no trace         the trace line is missing or broken, so the stage is unknown
"""
import argparse
import collections
import json
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["timeout", "rules: no join", "rules: other", "gold not read", "gold not in KB",
          "bridge not in KB", "near miss", "distractor", "wrong rule", "no trace"]


def norm(s):
    s = re.sub(r"[^a-z0-9]+", " ", str(s).lower().replace("_", " "))
    return re.sub(r"^(the|a|an) ", "", s.strip())


def same(a, b):
    """True when one name contains the other (at least 4 characters), ignoring case and punctuation."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    return a == b or (min(len(a), len(b)) >= 4 and (a in b or b in a))


def last_by_index(path):
    """Answer files can hold retried lines; the last line per question wins."""
    rows = {}
    broken = 0
    for line in filter(str.strip, open(path)):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:            # a line cut by an interrupted write
            broken += 1
            continue
        rows[r["index"]] = r
    if broken:
        print(f"skipped {broken} broken line(s) in {path.parent.name}/{path.name}")
    return rows


def classify(a, t, q):
    if a.get("error") or "answer" not in a:
        return "timeout"
    golds = a.get("golds") or [a["gold"]]
    if a.get("answer") is None:
        last = t["rule_attempts"][-1][1] if t and t.get("rule_attempts") else []
        return "rules: no join" if any("no entities satisfy all of them together" in m for m in last) else "rules: other"
    if t is None:
        return "no trace"
    support = [f for p in q["supporting"] for f in p["facts"]]
    if not any(same(g, f) or norm(g) in norm(f) for g in golds for f in support):
        return "gold not read"
    triples = t["triples"]
    entities = [x for tr in triples for x in (tr["subject"], tr["object"])]
    if not any(same(g, e) for g in golds for e in entities):
        return "gold not in KB"
    bridges = [d["answer"] for d in q["decomposition"][:-1]]
    if bridges and not all(any(same(b, e) for e in entities) for b in bridges):
        return "bridge not in KB"
    if any(same(g, a["answer"]) for g in golds):
        return "near miss"
    sources = [tr.get("fact", "") for tr in triples if same(a["answer"], tr["subject"]) or same(a["answer"], tr["object"])]
    if sources and not any(f in src for src in sources for f in support):
        return "distractor"
    return "wrong rule"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("system", help="folder name under results/answers, e.g. NARS_8B_LoRA_v2")
    p.add_argument("--results", help="results folder (default: <dataset>/results)")
    p.add_argument("--examples", type=int, default=0, help="print this many examples per stage")
    args = p.parse_args()
    results = Path(args.results) if args.results else ROOT / args.dataset / "results"
    questions = {q["id"]: q for q in pickle.load(open(ROOT / args.dataset / "questions.pkl", "rb"))}
    ref = results / "answers" / "Llama_3_1_8B"

    table, examples = {}, collections.defaultdict(list)
    for path in sorted((results / "answers" / args.system).glob("k*.jsonl")):
        k = path.stem
        answers = last_by_index(path)
        traces = last_by_index(results / "traces" / args.system / path.name)
        llama = last_by_index(ref / path.name) if (ref / path.name).exists() else {}
        counts, only_8b = collections.Counter(), collections.Counter()
        for i, a in answers.items():
            if a.get("judge"):
                counts["right"] += 1
                continue
            stage = classify(a, traces.get(i), questions[a["id"]])
            counts[stage] += 1
            if llama.get(i, {}).get("judge"):
                only_8b[stage] += 1
            examples[stage].append((k, i, a, traces.get(i)))
        table[k] = (counts, only_8b, len(answers))

    ks = list(table)
    print(f"{args.system} on {args.dataset}: wrong or missing answers by stage (in brackets: of those, the 8B got right)\n")
    print(f"{'stage':18}" + "".join(f"{k:>14}" for k in ks))
    for s in ["right"] + STAGES:
        cells = [f"{table[k][0][s]}" + ("" if s == "right" else f" [{table[k][1][s]}]") for k in ks]
        print(f"{s:18}" + "".join(f"{c:>14}" for c in cells))
    print(f"{'questions':18}" + "".join(f"{table[k][2]:>14}" for k in ks))

    for s in STAGES:
        for k, i, a, t in examples[s][:args.examples]:
            print(f"\n[{s}] {k} q{i}: {a['question']}\n   gold {a['gold']!r} · NARS {a.get('answer')!r}")
            if t and t.get("rules"):
                print(f"   rule {t['rules'][0]['conditions']}")
            elif t and t.get("rule_attempts"):
                print(f"   rejected: {t['rule_attempts'][-1][1][0][:220]}")


if __name__ == "__main__":
    main()
