"""Turn the NARS training samples into chat files for LoRA training.

Each kept sample holds every DeepSeek call that led to a correct answer. One example per prompt, and the target is
what ONA.py actually used, not DeepSeek's raw reply:
  triples  each fact chunk → its triples (the TRIPLE_REPAIR reply only where ONA.py kept it)
  aliases  the names → their groups, minus the merges ONA.py refused (plausible_alias)
  plan     the question → its hops
  rules    the winning vote's prompt → the rules after ONA.py validated and auto-repaired them, written back in the
           prompt's names (no answer written into a rule, no constant used as a bridge, no rule that did not run)
Truncated replies are dropped. JSON targets are bare JSON, one item per line.

    python make_sft.py                         # MuSiQue + HotpotQA (2Wiki held out)
    python make_sft.py --datasets MuSiQue HotpotQA 2Wiki
    python make_sft.py --rules-copies 1        # no upsampling of the rules examples

Writes data/sft/train.jsonl and data/sft/val.jsonl, split by question (no question in both). The split hashes the
question id, so a question stays on the same side when more samples are generated later.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # the repository: data/ lives there
sys.path.insert(0, str(ROOT / "common"))

from ONA import OnaReasoner, atom, is_var, missing_terms, plausible_alias
from train_data import from_question

TASKS = {"Convert the facts below": "triples", "Group the names below": "aliases",
         "Break the question into hops": "plan", "Write rules that answer": "rules"}


def task_of(prompt):
    return next((task for start, task in TASKS.items() if prompt.startswith(start)), None)


def json_lines(items):
    """A JSON list with one item per line: compact, and easy to read."""
    if not items:
        return "[]"
    return "[\n" + ",\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n]"


def last_block(prompt, header, end=None):
    """The lines of the prompt's last `header` block (the real input, after the few-shot examples)."""
    block = prompt[prompt.rfind(header) + len(header):]
    return block.split(end)[0].strip().splitlines() if end else block.strip().splitlines()


def triples_target(calls):
    """The triples ONA.py kept for one fact chunk: the repair only if it left fewer names out."""
    first, repair = calls[0], next((c for c in calls[1:] if len(c["messages"]) > 1), None)
    facts = [re.sub(r"^\d+\.\s*", "", line) for line in last_block(first["messages"][0]["content"], "Facts:", "Answer:")]
    if first["finish_reason"] != "stop":
        return None
    triples = OnaReasoner._parse_triples(first["response"])
    if repair and repair["finish_reason"] == "stop":
        repaired = OnaReasoner._parse_triples(repair["response"])
        if repaired and len(missing_terms(facts, repaired)) < len(missing_terms(facts, triples)):
            triples = repaired
    return json_lines(triples) if triples else None


def aliases_target(call):
    """DeepSeek's groups split into the merges ONA.py accepts (plausible_alias), as ONA.py's union does."""
    if call["finish_reason"] != "stop":
        return None
    names = {n.casefold(): n for n in last_block(call["messages"][0]["content"], "Names:")}
    groups = []
    for group in parse_groups(call["response"]):
        members = sorted({names[x.strip().casefold()] for x in group if isinstance(x, str) and x.strip().casefold() in names})
        parent = {m: m for m in members}
        find = lambda x: x if parent[x] == x else find(parent[x])
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if plausible_alias(a, b):
                    parent[find(a)] = find(b)
        parts = collections.defaultdict(list)
        for m in members:
            parts[find(m)].append(m)
        groups += [p for p in parts.values() if len(p) > 1 and p not in groups]
    return json_lines(groups)


def parse_groups(raw):
    """The last list of lists in the reply (DeepSeek sometimes corrects itself mid-reply), as ONA.py reads it."""
    decoder, found, flat = json.JSONDecoder(), [], []
    for i, ch in enumerate(raw):
        if ch == "[":
            try:
                value = decoder.raw_decode(raw[i:])[0]
            except json.JSONDecodeError:
                continue
            if isinstance(value, list) and any(isinstance(x, list) for x in value):
                found.append([x for x in value if isinstance(x, list)])
            elif isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                flat.append(value)
    return found[-1] if found else flat


def rules_target(sample, prompt):
    """The validated rules in the prompt's names, variables renamed ?A, ?B, ... in order of appearance."""
    names = {}
    for line in last_block(prompt, "Triples:", "Answer:"):
        for part in line.split(" | "):
            names.setdefault(atom(part.strip()), part.strip())
    rules = []
    for rule in sample["nars"]["rules"]:
        variables = {}
        def name(x):
            if is_var(x):
                return variables.setdefault(x, "?" + chr(ord("A") + len(variables)))
            return names[x]
        conditions = [[name(x) for x in condition] for condition in rule["conditions"]]
        rules.append({"conditions": conditions, "answer": name(rule["answer"])})
    reply = '{"answer_type": ' + json.dumps(sample["nars"]["answer_type"]) + ', "rules": ' + json_lines(rules) + "}"
    return reply, rules


def examples(sample, rules_copies):
    """The training examples of one sample, and the reason any of its calls were left out."""
    by_prompt, skipped = collections.defaultdict(list), collections.Counter()
    for call in sample["llm_calls"]:
        by_prompt[call["messages"][0]["content"]].append(call)
    rows = []
    for prompt, calls in by_prompt.items():
        task = task_of(prompt)
        if task == "triples":
            reply = triples_target(calls)
        elif task == "aliases":
            reply = aliases_target(calls[0])
        elif task == "plan":
            reply = "\n".join(line.strip() for line in calls[0]["response"].strip().splitlines())
            reply = reply if calls[0]["finish_reason"] == "stop" else None
        else:
            continue
        if reply is None:
            skipped[f"{task} truncated or empty"] += 1
            continue
        rows.append((task, prompt, reply))
    winner = next(c for c in sample["llm_calls"] if c["response"] == sample["nars"]["rules_reply"])
    reply, rules = rules_target(sample, winner["messages"][0]["content"])
    if from_question(rules, sample["question"]):
        rows += [("rules", winner["messages"][0]["content"], reply)] * rules_copies
    else:
        skipped["rules not from question"] += 1
    return [{"id": sample["id"], "dataset": sample["dataset"], "task": task,
             "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]}
            for task, prompt, reply in rows], skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["MuSiQue", "HotpotQA"])
    parser.add_argument("--rules-copies", type=int, default=2,
                        help="copies of each rules example in train: rules are the hardest job and the fewest tokens")
    parser.add_argument("--val", type=float, default=0.1, help="share of questions held out for validation")
    args = parser.parse_args()

    samples = [json.loads(line) for dataset in args.datasets
               for line in open(ROOT / "data" / "train" / dataset / f"{dataset}_train.jsonl")]
    is_val = lambda sample: int(hashlib.md5(sample["id"].encode()).hexdigest(), 16) % 1000 < args.val * 1000
    splits = {"val": [s for s in samples if is_val(s)], "train": [s for s in samples if not is_val(s)]}

    out = ROOT / "data" / "sft"
    out.mkdir(parents=True, exist_ok=True)
    for name, group in splits.items():
        rows, skipped = [], collections.Counter()
        for sample in group:
            more, why = examples(sample, args.rules_copies if name == "train" else 1)
            rows += more
            skipped += why
        with open(out / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tasks = collections.Counter(row["task"] for row in rows)
        chars = collections.Counter()
        for row in rows:
            chars[row["task"]] += len(row["messages"][1]["content"])
        share = {task: f"{chars[task] / sum(chars.values()):.0%}" for task in chars}
        datasets = collections.Counter(sample["dataset"] for sample in group)
        print(f"{name}: {len(group)} questions {dict(datasets)} → {len(rows)} examples {dict(tasks)}")
        print(f"   share of reply tokens {share} · left out {dict(skipped)}")
    print(f"wrote {out}/train.jsonl and {out}/val.jsonl")


if __name__ == "__main__":
    main()
