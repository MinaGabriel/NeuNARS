# train_data.py: NARS training data (for distillation) from a dataset's TRAIN split, built one question at a time
# and saved as it goes, so a run that stops (lost connection, Ctrl-C, budget) resumes exactly where it left off.
# Run it through gen_musique.py, gen_hotpotqa.py or gen_2wiki.py:
#
#   python gen_musique.py --n 500      keep 500 samples in total
#   python gen_musique.py --n 1000     later: 500 MORE (--n is the total wanted, not how many to add); it continues
#                                      where the last run stopped and never redoes or duplicates a question
#   options: --workers 8 (questions at the same time), --budget 20 (stop once $20 has been spent in total over
#            all runs of this dataset, as recorded in log.jsonl), --quiet (print kept samples only)
#
# Per question (2-hop, train split, sharing no question or paragraph with the dev and test sets):
#   1. free checks   single-term answer, not time-sensitive, 2 supporting paragraphs, not a repeated chain
#   2. closed book   the 70B answers with no context; a question it already knows is dropped (as for the test set)
#   3. evidence      the 70B answers from the supporting paragraphs; a question it gets wrong is dropped
#   4. facts         the 70B lists the facts of the 2 supporting + --distractors (default 2) distractor paragraphs,
#                    in shuffled order; each sample records how many it had ("distractors")
#   5. NARS          ONA.py (hop plan, 3 votes, merged relations) with DeepSeek-V3 as the rule writer (the teacher)
#   6. keep          only if NARS's answer is an EXACT match of a gold answer
#
# Output in data/train/<dataset>/:
#   <dataset>_train.jsonl   kept samples: paragraphs and facts, triples, plan, rules, the Narsese ONA ran, ONA's
#                           output, and every LLM call (prompt and reply), ready for fine-tuning
#   log.jsonl               every question tried: kept, or why not, and its cost (also the cost of attempts cut off by
#                           the network, as "retry"). Resuming skips every id it has finished; the total spent is
#                           the sum of its costs, at live DeepInfra prices from the tokens each response reports.
#   surplus.jsonl           correct samples that finished after --n was reached; a later, larger --n uses them first
import argparse
import importlib
import json
import os
import random
import re
import time
import unicodedata
import urllib.request
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / "common"))

import openai
from openai import OpenAI
from tqdm import tqdm

import ONA
from keys import deepinfra_key
from llm import MODEL as FILTER_MODEL
from pipeline import CLOSED_BOOK, EVIDENCE, EXTRACT, JUDGE, TIME_SENSITIVE, is_correct, parse_facts, removal_reason

ROOT = Path(__file__).resolve().parent.parent   # the repository: data/train/ lives there
NARS_MODEL = "deepseek-ai/DeepSeek-V3"  # the teacher: the rule writer of the NARS reported on the test sets
ONA_OPTIONS = {"plan": True, "votes": 3, "merge_relations": True}
ONA_PATH = "/Users/mina/Developer/OpenNARS-for-Applications/NAR"
MODULES = {"MuSiQue": "MuSiQue", "HotpotQA": "HotpotQA", "2Wiki": "TwoWiki"}
HOPS, DISTRACTORS, SEED = 2, 2, 0   # DISTRACTORS is set by --distractors

G, R, Y, C, M, B, D, X = ("\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[35m", "\033[1m", "\033[2m", "\033[0m")


class NetworkError(Exception):
    """The API could not be reached: the question is not logged, so it is tried again (now or on the next run)."""


# ── API: one client, live prices, every call costed ─────────────────────────────────────────────────────────────────
def api_key():
    return deepinfra_key()


CLIENT = OpenAI(api_key=api_key(), base_url="https://api.deepinfra.com/v1/openai", max_retries=2, timeout=90)


def load_prices():
    """Live DeepInfra prices, USD per million tokens (input, output); {} if the lookup fails."""
    for _ in range(3):
        try:
            with urllib.request.urlopen("https://api.deepinfra.com/models/list", timeout=20) as response:
                return {m["model_name"]: (m["pricing"]["cents_per_input_token"] * 1e4,
                                          m["pricing"]["cents_per_output_token"] * 1e4)
                        for m in json.load(response) if (m.get("pricing") or {}).get("type") == "tokens"}
        except Exception:
            time.sleep(2)
    return {}


PRICES = load_prices()  # checked in main(): no run starts without live prices, so every cost is exact


def cost(model, input_tokens, output_tokens):
    price = PRICES[model]
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def chat(prompt, max_tokens):
    """One filter-model call -> (reply, cost). Retries for about a minute; NetworkError if the API stays down."""
    for attempt in range(6):
        try:
            response = CLIENT.chat.completions.create(model=FILTER_MODEL, temperature=0, max_tokens=max_tokens,
                                                      messages=[{"role": "user", "content": prompt}])
            usage = response.usage
            return ((response.choices[0].message.content or "").strip(),
                    cost(FILTER_MODEL, usage.prompt_tokens or 0, usage.completion_tokens or 0))
        except openai.BadRequestError:
            raise  # the request itself is wrong (e.g. too long): retrying cannot help
        except Exception:
            time.sleep(min(32, 2 ** attempt))
    raise NetworkError("the API did not answer for about a minute")


# ── Scoring: the notebook's exact match ─────────────────────────────────────────────────────────────────────────────
def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(a|an|the)\b", " ", text))


def exact(answer, golds):
    return bool(answer) and any(normalize(answer) == normalize(g) for g in golds)


STOPWORDS = {"the", "and", "for", "with", "who", "what", "which", "when", "where", "was", "did", "does", "from", "film"}


def from_question(rules, question):
    """True if every rule starts from entities named in the question (as in the notebook): no hop filled in by the LLM."""
    words = lambda text: set(re.findall(r"[a-z0-9]{3,}", unicodedata.normalize("NFKD", str(text)).encode(
        "ascii", "ignore").decode().lower().replace("_", " ")))
    q_words = words(question) - STOPWORDS
    for rule in rules:
        constants = [x for c in rule["conditions"] for x in (c[0], c[2]) if not x.startswith("?")]
        if not constants or not all(words(x) & q_words for x in constants):
            return False
    return bool(rules)


# ── The question pool ───────────────────────────────────────────────────────────────────────────────────────────────
def held_out(dataset):
    """Ids and paragraph texts of the dev and test sets: a training question may share neither."""
    import pickle
    ids, texts = set(), set()
    for name in ("test", "dev"):
        path = ROOT / "common" / "excluded" / f"{dataset}_{name}.pkl"
        if path.exists():
            for q in pickle.load(open(path, "rb")):
                ids.add(q["id"])
                texts.update(p["text"] for p in q.get("paragraphs") or q.get("supporting") or [])
    evaluation = ROOT / dataset / "questions.pkl"       # the experiment's questions: never train on them or their paragraphs
    if evaluation.exists():
        for q in pickle.load(open(evaluation, "rb")):
            ids.add(q["id"])
            texts.update(p["text"] for p in q["layout"])
    return ids, texts


def build_pool(dataset):
    """The train questions in a fixed shuffled order, after the free checks. Each comes with its 2 supporting and
    2 distractor paragraphs, already in the order NARS will read them. Returns (pool, counts of free skips)."""
    ids, texts = held_out(dataset)
    if not ids:
        raise SystemExit(f"No evaluation set ({dataset}/questions.pkl) or dev/test set for {dataset}: build the evaluation questions first, so training can never overlap them.")
    skipped, chains, pool = Counter(), set(), []
    for q in importlib.import_module(MODULES[dataset]).load_pool(SEED, split="train"):
        support = [p for p in q["paragraphs"] if p["is_supporting"] and p["text"].strip()]
        chain = frozenset(p["text"] for p in support)
        if q["hops"] != HOPS or len(support) != 2:
            reason = "not a 2-hop chain"
        elif not q["answer"].strip() or removal_reason(q):
            reason = "answer is not a single term"
        elif TIME_SENSITIVE.search(q["question"]):
            reason = "time-sensitive question"
        elif q["id"] in ids or chain & texts:
            reason = "shared with dev/test"
        elif chain in chains:
            reason = "repeated chain"
        else:
            reason = None
        if reason:
            skipped[reason] += 1
            continue
        chains.add(chain)
        rng = random.Random(q["id"])
        others = [p for p in q["paragraphs"] if not p["is_supporting"] and p["text"].strip() and p["text"] not in texts]
        shown = support + rng.sample(others, min(DISTRACTORS, len(others)))
        rng.shuffle(shown)
        pool.append({**q, "shown": [{"title": p["title"], "text": p["text"], "is_supporting": p["is_supporting"]}
                                    for p in shown]})
    return pool, skipped


# ── One question, start to finish ───────────────────────────────────────────────────────────────────────────────────
def process(q):
    """Run every step for one question. Returns its log entry, and the training sample if it is kept."""
    golds = [q["answer"], *(q.get("gold_answers") or [])]
    costs = Counter()

    def graded(answer, step):
        """String match, else the 70B judge (as in pipeline.py)."""
        if is_correct(answer, golds):
            return True
        verdict, c = chat(JUDGE.format(question=q["question"], gold=" / ".join(golds), answer=answer), 3)
        costs[step] += c
        return verdict.lower().startswith("yes")

    def entry(status, reason, **extra):
        return {"id": q["id"], "status": status, "reason": reason, "cost": round(sum(costs.values()), 6),
                "costs": {k: round(v, 6) for k, v in costs.items()}, **extra}

    try:
        answer, c = chat(CLOSED_BOOK.format(question=q["question"]), 32)
        costs["closed_book"] += c
        closed_book = answer.split("\n")[0].strip()
        if closed_book and graded(closed_book, "closed_book"):
            return entry("rejected", "70B already knows it"), None

        support = [p for p in q["shown"] if p["is_supporting"]]
        answer, c = chat(EVIDENCE.format(paragraphs="\n\n".join(f"{p['title']}: {p['text']}" for p in support),
                                         question=q["question"]), 32)
        costs["evidence"] += c
        evidence = answer.split("\n")[0].strip()
        if not evidence or not graded(evidence, "evidence"):
            return entry("rejected", "paragraphs do not answer it"), None

        paragraphs = []
        for p in q["shown"]:
            reply, c = chat(EXTRACT.format(title=p["title"], text=p["text"]), 1024)   # 384 cut long paragraphs
            costs["facts"] += c
            paragraphs.append({**p, "facts": parse_facts(reply)})
        if not all(p["facts"] for p in paragraphs if p["is_supporting"]):
            return entry("rejected", "no facts from a supporting paragraph"), None
    except openai.BadRequestError as error:
        return entry("rejected", f"API refused the request ({str(error)[:60]})"), None
    except NetworkError as error:
        error.spent = sum(costs.values())
        raise

    ona = ONA.OnaReasoner(CLIENT, model=NARS_MODEL, **ONA_OPTIONS)
    nars_answer, nars_error = None, None
    try:
        nars_answer = ona.solve([p["facts"] for p in paragraphs if p["facts"]], q["question"], ONA_PATH)
    except Exception as error:
        nars_error = f"{type(error).__name__}: {str(error)[:200]}"
    costs["nars"] += cost(NARS_MODEL, ona.usage["input_tokens"], ona.usage["output_tokens"])
    if ona.usage["failed_requests"]:  # a vote lost to the network is not NARS's mistake: try again later
        error = NetworkError("an ONA request failed")
        error.spent = sum(costs.values())
        raise error
    if not exact(nars_answer, golds):
        reason = "NARS error" if nars_error else "NARS gave no answer" if not nars_answer else "NARS answer not exact"
        return entry("rejected", reason, nars_answer=nars_answer, gold=q["answer"]), None

    sample = {
        "id": q["id"], "dataset": q["dataset"], "split": "train", "source": q.get("source"),
        "question": q["question"], "answer": q["answer"], "gold_answers": golds, "hops": q["hops"],
        "type": q.get("type"), "decomposition": q.get("decomposition"),
        "paragraphs": paragraphs,  # in the order NARS read them: 2 supporting + DISTRACTORS distractors
        "distractors": DISTRACTORS,
        "filters": {"closed_book": closed_book, "evidence": evidence},
        "nars": {
            "answer": nars_answer, "votes": [s["answer"] for s in ona.samples], "plan": ona.plan_text,
            "triples": ona.last_triples, "aliases": {a: c for a, c in ona.aliases.items() if a != c},
            "answer_type": ona.answer_type, "rules": ona.last_rules,
            "rules_reply": ona.rule_attempts[-1][0] if ona.rule_attempts else None,
            "auto_repairs": dict(ona.auto_repairs), "from_question": from_question(ona.last_rules, q["question"]),
            "candidates": [ona.names.get(c, c) for c in ona.candidates],
            "narsese": ona.last_narsese, "ona_output": ona.last_output,
        },
        "llm_calls": ona.request_log,  # every NARS prompt and reply, in order
        "models": {"filters_and_facts": FILTER_MODEL, "nars": NARS_MODEL}, "ona_options": ONA_OPTIONS,
        "cost": round(sum(costs.values()), 6), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return entry("kept", "exact match", nars_answer=nars_answer,
                 from_question=sample["nars"]["from_question"]), sample


# ── Files: append-only, flushed to disk, safe to resume ─────────────────────────────────────────────────────────────
def read_jsonl(path):
    """Records of a JSONL file; a line cut off by a crash is dropped from the file so appends stay clean."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[:raw.rfind(b"\n") + 1]
        rewrite(path, raw.decode("utf-8").splitlines())
    records = []
    for line in raw.decode("utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def rewrite(path, lines):
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(line if line.endswith("\n") else line + "\n" for line in lines), encoding="utf-8")
    os.replace(tmp, path)


def append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def dedupe(path, records):
    """Keep the first record per id (and fix the file if it ever had a repeat)."""
    seen, unique = set(), []
    for r in records:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
    if len(unique) < len(records):
        rewrite(path, [json.dumps(r, ensure_ascii=False, default=str) for r in unique])
    return unique


# ── The run ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def box(title, lines, width=88):
    print(f"{C}╭─ {B}{title}{X}{C} {'─' * (width - len(title) - 4)}╮{X}")
    for line in lines:
        visible = re.sub(r"\033\[[0-9;]*m", "", line)
        if len(visible) > width - 2:  # long lines (e.g. many rejection reasons) are shown plain and cut to fit
            line = visible = visible[:width - 3] + "…"
        print(f"{C}│{X} {line}{' ' * max(0, width - 2 - len(visible))}{C}│{X}")
    print(f"{C}╰{'─' * (width - 1)}╯{X}")


def main(dataset):
    parser = argparse.ArgumentParser(description=f"NARS training data from {dataset} (train split). Resumable.")
    parser.add_argument("--n", type=int, required=True,
                        help="samples wanted in total, counting earlier runs (500 now, --n 1000 later adds 500 more)")
    parser.add_argument("--workers", type=int, default=8, help="questions processed at the same time")
    parser.add_argument("--budget", type=float,
                        help="stop once this many USD have been spent on this dataset in total, earlier runs included")
    parser.add_argument("--quiet", action="store_true", help="print only kept samples, not rejections")
    parser.add_argument("--distractors", type=int, default=2,
                        help="distractor paragraphs per question (2 was the first training set; 4 or 8 add noise)")
    args = parser.parse_args()
    global DISTRACTORS
    DISTRACTORS = args.distractors

    missing = [m for m in (FILTER_MODEL, NARS_MODEL) if m not in PRICES]
    if missing:
        raise SystemExit(f"No live DeepInfra price for {missing}: not starting, so that every cost stays exact. "
                         "Check the connection and run again.")
    out = ROOT / "data" / "train" / dataset
    out.mkdir(parents=True, exist_ok=True)
    kept_path, log_path, surplus_path = out / f"{dataset}_train.jsonl", out / "log.jsonl", out / "surplus.jsonl"

    # Resume: what is already on disk.
    kept = dedupe(kept_path, read_jsonl(kept_path))
    surplus = dedupe(surplus_path, read_jsonl(surplus_path))
    log = read_jsonl(log_path)
    kept_ids = {r["id"] for r in kept}
    promote = [r for r in surplus if r["id"] not in kept_ids and r.get("distractors", 2) == DISTRACTORS][:max(0, args.n - len(kept))]   # same distractor count only
    for r in promote:  # samples paid for in an earlier run go in first
        append(kept_path, r)
        kept_ids.add(r["id"])
    if promote:
        rewrite(surplus_path, [json.dumps(r, ensure_ascii=False, default=str) for r in surplus if r["id"] not in kept_ids])
    seen = kept_ids | {r["id"] for r in surplus} | {e["id"] for e in log if e["status"] in ("kept", "rejected", "surplus")}
    spent_before = sum(e.get("cost", 0) for e in log)
    reasons = Counter(e["reason"] for e in log if e["status"] == "rejected")
    tried_before = sum(e["status"] in ("kept", "rejected", "surplus") for e in log)
    n_kept = len(kept_ids)
    if n_kept >= args.n:
        print(f"{G}✓ Already {n_kept} samples in {kept_path.relative_to(ROOT)} (target {args.n}). Nothing to do.{X}")
        return

    print(f"{D}Loading the {dataset} train split and the dev/test sets (the first time downloads the dataset)...{X}")
    pool, skipped = build_pool(dataset)
    todo = deque(q for q in pool if q["id"] not in seen)
    box(f"NARS training data · {dataset} · train split", [
        f"target {B}{args.n}{X} samples · already kept {B}{n_kept}{X} · tried {tried_before} · spent "
        f"{B}${spent_before:.2f}{X}" + (f" · budget ${args.budget:.2f}" if args.budget else ""),
        f"pool {len(pool):,} questions ({len(todo):,} not tried yet) · free skips: "
        + ", ".join(f"{k} {v:,}" for k, v in skipped.most_common()),
        f"facts {FILTER_MODEL.split('/')[-1]} · NARS {NARS_MODEL.split('/')[-1]} · ONA plan + "
        f"{ONA_OPTIONS['votes']} votes + merged relations",
        f"prices (live, $/1M in/out): " + " · ".join(f"{m.split('/')[-1]} {PRICES[m][0]:.2f}/{PRICES[m][1]:.2f}"
                                                     for m in (FILTER_MODEL, NARS_MODEL)),
        f"keep: NARS exact match · paragraphs: 2 supporting + {DISTRACTORS} distractors · {args.workers} workers · "
        f"kept so far by distractors: {dict(Counter(r.get('distractors', 2) for r in kept))}",
        f"output {kept_path.relative_to(ROOT)}",
    ])
    if not todo:
        print(f"{Y}Every question in the pool has been tried: {n_kept} samples kept.{X}")
        return

    bar_kept = tqdm(total=args.n, initial=n_kept, desc=f"{G}🧠 kept {X}", unit=" sample", colour="green",
                    position=0, dynamic_ncols=True,
                    bar_format="{desc}{bar} {n_fmt}/{total_fmt} {percentage:3.0f}% · {elapsed}<{remaining} · {rate_fmt}")
    bar_tried = tqdm(total=len(todo) + tried_before, initial=tried_before, desc=f"{C}🔎 tried{X}", unit=" q",
                     colour="cyan", position=1, dynamic_ncols=True,
                     bar_format="{desc}{bar} {n_fmt}/{total_fmt} questions · {rate_fmt}")
    status = tqdm(total=0, position=2, bar_format="{desc}", dynamic_ncols=True)
    tried_now = {"n": 0, "kept": 0}

    def spent():
        return spent_before + sum(e.get("cost", 0) for e in session)

    def refresh():
        total = spent()
        per_sample = total / n_kept if n_kept else 0
        yield_ = 100 * tried_now["kept"] / tried_now["n"] if tried_now["n"] else 0
        to_go = per_sample * max(0, args.n - n_kept)
        top = " · ".join(f"{k} {v}" for k, v in reasons.most_common(4))
        status.set_description_str(
            f"{M}💰 ${total:.3f}{X} spent · {B}${per_sample:.4f}{X}/sample · ≈ {B}${to_go:.2f}{X} to reach {args.n}"
            f" · yield {yield_:.0f}% this run · {D}✗ {top}{X}" + (f" · {R}⚡ net errors {net['errors']}{X}" if net["errors"] else ""))

    def say(text):
        if not args.quiet or text.startswith(G):
            tqdm.write(text)

    session, net, crashes = [], {"errors": 0, "streak": 0}, {"n": 0}
    refresh()
    stop_reason = None
    executor = ThreadPoolExecutor(args.workers)
    running = {}

    def fill():
        while not stop_reason and len(running) < args.workers and todo:
            q = todo.popleft()
            running[executor.submit(process, q)] = q

    fill()
    interrupted = False
    while running:
        try:
            done, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
        except KeyboardInterrupt:
            if interrupted:
                print(f"\n{R}Stopped now. Everything finished so far is saved; rerun to resume.{X}")
                os._exit(1)
            interrupted, stop_reason = True, "Ctrl-C"
            tqdm.write(f"{Y}⏸  Finishing the {len(running)} questions in progress (Ctrl-C again to quit now)...{X}")
            continue
        for future in done:
            q = running.pop(future)
            try:
                entry, sample = future.result()
            except NetworkError as error:
                retry = {"id": q["id"], "status": "retry", "reason": "network", "cost": round(getattr(error, "spent", 0), 6)}
                append(log_path, retry)  # its cost counts; the question stays untried and is run again
                session.append(retry)
                net["errors"] += 1
                net["streak"] += 1
                todo.append(q)  # try it again later
                say(f"{R}⚡ network  {q['question'][:70]}  ({error}){X}")
                if net["streak"] >= 2 * args.workers:
                    stop_reason = "the connection looks down"
                refresh()
                continue
            except Exception as error:  # a bug, not the question's fault: not logged, and 3 in a row stop the run
                crashes["n"] += 1
                todo.append(q)
                tqdm.write(f"{R}💥 {type(error).__name__}: {str(error)[:200]}{X}")
                if crashes["n"] >= 3:
                    stop_reason = "3 unexpected errors (see above)"
                continue
            net["streak"], crashes["n"] = 0, 0
            if sample is not None and n_kept >= args.n:
                entry["status"] = "surplus"
                append(surplus_path, sample)
            elif sample is not None:
                append(kept_path, sample)  # the sample first: a crash between the two writes still counts it as seen
                n_kept += 1
                tried_now["kept"] += 1
                bar_kept.update(1)
            append(log_path, entry)
            session.append(entry)
            tried_now["n"] += 1
            bar_tried.update(1)
            if entry["status"] == "rejected":
                reasons[entry["reason"]] += 1
                got = f"  got {entry['nars_answer']!r} for {entry['gold']!r}" if entry.get("nars_answer") else ""
                say(f"{D}✗ {entry['reason']:<34}{X} {D}{q['question'][:60]}{got}  ${entry['cost']:.4f}{X}")
            else:
                tag = "from question" if entry.get("from_question") else "LLM filled a hop"
                say(f"{G}✓ {'KEPT' if entry['status'] == 'kept' else 'SURPLUS'} {n_kept:>5}/{args.n}{X}  "
                    f"{q['question'][:60]}  → {B}{entry['nars_answer']}{X}  {D}[{tag}] ${entry['cost']:.4f}{X}")
            if n_kept >= args.n and not stop_reason:
                stop_reason = "target reached"
            if args.budget and spent() >= args.budget and not stop_reason:
                stop_reason = f"budget ${args.budget:.2f} reached"
            refresh()
        fill()
    executor.shutdown(wait=False, cancel_futures=True)
    for bar in (bar_kept, bar_tried, status):
        bar.close()
    if not stop_reason:
        stop_reason = "the pool is exhausted" if not todo else "stopped"

    total = spent()
    box(f"Done: {stop_reason}", [
        f"kept {B}{n_kept}{X}/{args.n} samples · this run: tried {tried_now['n']}, kept {tried_now['kept']}"
        + (f" (+{len(promote)} from surplus)" if promote else ""),
        f"spent {M}{B}${total:.3f}{X} in total ({M}${total - spent_before:.3f}{X} this run) · "
        f"${total / max(1, n_kept):.4f} per kept sample",
        "rejected: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()),
        f"files: {kept_path.relative_to(ROOT)}, {log_path.name}"
        + (f", {surplus_path.name}" if surplus_path.exists() else ""),
        f"{D}rerun the same command any time: it continues where it stopped{X}" if n_kept < args.n else
        f"{G}target reached{X}",
    ])
