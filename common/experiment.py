"""Shared code for the per-dataset experiments: inputs, systems, running, scoring, storage and the report.

Every notebook in MuSiQue/, HotpotQA/ and 2Wiki/ imports this file. All systems read the same 70B-extracted facts from
<dataset>/questions.pkl (built by that folder's build_data.ipynb): the 2 supporting paragraphs plus the first k of the
question's distractor paragraphs, in a fixed reading order (the sets nest across k).

Results: <dataset>/results/answers/<system>/k<k>.jsonl, one line per answer (inputs read, raw reply, answer, tokens,
cost, time, every metric); NARS rows also write <dataset>/results/traces/<system>/k<k>.jsonl (triples, plan, rules,
votes, Narsese, ONA output, every LLM call). A run resumes where it stopped; failed answers are retried.
"""
import hashlib
import json
import math
import os
import pickle
import random
import re
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from openai import OpenAI

import ONA
from keys import deepinfra_key

# ── Datasets ────────────────────────────────────────────────────────────────────────────────────────────────────────
K_VALUES = {"MuSiQue": [0, 4, 8, 16], "HotpotQA": [0, 4, 8], "2Wiki": [0, 4, 8]}

# ── Systems ─────────────────────────────────────────────────────────────────────────────────────────────────────────
BASELINES = {                                   # LLMs answering directly from the facts (all on DeepInfra)
    "gemma-3-4b": "google/gemma-3-4b-it",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "Mistral-Small-24B": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "gemma-3-27b": "google/gemma-3-27b-it",
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen3-Next-80B-A3B": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
}
SIZES = {"gemma-3-4b": "4B", "gemma-3-12b": "12B", "Mistral-Small-24B": "24B", "gemma-3-27b": "27B",
         "Llama-3.3-70B": "70B", "Qwen3-Next-80B-A3B": "80B (3B active)", "DeepSeek-V3": "671B (37B active)",
         "Llama-3.1-8B": "8B", "Llama-3.1-8B + CoT": "8B", "NARS + 8B + LoRA": "8B", "NARS + 8B + LoRA v2": "8B",
         "Prolog + 8B + LoRA": "8B"}
REFERENCE = "Llama-3.1-8B"                      # the student's own weights answering directly (run_nars_lora.ipynb)
NARS_SYSTEMS = {"NARS + 8B + LoRA", "NARS + 8B + LoRA v2"}
ONA_OPTIONS = {"plan": True, "votes": 3, "merge_relations": True}
ONA_COMMIT = "dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a"
ADAPTER_REPO = "MinaGabriel/nars-llama31-8b-lora"
JUDGE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"   # not one of the systems


def deepinfra():
    return OpenAI(api_key=deepinfra_key(), base_url="https://api.deepinfra.com/v1/openai", max_retries=3, timeout=120)


def check_models(pairs):
    """[(client, model)]: stop unless every model answers under its own name (DeepInfra has silently served retired
    models with another one)."""
    for client, model in pairs:
        served = client.chat.completions.create(model=model, max_tokens=1, messages=[{"role": "user", "content": "Hi"}]).model
        assert served == model, f"{model} is served as {served}"
    print(f"{len(pairs)} models answer under their own names")


def load_prices():
    try:
        with urllib.request.urlopen("https://api.deepinfra.com/models/list", timeout=20) as response:
            return {m["model_name"]: (m["pricing"]["cents_per_input_token"] * 1e4, m["pricing"]["cents_per_output_token"] * 1e4)
                    for m in json.load(response) if (m.get("pricing") or {}).get("type") == "tokens"}
    except Exception:
        return {}


PRICES = load_prices()


def cost(input_tokens, output_tokens, model):   # models served on our own GPU cost $0 here
    price = PRICES.get(model, (0, 0))
    return ((input_tokens or 0) * price[0] + (output_tokens or 0) * price[1]) / 1_000_000


# ── Questions and inputs ────────────────────────────────────────────────────────────────────────────────────────────
def load_questions(path="questions.pkl", n=None):
    """The dataset's questions (the first n, in their fixed order) and the file's hash."""
    with open(path, "rb") as f:
        data = f.read()
    questions = pickle.loads(data)
    return questions[:n] if n else questions, hashlib.sha256(data).hexdigest()


def paragraphs_at(q, k):
    """The 2 supporting paragraphs and the first k distractors, in their fixed reading order (nested in k)."""
    return [p for p in q["layout"] if p["is_supporting"] or p["rank"] < k]


def facts(q, k):
    return [p["facts"] for p in paragraphs_at(q, k)]


# ── Scoring ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def golds(q):
    return list(dict.fromkeys([q["answer"], *(q.get("gold_answers") or [])]))


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(a|an|the)\b", " ", text))


def exact(answer, g):
    return any(normalize(answer or "") == normalize(x) for x in g)


def lenient(answer, g):
    """One contains the other, the shorter covering at least half of the longer."""
    a = normalize(answer or "")
    for x in map(normalize, g):
        short, long_ = sorted((a, x), key=len)
        if short and short in long_ and 2 * len(short) >= len(long_):
            return True
    return False


DIRECT_PROMPT = """Answer the question using only the facts below. Return only the short answer, without explanation.

Facts:
{facts}

Question: {question}
Answer:"""

COT = "cot"                                     # third element of a system spec: the LLM reasons step by step, then answers
COT_PROMPT = """Answer the question using only the facts below. Think step by step: find the facts that answer each part
of the question and connect them, then give the final answer.

Facts:
{facts}

Question: {question}

Write your reasoning, then end with one last line of the form "Answer: <short answer>"."""
COT_ANSWER = re.compile(r"^\W*answer\W*:\s*(.+)$", re.I | re.M)

JUDGE_PROMPT = """You grade one answer to a question against the gold answer. Be strict and consistent.

Question: {question}
Gold answer: {gold}
Model answer: {answer}

YES if the model answer gives the same entity, date, number or place as the gold answer, even with extra words around
it, a fuller or shorter name, or an abbreviation:
  "IOC president Thomas Bach" for "Thomas Bach"; "UN (United Nations)" for "United Nations";
  "Marie Curie won it" for "Marie Curie"; "the 1990s" for "1990s"
NO if it gives anything else, even when it shares words with the gold answer or is related to it:
  a different thing with a similar name ("The Boston Globe" for "Boston"); a part or a larger whole ("Oregon" for
  "Portland"; "Portland" for "Oregon"); something about the answer instead of it ("Einstein's theory" for "Einstein");
  several candidates ("Paris or Lyon" for "Paris"); a different date or number; no answer ("not stated").
Reply with only yes or no."""


def metrics(judge_client, q, answer):
    """exact; lenient; judge = exact, or else the judge model says it is the same thing (its raw reply is kept)."""
    g = golds(q)
    result = {"exact": exact(answer, g), "lenient": lenient(answer, g), "judge_reply": None, "judge_cost": 0.0}
    result["judge"] = result["exact"]
    if answer and not result["exact"]:
        response = judge_client.chat.completions.create(model=JUDGE_MODEL, temperature=0, max_tokens=3, messages=[{
            "role": "user", "content": JUDGE_PROMPT.format(question=q["question"], gold=" / ".join(g), answer=answer)}])
        result["judge_reply"] = (response.choices[0].message.content or "").strip()
        result["judge"] = result["judge_reply"].lower().startswith("yes")
        result["judge_cost"] = cost(response.usage.prompt_tokens, response.usage.completion_tokens, JUDGE_MODEL)
    return result


# ── One answer ──────────────────────────────────────────────────────────────────────────────────────────────────────
def run_one(q, index, k, system, client, model, judge_client, ona_path=None):
    """Returns (answer record, trace or None). ona_path: None = direct answer, COT = step-by-step reasoning then the
    answer, else NARS with that ONA binary."""
    read = paragraphs_at(q, k)
    kind = "direct" if ona_path is None else "cot" if ona_path == COT else "nars"
    record = {"index": index, "id": q["id"], "source": q.get("source_split"), "question": q["question"], "gold": q["answer"],
              "golds": golds(q), "k": k, "system": system, "kind": kind, "model": model,
              "paragraphs": [{"title": p["title"], "supporting": p["is_supporting"], "rank": p["rank"]} for p in read],
              "facts_read": sum(len(p["facts"]) for p in read), "error": None, "cost": 0.0,
              "started": datetime.now(timezone.utc).isoformat()}
    trace, start = None, time.time()
    try:
        if kind in ("direct", "cot"):
            template = COT_PROMPT if kind == "cot" else DIRECT_PROMPT
            prompt = template.format(facts="\n".join(f"- {f}" for p in facts(q, k) for f in p), question=q["question"])
            response = client.chat.completions.create(model=model, temperature=0, max_tokens=768 if kind == "cot" else 64,
                                                      messages=[{"role": "user", "content": prompt}])
            raw = response.choices[0].message.content or ""
            if kind == "cot":                   # the last "Answer: ..." line; none (e.g. cut off) = no answer
                found = COT_ANSWER.findall(raw)
                answer = found[-1].strip().strip("*").strip() if found else None
            else:
                answer = raw.strip().split("\n")[0]
            record.update(served_model=response.model, raw_reply=raw, answer=answer or None,
                          finish_reason=response.choices[0].finish_reason,
                          prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                          input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens,
                          cost=cost(response.usage.prompt_tokens, response.usage.completion_tokens, model))
        else:
            ona = ONA.OnaReasoner(client, model=model, **ONA_OPTIONS)
            answer = ona.solve(facts(q, k), q["question"], ona_path)
            record.update(answer=answer, has_rules=bool(ona.last_rules), rules=ona.last_rules,
                          votes=[s["answer"] for s in ona.samples], auto_repairs=dict(ona.auto_repairs),
                          answer_type=ona.answer_type, ona_top=ona.ona_top, triples=len(ona.last_triples),
                          llm_calls=len(ona.request_log), input_tokens=ona.usage["input_tokens"],
                          output_tokens=ona.usage["output_tokens"],
                          cost=cost(ona.usage["input_tokens"], ona.usage["output_tokens"], model))
            trace = {"index": index, "id": q["id"], "k": k, "system": system, "answer": answer, "plan": ona.plan_text,
                     "triples": ona.last_triples, "aliases": ona.aliases, "relation_groups": ona.relation_groups,
                     "missing_terms": ona.missing_terms, "rule_attempts": ona.rule_attempts, "rules": ona.last_rules,
                     "samples": [{key: s.get(key) for key in ("answer", "rules", "auto_repairs")} for s in ona.samples],
                     "candidates": [ona.names.get(c, c) for c in ona.candidates], "narsese": ona.last_narsese,
                     "ona_output": ona.last_output, "llm_calls": ona.request_log}
        record.update(metrics(judge_client, q, record["answer"]))
    except Exception as error:                  # saved, and retried on the next pass
        record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
    record["seconds"] = round(time.time() - start, 1)
    return record, trace


# ── Storage and the run loop ────────────────────────────────────────────────────────────────────────────────────────
def slug(system):
    return re.sub(r"[^A-Za-z0-9]+", "_", system).strip("_")


def answers_path(results, system, k):
    return f"{results}/answers/{slug(system)}/k{k:02d}.jsonl"


def load_answers(results, system, k):
    path = answers_path(results, system, k)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["index"]: r for r in map(json.loads, f) if not r.get("error")}


def guard(results, question_hash):
    """One results folder per question set: stop instead of mixing answers to different questions."""
    os.makedirs(results, exist_ok=True)
    path = f"{results}/questions.sha256"
    if os.path.exists(path):
        saved = open(path).read().strip()
        assert saved == question_hash, (f"{results} holds answers to another question set ({saved[:8]}, now "
                                        f"{question_hash[:8]}): move it away before running")
    else:
        open(path, "w").write(question_hash)


def run_all(questions, question_hash, results, systems, k_values, workers=16, passes=3):
    """systems: {name: (client, model, ona_path, or None for a direct answer, or COT)}. Runs every missing (system, k, question), saving each answer
    as it finishes; up to `passes` passes so failed calls are retried.

    A progress bar shows answers done, time left, money spent this session (systems + judge), errors, and the running
    judged accuracy of each system (all its saved answers at these k, earlier sessions included). Errors and finished
    (system, k) pairs are printed above the bar."""
    from tqdm.auto import tqdm

    guard(results, question_hash)
    judge = deepinfra()
    spent = {"systems": 0.0, "judge": 0.0}
    n, errors = len(questions), 0
    for attempt in range(passes):
        saved = {(s, k): load_answers(results, s, k) for s in systems for k in k_values}
        jobs = [(i, s, k) for s in systems for k in k_values for i in range(n) if i not in saved[s, k]]
        total = n * len(systems) * len(k_values)
        print(f"pass {attempt + 1}: {len(jobs)} answers to run ({total - len(jobs)} of {total} already saved: "
              f"{n} questions × {len(systems)} systems × k = {', '.join(map(str, k_values))})")
        if not jobs:
            break
        right = {key: sum(r["judge"] for r in answers.values()) for key, answers in saved.items()}
        count = {key: len(answers) for key, answers in saved.items()}
        left = {key: n - count[key] for key in saved}
        with ThreadPoolExecutor(workers) as pool, tqdm(total=len(jobs), desc=f"pass {attempt + 1}", unit="answer",
                                                      dynamic_ncols=True) as bar:
            futures = {}
            for i, s, k in jobs:
                client, model, ona_path = systems[s]
                futures[pool.submit(run_one, questions[i], i, k, s, client, model, judge, ona_path=ona_path)] = (s, k)
            for future in as_completed(futures):
                system, k = futures[future]
                record, trace = future.result()
                for folder, item in (("answers", record), ("traces", trace)):
                    if item is not None:
                        path = f"{results}/{folder}/{slug(system)}/k{k:02d}.jsonl"
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "a") as f:
                            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
                spent["systems"] += record["cost"]
                spent["judge"] += record.get("judge_cost", 0.0)
                key = (system, k)
                left[key] -= 1
                if record["error"]:
                    errors += 1
                    bar.write(f"   ! {system} k={k} question {record['index']}: {record['error'][:90]} (retried next pass)")
                else:
                    right[key] += record["judge"]
                    count[key] += 1
                if left[key] == 0:
                    bar.write(f"✔ {system:<22} k={k:<2} finished: {100 * right[key] / max(count[key], 1):5.1f}% judged "
                              f"correct ({right[key]}/{count[key]})")
                accuracy = {}
                for s in systems:
                    r, c = sum(right[s, kk] for kk in k_values), sum(count[s, kk] for kk in k_values)
                    if c:
                        accuracy[s[:24]] = f"{100 * r / c:.0f}% of {c}"
                bar.set_postfix({"$": f"{sum(spent.values()):.2f}", "errors": errors, **accuracy}, refresh=False)
                bar.update()
    missing = sum(1 for s in systems for k in k_values for i in range(n) if i not in load_answers(results, s, k))
    print(f"done: ${sum(spent.values()):.2f} this session (systems ${spent['systems']:.2f}, judge ${spent['judge']:.2f})"
          + (f" · {missing} answers still failing: run this cell again" if missing else " · every answer saved"))


def compare_k(results, names, k, n_questions=None):
    """One k, for the systems in `names` (saved answers, earlier sessions included): each system's judged and exact
    accuracy, no-answer rate, cost and time, then every system against the first one question by question."""
    from prettytable import PrettyTable

    answers = {s: {i: r for i, r in load_answers(results, s, k).items() if n_questions is None or i < n_questions}
               for s in dict.fromkeys(names)}
    table = PrettyTable(["System", "judge %", "exact %", "no answer", "answers", "cost $", "s / answer"])
    table.align["System"] = "l"
    for s, rows in answers.items():
        r = list(rows.values())
        if not r:                               # nothing saved for this system at this k
            continue
        table.add_row([s, f"{100 * sum(x['judge'] for x in r) / len(r):.1f}", f"{100 * sum(x['exact'] for x in r) / len(r):.1f}",
                       f"{100 * sum(x['answer'] is None for x in r) / len(r):.0f}%", len(r),
                       f"{sum(x['cost'] + x.get('judge_cost', 0.0) for x in r):.3f}",
                       f"{sum(x['seconds'] for x in r) / len(r):.1f}"])
    present = [s for s in answers if answers[s]]
    if not present:
        print(f"k = {k}: no answers saved yet")
        return
    print(f"k = {k}\n{table}")
    first, *others = present
    for s in others:
        both = set(answers[first]) & set(answers[s])
        if not both:
            continue
        a = {i: answers[first][i]["judge"] for i in both}
        b = {i: answers[s][i]["judge"] for i in both}
        print(f"{s} vs {first} on {len(both)} questions: both right {sum(a[i] and b[i] for i in both)} · "
              f"only {s} {sum(b[i] and not a[i] for i in both)} · only {first} {sum(a[i] and not b[i] for i in both)} · "
              f"neither {sum(not a[i] and not b[i] for i in both)}")


# ── Report and chart ────────────────────────────────────────────────────────────────────────────────────────────────
def collect(results):
    """Every saved answer in the results folder, as rows (one per system, k and question)."""
    rows = []
    for path in sorted(glob_answers(results)):
        with open(path) as f:
            rows += [r for r in map(json.loads, f) if not r.get("error")]
    return rows


def glob_answers(results):
    base = f"{results}/answers"
    if not os.path.isdir(base):
        return []
    return [f"{base}/{d}/{f}" for d in os.listdir(base) for f in os.listdir(f"{base}/{d}") if f.endswith(".jsonl")]


def with_fallback(df):
    """Adds "<NARS system> + fallback" rows built from saved answers: the NARS answer when NARS derived one, else the
    8B reference's direct answer to the same question at the same k. Fallback answers are not traceable to a derivation,
    so these rows are reported next to pure NARS, never instead of it."""
    import pandas as pd

    if REFERENCE not in set(df.system):
        return df
    ref = df[df.system == REFERENCE].set_index(["index", "k"])[["answer", "exact", "lenient", "judge", "cost"]]
    extra = []
    for s in sorted(set(df[df.kind == "nars"].system)):
        part = df[df.system == s].merge(ref, left_on=["index", "k"], right_index=True, suffixes=("", "_ref"))
        none = part.answer.isna()
        for column in ("answer", "exact", "lenient", "judge"):
            part.loc[none, column] = part.loc[none, f"{column}_ref"]
        part.loc[none, "cost"] += part.loc[none, "cost_ref"]
        name = f"{s} + fallback"
        SIZES.setdefault(name, SIZES.get(s, "?"))
        extra.append(part.drop(columns=[c for c in part if c.endswith("_ref")]).assign(system=name, kind="fallback"))
    return pd.concat([df, *extra], ignore_index=True) if extra else df


def report(results, dataset, n_questions=None, fallback=True):
    """Tables (judge and exact per k), robustness vs the 8B reference when it is present, NARS failure rates, cost, and
    a chart of accuracy vs k. Reads whatever has been saved, so it works mid-run; writes report.txt and accuracy_vs_k.png.
    fallback: also report each NARS system with the 8B's direct answer where NARS derived none (see with_fallback)."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from prettytable import PrettyTable

    rows = collect(results)
    if not rows:
        print("no answers saved yet")
        return
    df = pd.DataFrame(rows)
    if n_questions:
        df = df[df["index"] < n_questions]
    for metric in ("exact", "lenient", "judge"):
        df[metric] = df[metric].astype(int)
    if fallback:
        df = with_fallback(df)
    k_values = sorted(df.k.unique())
    order = [s for s in [*BASELINES, REFERENCE, "NARS + 8B + LoRA"] if s in set(df.system)]
    order += sorted(set(df.system) - set(order))
    lines = [f"{dataset.upper()} · {df['index'].nunique()} questions · {len(df)} answers · k = {', '.join(map(str, k_values))}", ""]

    for metric in ("judge", "exact"):
        table = PrettyTable(["System", "Size", *[f"k={k}" for k in k_values], f"Δ 0→{k_values[-1]}", "n"])
        table.align["System"] = "l"
        for s in order:
            part = df[df.system == s]
            acc = [part[part.k == k][metric].mean() * 100 if (part.k == k).any() else float("nan") for k in k_values]
            table.add_row([s, SIZES.get(s, "?"), *[f"{a:.1f}" for a in acc], f"{acc[-1] - acc[0]:+.1f}",
                           part["index"].nunique()])
        lines += [f"ACCURACY (%) · {metric}", str(table), ""]

    if REFERENCE in set(df.system) and len(k_values) > 1:
        first, last = k_values[0], k_values[-1]
        for metric in ("judge", "exact"):
            table = PrettyTable(["System", f"loses less than {REFERENCE} ({metric}, k {first}→{last})", "95% CI", "n"])
            table.align["System"] = "l"
            for s in order:
                if s == REFERENCE:
                    continue
                w = df[df.system.isin([s, REFERENCE]) & df.k.isin([first, last])].pivot_table(
                    index="index", columns=["system", "k"], values=metric).dropna()
                if w.empty or s not in w.columns.get_level_values(0):
                    continue
                diff = ((w[s][last] - w[s][first]) - (w[REFERENCE][last] - w[REFERENCE][first])).to_numpy()
                rng = random.Random(0)
                boots = sorted(sum(rng.choice(diff) for _ in diff) / len(diff) for _ in range(2000))
                table.add_row([s, f"{100 * diff.mean():+.1f} points", f"[{100 * boots[50]:+.1f}, {100 * boots[1949]:+.1f}]",
                               len(diff)])
            lines += [f"ROBUSTNESS · {metric} (paired per question; positive = loses less accuracy than the 8B)", str(table), ""]

    nars = df[df.kind.isin(["nars", "cot"])]    # CoT gives no answer when its reasoning never reaches an "Answer:" line
    if not nars.empty:
        table = PrettyTable(["NARS / CoT system", *[f"no answer k={k}" for k in k_values]])
        for s in sorted(set(nars.system)):
            part = nars[nars.system == s]
            table.add_row([s, *[f"{part[part.k == k].answer.isna().mean() * 100:.0f}%" if (part.k == k).any() else "-"
                                for k in k_values]])
        lines += ["NARS / CoT: share of questions with no answer", str(table), ""]

    if "source" in df and df.source.nunique() > 1:
        lines.append("BY SOURCE SPLIT (judge accuracy per k)")
        for source, part in df.groupby("source"):
            lines.append(f"   {source} (n={part['index'].nunique()}): " + " · ".join(
                f"{s} " + "/".join(f"{part[(part.system == s) & (part.k == k)].judge.mean() * 100:.0f}" for k in k_values)
                for s in order))
        lines.append("")
    lines.append(f"COST: ${df.cost.sum() + df.judge_cost.sum():.2f} on DeepInfra "
                 f"(systems ${df.cost.sum():.2f}, judge ${df.judge_cost.sum():.2f})")
    text = "\n".join(lines)
    print(text)
    with open(f"{results}/report.txt", "w") as f:
        f.write(text + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    palette = plt.get_cmap("tab10")
    for ax, metric in zip(axes, ("judge", "exact")):
        for i, s in enumerate(order):
            part = df[df.system == s]
            ks = [k for k in k_values if (part.k == k).any()]
            acc = [100 * part[part.k == k][metric].mean() for k in ks]
            n = [len(part[part.k == k]) for k in ks]
            err = [196 * math.sqrt(a / 100 * (1 - a / 100) / m) for a, m in zip(acc, n)]
            strong = s in NARS_SYSTEMS or s in (REFERENCE, f"{REFERENCE} + CoT")
            ax.errorbar(ks, acc, yerr=err if strong else None, marker="o", capsize=3, color=palette(i % 10),
                        linewidth=2.6 if s in NARS_SYSTEMS else 1.3, linestyle="-" if strong else "--",
                        markersize=6 if strong else 4, label=f"{s} ({SIZES.get(s, '?')})")
        ax.set_title(f"{dataset} · {'judged accuracy' if metric == 'judge' else 'exact match'}", fontsize=12)
        ax.set_xlabel("distractor paragraphs added (k)")
        ax.set_xticks(k_values)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy (%)  ·  95% CI on NARS and the 8B")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(f"{results}/accuracy_vs_k.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"saved {results}/report.txt and {results}/accuracy_vs_k.png")
