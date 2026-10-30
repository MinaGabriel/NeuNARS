"""Ablation: the same triples and rules a NARS run used, reasoned over by Prolog (SWI-Prolog) instead of ONA.

For every saved NARS trace (one question at one k) this compiles the triples and each voted rule set into a Prolog
program, runs it with `swipl`, and picks the answer the way NARS does, as far as plain logic allows:
- triples: exactly the ones ONA received (after alias merging, the general relation forms and the inverses);
- rule sets: every sample NARS voted over; each rule set's answer is the binding derived most often by its rules
  (ties: the first rule, then the first binding); the final answer is the one most rule sets agree on, ties to the
  earliest, as in ONA.solve;
- no truth values, no answer-type preference, no partial matching: Prolog proves a binding or it does not.
"""
import json
import os
import re
import subprocess
import tempfile
from collections import Counter

import ONA

SWIPL = "swipl"


def is_var(term):
    return isinstance(term, str) and term.startswith("?")


def term(value):
    """A rule term as Prolog: variables become V_<NAME>, constants are already ONA atoms ([a-z0-9_], safe as atoms)."""
    if is_var(value):
        return "V_" + re.sub(r"[^A-Za-z0-9_]", "_", value[1:])
    return ONA.atom(value)


def program(triples, rule_set):
    """Prolog text: one triple/3 fact per triple, one r<i>/1 clause per rule (head = its answer variable)."""
    lines = [":- dynamic triple/3.", ":- style_check(-singleton)."]
    facts = sorted({(ONA.atom(t["subject"]), ONA.atom(t["relation"]), ONA.atom(t["object"])) for t in triples})
    lines += [f"triple({s}, {r}, {o})." for s, r, o in facts]
    for i, rule in enumerate(rule_set, 1):
        body = ", ".join(f"triple({term(c[0])}, {term(c[1])}, {term(c[2])})" for c in rule["conditions"])
        lines.append(f"r{i}({term(rule['answer'])}) :- {body}.")
    n = len(rule_set)
    lines.append(f"main :- forall(between(1, {n}, I), (atom_concat(r, I, P), G =.. [P, X], "
                 "(catch(findall(X, G, L), _, L = []) -> true ; L = []), "
                 "forall(member(A, L), (atom(A) -> format('~w\\t~w~n', [I, A]) ; true)))), halt.")
    return "\n".join(lines) + "\n"


def run_rules(triples, rule_set, timeout=60):
    """[(rule number, answer atom)] in derivation order, or None if Prolog failed."""
    if not rule_set:
        return []
    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as f:
        f.write(program(triples, rule_set))
        path = f.name
    try:
        out = subprocess.run([SWIPL, "-q", "-g", "main", "-t", "halt(1)", path], capture_output=True, text=True,
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(path)
    found = []
    for line in out.stdout.splitlines():
        rule, _, answer = line.partition("\t")
        if rule.isdigit() and answer:
            found.append((int(rule), answer))
    return found


def pick(found):
    """One rule set's answer: the binding derived most often; ties go to the earliest rule, then the earliest binding."""
    if not found:
        return None
    counts = Counter(answer for _, answer in found)
    first = {}
    for position, (rule, answer) in enumerate(found):
        first.setdefault(answer, (rule, position))
    return min(counts, key=lambda a: (-counts[a], first[a]))


def solve(trace):
    """The Prolog answer (display name) for one NARS trace, plus per-vote details."""
    names = {}
    for t in trace["triples"]:
        for key in ("subject", "object"):
            names.setdefault(ONA.atom(t[key]), t[key])
    rule_sets = [s["rules"] for s in trace.get("samples") or [] if s.get("rules")] or \
                ([trace["rules"]] if trace.get("rules") else [])
    votes, errors = [], 0
    for rule_set in rule_sets:
        found = run_rules(trace["triples"], rule_set)
        if found is None:
            errors += 1
            votes.append(None)
        else:
            choice = pick(found)
            votes.append(names.get(choice, choice) if choice else None)
    answered = [v for v in votes if v]
    if not answered:
        return None, votes, errors
    counts = Counter(answered)
    answer = min(counts, key=lambda a: (-counts[a], answered.index(a)))
    return answer, votes, errors


def load_traces(results, system, k):
    path = f"{results}/traces/{re.sub(r'[^A-Za-z0-9]+', '_', system).strip('_')}/k{k:02d}.jsonl"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {t["index"]: t for t in map(json.loads, f)}         # the last trace of a question wins (a retry)
