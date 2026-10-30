"""DeepInfra notebook helper: fact list + question -> runnable Narsese.

    ona = OnaReasoner(client, plan=True, votes=3, merge_relations=True)
    answer = ona.solve([paragraph_1_facts, paragraph_2_facts], question, "/path/to/ONA/NAR")   # or a flat fact list
    print(answer, ona.last_rules)
    narsese = ona.run(facts, question)   # just the Narsese for one rule set, without running ONA

Per question: each paragraph's facts -> triples together (TRIPLES prompt, so "this feature" resolves from
the neighbouring facts; a name, title or number dropped from every triple is sent back once with
TRIPLE_REPAIR) -> one name per entity (spelling rules plus the ALIASES prompt) -> [merge_relations: triples
also stated under their relation's general form, "youngest_son_of" -> "child_of"] -> inverse triples for relation pairs already in use
-> [plan: the question as hops, PLAN prompt] -> rules (RULES prompt over the full triple list, repaired with
REPAIR feedback when no rule matches) -> Narsese. solve() runs ONA too, and with votes > 1 writes several rule
sets and returns the answer most of them agree on.
No model weights, GPU, file outputs, or fact extraction are used here.
"""
import difflib
import hashlib
import json
import re
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

TRIPLES = """Convert the facts below into triples (subject, relation, object). The facts come from one paragraph.
- Give 1 to 3 triples per fact.
- Every name, title, date and number in a fact must appear in a triple, copied exactly as written.
- If a fact lists multiple people or items together (e.g. "A and B"), make a separate triple for each one.
- Keep subjects and objects as short, atomic entity names (e.g. extract "Ilse Marren" or "Norland" as its own triple rather than hiding it inside a long phrase like "early films of director Ilse Marren").
- When a fact uses a possessive phrase (e.g. "Ilse Marren's studio", "Norland Rail's QuickDock"), also add a triple linking the person/place to that thing (e.g. {{"subject": "Ilse Marren's studio", "relation": "of", "object": "Ilse Marren"}}).
- Never leave pronouns ("his father", "her", "this feature", "the company") in a triple; always replace them with the actual name from the facts.
- If a fact gives a nationality with a role (e.g. "Ilse Marren is a German film director"), also add a separate nationality triple (e.g. {{"subject": "Ilse Marren", "relation": "nationality", "object": "German"}}).
- Relation: short snake_case phrase, read as "subject relation object" (e.g. wife_of, born_in, directed, played).
- Reply with JSON only: a list of triples.

Facts:
1. Norland Rail laptops have a charging port marked with a lightning icon.
2. Norland Rail calls this feature QuickDock.
3. Ilse Marren directed The Glass Harbor in 1987.
Answer: [{{"subject": "Norland Rail laptops", "relation": "have", "object": "charging port"}}, {{"subject": "charging port", "relation": "marked_with", "object": "lightning icon"}}, {{"subject": "Norland Rail", "relation": "calls_feature", "object": "QuickDock"}}, {{"subject": "QuickDock", "relation": "name_for", "object": "charging port"}}, {{"subject": "Ilse Marren", "relation": "directed", "object": "The Glass Harbor"}}, {{"subject": "The Glass Harbor", "relation": "released_in", "object": "1987"}}]

Facts:
{facts}
Answer:"""

TRIPLE_REPAIR = """These names and values from the facts are missing from every triple: {missing}.
Reply with the full corrected list of triples, so that each of them appears in a triple. JSON only."""

ALIASES = """Group the names below that refer to the same real-world entity.
- Group only names that clearly mean the same thing, e.g. "Port Halden, Norland" and "Port Halden"; "Kestrel Valley Authority" and "KVA"; "Norland Rail" and "Norland Railway Company".
- Never group related but different things: two different family members (such as a father and son), a person and their standalone role/title, a band and its album, a building and its rooms, a city and its country/university, an event and an award, or two different sports franchises/teams.
- Copy names exactly as listed. Leave out names that have no alias.
- Reply with JSON only: a list of groups, each a list of names. Reply [] if there are none.

Names:
{names}
"""

RULES = """Write rules that answer the question from the triples below.
- Each triple reads "subject | relation | object". Use ONLY relations and entities from the triples, spelled exactly as listed, and keep each relation's direction.
- Follow the question hop by hop: one condition per hop, chained through shared variables (?A, ?B, ?C, ?D). Never use a real entity name to connect two conditions—always use a variable (?A, ?B, ?C) for any entity that appears in more than one condition!
- At most 5 conditions: use only the conditions needed to get from the known entity in the question to the answer, and leave out descriptive details of the question.
- Pay strict attention to subject vs. object order and which variable the question actually asks for (e.g. if the question asks "what character did X play", the answer variable must be the character, not the actor X).
- Never put a real entity name in the "answer" field, and never embed a variable inside a string (like "wife of ?A"). Always use standalone variables (?A, ?B, ?C, ?D).
- "answer_type" is what the question asks for: "date" (a date, year or time), "number" (a count, amount or measurement), "place", "person", or "other". The answer variable must hold that type.
- Give 1 to 3 alternative rules if the needed relation could be phrased in different ways.
- Reply with JSON only.

Example 1 (2-hop)
Question: When was the director of The Glass Harbor born?
Triples:
Ilse Marren | directed | The Glass Harbor
Ilse Marren | born_in | 1954
Answer: {{"answer_type": "date", "rules": [{{"conditions": [["?A", "directed", "The Glass Harbor"], ["?A", "born_in", "?B"]], "answer": "?B"}}]}}

Example 2 (3-hop)
Question: Which river flows through the birthplace of the director of The Glass Harbor?
Triples:
Ilse Marren | directed | The Glass Harbor
Ilse Marren | born_in | Port Halden
Lenne River | flows_through | Port Halden
Answer: {{"answer_type": "place", "rules": [{{"conditions": [["?A", "directed", "The Glass Harbor"], ["?A", "born_in", "?B"], ["?C", "flows_through", "?B"]], "answer": "?C"}}]}}

Question: {question}
{plan}Triples:
{triples}
Answer:"""

PLAN = """Break the question into hops: the chain of simple lookups that leads from what the question names to the
answer. One short line per hop, in order; refer to an earlier hop's result as #1, #2. Do not answer the question.

Question: Which river flows through the birthplace of the director of The Glass Harbor?
Plan:
1. Who directed The Glass Harbor?
2. Where was #1 born?
3. Which river flows through #2?

Question: {question}
Plan:"""

REPAIR = """Those rules do not work yet:
{problems}
Rewrite the rules using only the triples listed above. Make sure every hop shares a variable (?A, ?B, ?C) with the next hop. Reply with JSON only, in the same format."""

# Relation pairs that state the same fact in opposite directions; "x_by" also pairs with "x".
INVERSES = {"parent_of": "child_of", "child_of": "parent_of", "contains": "located_in",
            "located_in": "contains", "has_part": "part_of", "part_of": "has_part",
            "has_member": "member_of", "member_of": "has_member", "owns": "owned_by",
            "owned_by": "owns", "owner_of": "owned_by", "father_of": "child_of",
            "mother_of": "child_of", "son_of": "parent_of", "daughter_of": "parent_of",
            "center_of": "has_center", "has_center": "center_of", "largest_town_in": "contains",
            # "A brother_of B" says B is A's sibling, not B's brother (B may be a sister).
            "brother_of": "sibling_of", "sister_of": "sibling_of", "half_brother_of": "half_sibling_of",
            "half_sister_of": "half_sibling_of"}
# Relations that read the same in both directions. Gendered or ordered ones (brother_of, older_sister_of)
# are not symmetric: "A older_brother_of B" does not make B an older brother of A.
SYMMETRIC = {"spouse_of", "married_to", "married", "sibling_of", "half_sibling_of", "identical_twin_of", "twin_of",
             "cousin_of", "cousin_to", "made_agreement_with", "merged_with", "also_known_as", "also_called"}


def atom(name):
    """Convert an entity/relation to a compact ONA-safe identifier."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text or not text[0].isalpha():
        text = "e_" + text
    if len(text) > 29:
        # rstrip: a prefix ending in "_" would give "__", which atom() itself would collapse on a second pass,
        # so a rule's relation and its fact's relation would stop matching.
        text = text[:20].rstrip("_") + "_" + hashlib.sha256(text.encode()).hexdigest()[:8]
    return text


def raw_words(name):
    """Lowercase alphanumeric tokens of a name WITHOUT calling atom() (which truncates at 20 chars)."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\bst\.?\b", "saint", text)
    return re.findall(r"[a-z0-9]+", text)


def relation_name(relation):
    """A relation as readable snake_case, never truncated or hashed: what the RULES prompt shows the LLM.
    atom(relation_name(r)) == atom(r), so the LLM's copy still matches the facts."""
    text = unicodedata.normalize("NFKD", str(relation)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def readable(name):
    """Lowercase words of a name, for spelling comparisons: 'The Glass-Harbor (1987)' -> 'the glass harbor 1987'."""
    return " ".join(raw_words(name))


def surface_key(name):
    """Spelling-insensitive key: drops parentheticals, a leading "the", and ", Region" suffixes; normalizes St/Saint."""
    text = re.sub(r"\([^)]*\)", " ", str(name))
    text = re.sub(r",\s*(?!(?:jr|sr|junior|senior|ii|iii|iv|vi|vii|viii)\b)[A-Za-z].*$", "", text, flags=re.I)  # "Eldon, Missouri" -> "Eldon"; "4,567" and ", Jr." are kept
    text = re.sub(r"^\s*the\s+", "", text, flags=re.I)
    text = re.sub(r"\bst\.?\s+", "saint ", text, flags=re.I)
    return atom(text)


def qualifier(name):
    """The part after the first comma: "Kestrel County, Norland" -> "norland"; "" if none ("4,567" has none)."""
    match = re.search(r",\s*([A-Za-z].*)$", str(name))
    return atom(match.group(1)) if match else ""


ALIAS_STOPWORDS = {"the", "of", "and", "a", "an", "in", "on", "for", "to", "at", "by", "mainland", "de", "von", "van", "del", "da"}
# Words that distinguish different places/entities and must never be silently dropped in alias merging:
# directions, kinds of place, and family roles.
DISTINCTIVE_MODIFIERS = {
    "west", "western", "east", "eastern", "north", "northern", "south", "southern",
    "downs", "hills", "mountains", "range", "valley", "river", "lake", "bay", "island",
    "county", "district", "parish", "oblast", "province", "region", "state",
    "father", "mother", "son", "daughter", "wife", "husband", "brother", "sister", "family", "reign", "journey",
}
JR_SR = {"jr", "sr", "junior", "senior", "elder", "younger"}
ROMAN_SUFFIXES = {"ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii"}
def plausible_alias(a, b):
    """Could two names be the same entity by their wording? Uses full untruncated words (fixes the 20-char hash bug).
    Names differing in an ordinal ("1st" / "2nd"), a roman numeral ("Edward I" / "Edward II") or Jr/Sr never merge."""
    if atom(a) == atom(b) or surface_key(a) == surface_key(b):
        return True
    initials = lambda n: "".join(w[0] for w in raw_words(n) if w not in ALIAS_STOPWORDS)
    if any(len(raw_words(x)) == 1 and len(x) >= 2 and raw_words(x)[0] == initials(y) for x, y in ((a, b), (b, a))):
        return True  # acronym: "FDA" ~ "Food and Drug Administration", checked before "and" rules out lists
    ords_a = re.findall(r"\b\d+(?:st|nd|rd|th)\b", str(a).lower())
    ords_b = re.findall(r"\b\d+(?:st|nd|rd|th)\b", str(b).lower())
    if ords_a != ords_b:
        return False
    if re.search(r"\d", a + b) or re.search(r"\band\b|&", f"{a} {b}", re.I):
        return False  # dates, numbers and lists of things merge only when spelled the same
    # Do not merge an entity X with a possessive phrase "X's ..." (e.g. "Ilse Marren" vs "Ilse Marren's debut film")
    if ("'s " in a.lower() or "’s " in a.lower()) != ("'s " in b.lower() or "’s " in b.lower()):
        return False
    wa_full = raw_words(a)
    wb_full = raw_words(b)
    roman = lambda text, words: (set(words) & ROMAN_SUFFIXES) | set(re.findall(r"(?<=[a-z]\s)i\b", text.lower()))
    if roman(a, wa_full) != roman(b, wb_full):
        return False
    if set(wa_full) & JR_SR != set(wb_full) & JR_SR:
        return False
    # Never merge if one name has a distinctive modifier (like "West Kestrel" or "Kestrel Hills") that the other lacks
    mods_a = set(wa_full) & DISTINCTIVE_MODIFIERS
    mods_b = set(wb_full) & DISTINCTIVE_MODIFIERS
    if mods_a != mods_b:
        words_a = [w for w in wa_full if w not in ALIAS_STOPWORDS]
        words_b = [w for w in wb_full if w not in ALIAS_STOPWORDS]
        short_w, long_w = sorted((words_a, words_b), key=len)
        if not (len(short_w) == 1 and len(short_w[0]) >= 2 and short_w[0] == "".join(w[0] for w in long_w)):
            return False
    words = lambda wl: [w for w in wl if w not in ALIAS_STOPWORDS]
    short, long_ = sorted((words(wa_full), words(wb_full)), key=len)
    if not short:
        return False
    # Every word of the shorter name starts a word of the longer one ("fed" -> "federation").
    return all(any(w.startswith(s) or s.startswith(w) for w in long_) for s in short if len(s) >= 3) \
        and any(len(s) >= 3 for s in short)


def inverse(relation):
    if relation in SYMMETRIC:
        return relation
    if relation in INVERSES:
        return INVERSES[relation]
    return relation[:-3] if relation.endswith("_by") else relation + "_by"


MONTHS = r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
WH_WORDS = {"is", "was", "are", "were", "does", "did", "do", "has", "had", "the", "a", "an", "of", "kind", "type", "sort", "name"}
ROLE_WORDS = {"president", "vice-president", "vice president", "chairman", "director", "governor", "mayor", "chancellor", "minister", "senator", "member", "members", "emperor", "king", "queen", "prince", "princess"}

# answer_type values the RULES prompt may return, with how the retry message names them; "other" is unchecked.
ANSWER_TYPES = {"date": "a date or year", "number": "a number", "place": "a place", "person": "a person"}


def question_type(question):
    """The answer type the question's wording fixes ("Where ..." -> place), or None; overrides the LLM's guess."""
    q = question.lower().strip()
    if re.search(r"^(where|in which (city|country|state|place|town|province))\b|\bplace of (birth|death|burial)\b", q):
        return "place"
    if re.search(r"^(when|what year|in what year|in which year)\b|\b(date|year) of (birth|death|release)\b", q):
        return "date"
    if re.search(r"^how (many|much|long|old|tall|far|often)\b", q):
        return "number"
    if re.search(r"^who\b", q):
        return "person"
    return None


NUM_WORDS = (
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|"
    r"single|double|triple|once|twice)\b"
)


def answer_fit(question, name, answer_type=None):
    """How well a candidate fits the question: (matches answer_type, not already named in the question, noun/unit match)."""
    q, n = question.lower(), name.lower()
    has_digit = bool(re.search(r"\d", n))
    has_non_ordinal_digit = bool(re.search(r"\b\d+(?!(?:st|nd|rd|th)\b)", n))
    has_num_word = bool(re.search(NUM_WORDS, n))
    date_like = (
        (has_digit and bool(re.search(rf"\b(\d{{3,4}}|{MONTHS}\w*|bc|ad|bce|ce|\d{{1,2}}\s*[ap]\.?m\.?|\d{{1,2}}:\d{{2}})\b", n)))
        or bool(re.search(r"\b(century|centuries|decade|season|era|millennium)\b", n))
    )
    if answer_type == "date":
        fits = date_like
    elif answer_type == "number":
        fits = has_digit or has_num_word
    elif answer_type == "person":
        fits = not has_non_ordinal_digit and n.strip() not in ROLE_WORDS and not re.match(r"(current )?(vice[- ])?president of\b", n)
    elif answer_type == "place":
        fits = not has_non_ordinal_digit
    else:
        fits = True
    # Deprioritize candidates whose full name is already stated in the question (unless it is an "X or Y" choice question).
    not_in_q = not (len(n) >= 4 and " or " not in q and re.search(rf"\b{re.escape(n)}\b", q) is not None)
    # "Which county ..." / "What Irish county ..." / "What kind of producer ..." prefers the matching head noun.
    nouns = [w.lower() for w in re.findall(
        r"\b(?:[Ww]hich|[Ww]hat)\s+(?:(?:kind|type|sort|name)\s+of\s+)?(?:[A-Z][\w,.-]*\s+)*([a-z]{3,})\b", question
    ) if w.lower() not in WH_WORDS and w.lower() not in QUESTION_WORDS]
    noun = nouns[0] if nouns else next((w for w in re.findall(r"\b(?:which|what)\s+(\w+)", q) if w not in WH_WORDS), None)
    noun_match = 0
    if noun:
        stem = noun.rstrip("s")
        if re.search(rf"\b{re.escape(stem)}", n) is not None:
            noun_match = 1
        elif stem in ("college", "university") and re.search(r"\b(university|college|institute|academy|school)\b", n):
            noun_match = 1
    if "administrative territorial entity" in q and re.search(r"\b(county|district|oblast|parish|province|prefecture|municipality|borough|region|state)\b", n):
        noun_match = max(noun_match, 1)
    # Unit match: if the question asks "in square miles", prefer "526 people per square mile" (3 shared tokens) over "203 people per square kilometer" (2).
    q_tokens = {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", q) if w not in WH_WORDS and w not in QUESTION_WORDS}
    n_tokens = {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", n)}
    if has_digit and (q_tokens & n_tokens):
        noun_match = max(noun_match, len(q_tokens & n_tokens))
    return fits, not_in_q, noun_match


# A capitalized run, joined only by "of"/"the"/"de"/"von"...: "and" or "on" would glue two names
# together ("Ilse Marren and Tove Marren", "Norland Rail on March") and flag them as missing.
NAME = re.compile(r"[A-Z][\w'’&.-]*(?:\s+(?:(?:of|the|de|del|da|von|van)\s+)*[A-Z][\w'’&.-]*)*")
NUMBER = re.compile(r"\d[\d,.]*\d|\d")


def squash(text):
    """Lowercase letters and digits only: 'The Glass-Harbor' -> 'theglassharbor'."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower())


def missing_terms(facts, triples):
    """Names, titles and numbers written in the facts that no triple contains.
    A capitalized run also counts as kept without its first word ("Some Dell laptops" -> "Dell laptops")."""
    kept = " ".join(squash(t[k]) for t in triples for k in ("subject", "relation", "object"))
    missing = []
    for fact in facts:
        first = fact.split()[0] if fact.split() else ""
        for term in [*NAME.findall(fact), *NUMBER.findall(fact)]:
            term = term.rstrip(".")
            words = term.split()
            if words and words[0] == first and len(words) > 1:
                term, words = " ".join(words[1:]), words[1:]  # "Some Apple MacBook" -> "Apple MacBook"
            elif term == first and not term.isupper():
                continue  # a lone capitalized sentence start ("Some", "This") is not a name
            variants = [squash(term)] + ([squash(" ".join(words[1:]))] if len(words) > 1 else [])
            if all(v not in kept for v in variants) and len(variants[0]) > 1 and term not in missing:
                missing.append(term)
    return missing


QUESTION_WORDS = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how", "in", "on", "at", "of",
                  "the", "a", "an", "is", "was", "are", "were", "did", "does", "do", "for", "from", "by", "to"}


def question_anchors(question):
    """Named entities in a question: capitalized runs, without a leading question word, the sentence's own
    capitalized first word ("According", "Along") or a trailing "'s".
    "What ranking is held by Ilse Marren's alma mater?" -> ["Ilse Marren"]."""
    anchors = []
    for match in NAME.finditer(question):
        words = re.sub(r"['’]s?$", "", match.group()).split()
        if match.start() == 0:
            words = words[1:]  # the first word of a sentence is capitalized anyway
        while words and words[0].lower() in QUESTION_WORDS:
            words = words[1:]
        if words:
            anchors.append(" ".join(words))
    return anchors


def missing_anchors(question, texts):
    """Question entities none of whose words (3+ letters) appear in the texts. Strict on purpose: one shared
    word is enough to count as present, so paraphrases ("Ilse Marren's Port Halden studio") are never flagged."""
    words_in_texts = set(re.findall(r"[a-z0-9]+", " ".join(texts).lower()))
    missing = []
    for anchor in question_anchors(question):
        words = [w for w in re.findall(r"[a-z0-9]+", anchor.lower()) if len(w) >= 3]
        if words and not any(w in words_in_texts for w in words):
            missing.append(anchor)
    return missing


GENDERED = {"mother", "father", "son", "daughter", "brother", "sister", "wife", "husband", "aunt", "uncle",
            "niece", "nephew", "grandmother", "grandfather", "grandson", "granddaughter", "king", "queen"}


# Words that make a relation more specific: "youngest_son_of" says more than "son_of".
QUALIFIERS = {"youngest", "eldest", "oldest", "elder", "younger", "older", "first", "second", "third", "only", "half",
              "step", "adopted", "illegitimate", "biological", "former", "current", "late", "great", "grand"}


def gender_words(relation):
    """Gendered and qualifying words of a relation: a copy may drop them ("youngest_son_of" -> "child_of") but never
    add them ("son_of" -> "youngest_son_of", "child_of" -> "son_of")."""
    return set(raw_words(relation)) & (GENDERED | QUALIFIERS)


# Gendered family words and their neutral form: "A son_of B" implies "A child_of B" (never the reverse).
KIN = {"son": "child", "daughter": "child", "mother": "parent", "father": "parent", "brother": "sibling",
       "sister": "sibling", "husband": "spouse", "wife": "spouse"}
AUXILIARIES = {"is", "was", "are", "were", "the", "a", "an", "been", "being"}


def general_relations(relation):
    """More general forms of a relation, from specific to general: drop auxiliaries and qualifiers, then make
    family words neutral. "was_the_youngest_son_of" -> ["son_of", "child_of"]; "directed" -> []."""
    words = relation_name(relation).split("_")
    out = []
    plain = [w for w in words if w not in AUXILIARIES and w not in QUALIFIERS] or words
    neutral = [KIN.get(w, w) for w in plain]
    for form in ("_".join(plain), "_".join(neutral)):
        if form and form != relation_name(relation) and form not in out:
            out.append(form)
    return out


# Relations meaning "subject lies inside object", for preferring the more specific of two place answers.
INSIDE = {"located_in", "part_of", "is_in", "in", "within", "situated_in", "lies_in", "borough_of", "district_of",
          "city_in", "town_in", "capital_of", "is_located_in"}


def pick_answer(output, question, names, answer_type=None, candidates=None, inside=None):
    """ONA's best-fitting answer: an entity (not a compound term) of the asked-for type, as a readable name.
    names maps ONA atoms to entity names (OnaReasoner.names); works on saved runs too. inside: atom pairs
    (a, b) meaning a lies inside b; when both are answers of equal fit, the narrower a wins (a city, not its region)."""
    best = {}
    for term, f, c in re.findall(r"^Answer:\s*<(\S+) --> answer>\.[^\n]*frequency=([\d.]+), confidence=([\d.]+)",
                                 output, flags=re.M):
        if term in names:
            expectation = float(c) * (float(f) - 0.5) + 0.5
            best[term] = max(best.get(term, 0.0), expectation)
    if not best:
        return None
    cand_list = list(candidates) if candidates else []
    cand_set = set(cand_list)
    # Prefer answers the rules bind over the triples (ONA can also derive a self-join the rules do not allow),
    # and on exact confidence ties prefer the candidate from the most specific rule (earliest in cand_list).
    in_cands = lambda t: 1 if (not cand_set or t in cand_set) else 0
    cand_order = lambda t: -cand_list.index(t) if t in cand_set else -999
    narrower = lambda t: int(any((t, other) in (inside or ()) for other in best if other != t))
    term = max(best, key=lambda t: (in_cands(t), *answer_fit(question, names[t], answer_type),
                                    narrower(t), round(best[t], 4), cand_order(t)))
    ans = names[term]
    if re.search(r"\b(what|which) year\b", question.lower()):  # "In what year ..." -> "1987", not "4 May 1987"
        ym = re.fullmatch(rf"(?:\d{{1,2}}\s+)?(?:{MONTHS})[a-z]*\s+(?:\d{{1,2}},?\s+)?(\d{{4}})", ans.strip(), re.I)
        if ym:
            return ym.group(1)
    return ans


def is_var(x):
    return bool(re.fullmatch(r"\?[A-Za-z][A-Za-z0-9_]*", x))


def parse_json(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char in "[{":
            try:
                return decoder.raw_decode(text[index:])[0]
            except json.JSONDecodeError:
                pass
    return None


def error_details(error):
    """Keep the underlying network/SSL error hidden by the API client's wrapper."""
    parts, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        parts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__
    return " <- ".join(parts)


class OnaReasoner:
    def __init__(self, client, model=MODEL, cycles=500, repairs=2, plan=False, votes=1, merge_relations=False):
        """Use your existing DeepInfra OpenAI-compatible client. Options (all off reproduces the original pipeline):
        plan: the LLM first breaks the question into hops (PLAN prompt, question only), and the RULES prompt shows them.
        votes: solve() writes up to this many rule sets (the first at temperature 0, the others sampled), runs ONA on
               each, and returns the answer most of them agree on.
        merge_relations: each triple is also stated under its relation's general form ("was_youngest_son_of" ->
               "son_of" -> "child_of"), so rules match facts whichever wording the triples used."""
        self.client, self.model, self.cycles, self.repairs = client, model, cycles, repairs
        self.use_plan, self.votes, self.merge = plan, votes, merge_relations
        self.temperature = 0.0
        self.plan_text = ""  # the last question's hop plan
        self.relation_groups = []  # the last question's merged relation groups
        self.samples = []  # solve(): one entry per rule set written, with its answer
        self.last_narsese, self.last_output = None, None
        self.last_response = ""
        self.rules_raw = ""
        self.rule_attempts = []  # (raw reply, problems) per rule-generation attempt
        self.last_raw_triples = []
        self.last_triples = []
        self.last_rules = []
        self.aliases = {}  # entity name -> canonical name, for the last question
        self.candidates = []  # entity atoms the last question's rules can bind to the answer
        self.answer_type = None  # "date", "number", "place", "person" or "other", from the RULES reply
        self.names = {}
        self.ona_top = None  # ONA's own best answer, without preferring the rules' bindings (see answer())
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0,
                      "missing_usage": 0, "failed_requests": 0}
        self.request_log = []
        self._cache = {}
        self._alias_cache = {}
        self.skipped = []  # facts that produced no valid triples
        # How often each auto-repair in _validate fired for the last question, so a run can be audited:
        # literal_answer, inverse_relation, pruned_unlinked, bridged_constant, auto_hop, embedded_var.
        self.auto_repairs = Counter()
        self.missing_terms = []  # names/numbers still missing from every triple after the repair

    def _generate(self, prompt, max_tokens):
        """prompt is a string, or a chat message list for multi-turn repair. Retries transient API timeouts."""
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                break
            except Exception as error:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                self.usage["failed_requests"] += 1
                self.request_log.append({"model": self.model, "messages": list(messages), "error": error_details(error),
                                         "input_tokens": None, "output_tokens": None})
                raise
        # Account before parsing: malformed or truncated generations still use tokens.
        self.usage["requests"] += 1
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        self.usage["input_tokens"] += input_tokens or 0
        self.usage["output_tokens"] += output_tokens or 0
        if input_tokens is None or output_tokens is None:
            self.usage["missing_usage"] += 1
        self.last_response = response.choices[0].message.content or ""
        # The full exchange, so a run can be inspected (or reused as training data) afterwards.
        self.request_log.append({"model": self.model, "messages": list(messages), "response": self.last_response,
                                 "finish_reason": response.choices[0].finish_reason,
                                 "input_tokens": input_tokens, "output_tokens": output_tokens})
        if response.choices[0].finish_reason == "length":
            raise ValueError("Model response was truncated. Inspect ona.last_response.")
        return self.last_response

    @staticmethod
    def paragraphs(facts, size=12):
        """Facts as paragraph groups of at most `size` facts. A flat list counts as one paragraph.
        Facts are strings or dictionaries with a "text" field."""
        groups = facts if facts and all(isinstance(g, (list, tuple)) for g in facts) else [facts]
        out = []
        for group in groups:
            texts = [f["text"] if isinstance(f, dict) else f for f in group]
            if not all(isinstance(t, str) and t.strip() for t in texts):
                raise ValueError("Each fact must be a nonempty string or a dictionary containing text.")
            out += [texts[i:i + size] for i in range(0, len(texts), size)]
        return [g for g in out if g]

    @staticmethod
    def _parse_triples(raw):
        parsed = parse_json(raw)
        parsed = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []
        return [{k: t[k].strip() for k in ("subject", "relation", "object")} for t in parsed
                if isinstance(t, dict) and all(isinstance(t.get(k), str) and t[k].strip()
                                               for k in ("subject", "relation", "object"))]

    def _paragraph_triples(self, facts):
        """One paragraph's facts -> triples in one call; names or numbers dropped from every triple are
        sent back once (TRIPLE_REPAIR), and the repair is kept only if it drops fewer of them."""
        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
        messages = [{"role": "user", "content": TRIPLES.format(facts=numbered)}]
        max_tokens = min(2048, 128 + 96 * len(facts))
        raw = self._generate(messages, max_tokens=max_tokens)
        triples = self._parse_triples(raw)
        missing = missing_terms(facts, triples)
        if missing and triples:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": TRIPLE_REPAIR.format(missing=", ".join(f'"{m}"' for m in missing))}]
            try:
                repaired = self._parse_triples(self._generate(messages, max_tokens=max_tokens))
            except Exception:  # truncated or timed-out repair: keep the first answer
                repaired = []
            still = missing_terms(facts, repaired)
            if repaired and len(still) < len(missing):
                triples, missing = repaired, still
        self.missing_terms += missing
        return triples

    def triples(self, facts):
        """Facts (a flat list or paragraph groups) -> triples, converting each paragraph's facts together."""
        triples = []
        for group in self.paragraphs(facts):
            key = (self.model, tuple(group))
            if key not in self._cache:
                try:
                    self._cache[key] = self._paragraph_triples(group)
                except ValueError:  # truncated reply: skip this paragraph instead of failing the question
                    self._cache[key] = []
                if not self._cache[key]:
                    self.skipped += group
            triples += [{**t, "f": 1.0, "c": 0.5, "fact": " ".join(group)} for t in self._cache[key]]
        return triples

    def _alias_groups(self, names):
        """LLM alias groups over this question's entity names; [] when the reply is unusable."""
        key = (self.model, tuple(names))
        if key not in self._alias_cache:
            try:
                raw = self._generate(ALIASES.format(names="\n".join(names)), max_tokens=512)
            except Exception:  # truncated or timed-out: fall back to the spelling rules alone
                raw = ""
            lookup = {n.casefold(): n for n in names}
            # Collect all valid JSON lists in raw (handles mid-reply self-corrections and flat single-group lists)
            decoder = json.JSONDecoder()
            lists_of_lists, flat_lists = [], []
            for idx, ch in enumerate(raw):
                if ch == "[":
                    try:
                        val = decoder.raw_decode(raw[idx:])[0]
                    except json.JSONDecodeError:
                        continue
                    if isinstance(val, list):
                        if any(isinstance(x, list) for x in val):
                            lists_of_lists.append([x for x in val if isinstance(x, list)])
                        elif val and all(isinstance(x, str) for x in val):
                            flat_lists.append(val)
            raw_groups = lists_of_lists[-1] if lists_of_lists else flat_lists
            groups = []
            for group in raw_groups:
                if isinstance(group, list):
                    members = {lookup[x.strip().casefold()] for x in group
                               if isinstance(x, str) and x.strip().casefold() in lookup}
                    if len(members) > 1 and sorted(members) not in groups:
                        groups.append(sorted(members))
            self._alias_cache[key] = groups
        return self._alias_cache[key]

    def canonicalize(self, triples):
        """Give each entity one name across all facts, so ONA can chain facts that spell it differently."""
        counts = Counter(t[k] for t in triples for k in ("subject", "object"))
        names = sorted(counts)
        parent = {n: n for n in names}
        members_of = {n: {n} for n in names}
        qualifiers = {n: {qualifier(n)} - {""} for n in names}  # per cluster root
        # Two entities linked by a non-alias relation (e.g. father_of, married_to) are distinct and never merge.
        alias_rels = {"also_known_as", "also_called", "birth_name", "real_name", "stage_name",
                      "renamed_from", "renamed_to", "later_known_as", "professionally_known_as", "nickname"}
        distinct_pairs = set()
        for t in triples:
            if relation_name(t["relation"]) not in alias_rels and t["subject"] != t["object"]:
                distinct_pairs.add((t["subject"], t["object"]))
                distinct_pairs.add((t["object"], t["subject"]))
        # Bases written with two different regions ("Kestrel County, Norland" / ", Vessia"): the bare
        # "Kestrel County" could be either, so names on such a base are never merged.
        regions = {}
        for n in names:
            regions.setdefault(surface_key(n), set()).add(qualifier(n))
        ambiguous = {key for key, quals in regions.items() if len(quals - {""}) >= 2}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b, check_plausible=False):
            ra, rb = find(a), find(b)
            if ra == rb or surface_key(a) in ambiguous or surface_key(b) in ambiguous:
                return
            if any((ma, mb) in distinct_pairs for ma in members_of[ra] for mb in members_of[rb]):
                return
            # "Kestrel County, Norland" and "Kestrel County, Vessia" are different places.
            if qualifiers[ra] and qualifiers[rb] and not qualifiers[ra] & qualifiers[rb]:
                return
            # Prevent transitive chaining of incompatible names (e.g. "West Kestrel" ~ "Kestrel" ~ "Kestrel Hills")
            if check_plausible and not all(plausible_alias(ma, mb) for ma in members_of[ra] for mb in members_of[rb]):
                return
            parent[rb] = ra
            qualifiers[ra] |= qualifiers[rb]
            members_of[ra] |= members_of[rb]

        # Spelling: same name up to a leading "the", parentheses, St/Saint, or ", Region" - unless the regions differ.
        by_key = {}
        for name in names:
            by_key.setdefault(surface_key(name), []).append(name)
        for group in by_key.values():
            if len({qualifier(n) for n in group} - {""}) <= 1:
                for other in group[1:]:
                    union(group[0], other, check_plausible=False)
        # The LLM's alias groups, keeping only pairs whose wording makes them plausible aliases.
        if len(names) > 1:
            for group in self._alias_groups(names):
                for i, a in enumerate(group):
                    for b in group[i + 1:]:
                        if plausible_alias(a, b):
                            union(a, b, check_plausible=True)
        clusters = {}
        for name in names:
            clusters.setdefault(find(name), []).append(name)
        self.aliases = {}
        for members in clusters.values():
            # Most-used spelling wins; ties go to the shortest.
            canonical = min(members, key=lambda n: (-counts[n], len(n), n))
            self.aliases.update({n: canonical for n in members})

        pure_date = re.compile(
            rf"(?:c\.\s*|around\s*|before\s*|after\s*)?(?:\d{{1,2}}\s+)?(?:(?:{MONTHS})[a-z]*\s+)?(?:\d{{1,2}},?\s+)?\d{{3,4}}(?:\s*(?:bc|ad|bce|ce))?",
            re.I,
        )
        result = []
        for t in triples:
            subject, obj = self.aliases[t["subject"]], self.aliases[t["object"]]
            rel = t["relation"]
            # Fix reversed parent/child triples on possessive nouns: "Ann | mother_of | Ann's mother" -> flip
            if re.fullmatch(rf"{re.escape(subject)}['’]s\s+(mother|father|parent)", obj, flags=re.I) and atom(rel) in ("mother_of", "father_of", "parent_of"):
                subject, obj = obj, subject
            # "born in 1943" / "died in 1752" is a date (born_on / died_on), not a place (born_in / died_in).
            if relation_name(rel) in ("born_in", "was_born_in") and pure_date.fullmatch(obj.strip()):
                rel = "born_on"
            elif relation_name(rel) in ("died_in", "was_died_in") and pure_date.fullmatch(obj.strip()):
                rel = "died_on"
            if atom(subject) != atom(obj):  # "Norland Rail renamed_from Norland Railway" is now a self-loop
                result.append({**t, "subject": subject, "relation": rel, "object": obj})

        # Add automatic structural bridge triples when one entity contains another via prepositions/commas/proper nouns
        canon_names = sorted({x for t in result for x in (t["subject"], t["object"])})
        by_surf = {surface_key(c): c for c in canon_names}
        existing_pairs = {(atom(t["subject"]), atom(t["object"])) for t in result}

        def add_bridge(s, r, o):
            if atom(s) != atom(o) and (atom(s), atom(o)) not in existing_pairs:
                existing_pairs.add((atom(s), atom(o)))
                result.append({"subject": s, "relation": r, "object": o, "f": 1.0, "c": 0.5, "fact": ""})

        for c in canon_names:
            # 1. Comma-separated places: "Port Halden, Norland" -> located_in -> "Norland"
            if "," in c and not re.search(r"\d", c):
                for part in [p.strip() for p in c.split(",")[1:] if p.strip()]:
                    target = by_surf.get(surface_key(part), part)
                    add_bridge(c, "located_in", target)
            # 2. Possessive prefix: "Ilse Marren's Harbor Films" -> of -> "Ilse Marren" & "Harbor Films"
            poss_match = re.match(r"^(.+?)['’]s\s+(.+)$", c)
            if poss_match:
                owner, owned = poss_match.group(1).strip(), poss_match.group(2).strip()
                if surface_key(owner) in by_surf:
                    add_bridge(c, "of", by_surf[surface_key(owner)])
                if surface_key(owned) in by_surf:
                    add_bridge(c, "of", by_surf[surface_key(owned)])
            # 3. Prepositional / dash suffixes: "bay at Port Halden" -> "Port Halden", "wife of Tove" -> "Tove"
            for prep in (" at ", " in ", " of ", " from ", " with ", " - "):
                if prep in c.lower():
                    tail = c[c.lower().rfind(prep) + len(prep):].strip()
                    target = by_surf.get(surface_key(tail))
                    if target:
                        rel = "located_in" if prep in (" at ", " in ") else "of"
                        add_bridge(c, rel, target)
                    elif re.match(r"^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*$", tail) and tail.lower() not in QUESTION_WORDS:
                        # Even if "Oregon" or "Chile" or "Pyrrhus" wasn't a standalone entity yet, create the bridge!
                        rel = "located_in" if prep in (" at ", " in ") else "of"
                        add_bridge(c, rel, tail)
                        by_surf.setdefault(surface_key(tail), tail)
            # 4. Organization prefix: "KVA Halden office" -> part_of -> "KVA"
            words_c = c.split()
            if len(words_c) >= 2 and words_c[0].isupper() and len(words_c[0]) >= 3:
                sk = surface_key(words_c[0])
                if sk in by_surf:
                    add_bridge(c, "part_of", by_surf[sk])
            # 5. Sub-entity containment: if a multi-word proper entity (like "Ilse Marren") is a substring of a
            # longer entity ("early films of director Ilse Marren"), link them with "of"
            for other in canon_names:
                if other != c and len(other) >= 5 and other.lower() in c.lower() and not re.search(r"\d", other):
                    if re.search(rf"\b{re.escape(other.lower())}\b", c.lower()):
                        add_bridge(c, "of", other)
        return result

    def merge_relations(self, triples):
        """Restate each triple under its relation's general form (general_relation), so a rule matches a fact
        whichever wording the triples used: "Arthur youngest_son_of Robert" also gives "Arthur son_of Robert" and
        "Arthur child_of Robert"; "X was_born_in Y" also gives "X born_in Y". Copies are added, never replacing a
        triple, and only go from specific to general, never the reverse."""
        result, seen = list(triples), {(atom(t["subject"]), atom(t["relation"]), atom(t["object"])) for t in triples}
        groups = {}
        for t in triples:
            r = relation_name(t["relation"])
            for general in general_relations(r):
                key = (atom(t["subject"]), atom(general), atom(t["object"]))
                groups.setdefault(r, set()).add(general)
                if key not in seen:
                    seen.add(key)
                    result.append({**t, "relation": general, "merged_from": r})
        self.relation_groups = [[r, *sorted(g)] for r, g in sorted(groups.items())]
        return result

    @staticmethod
    def add_inverses(triples):
        """Restate facts under the opposite relation (voiced_by <-> voiced) when both names are in use."""
        relations = {atom(t["relation"]) for t in triples}
        seen = {(atom(t["subject"]), atom(t["relation"]), atom(t["object"])) for t in triples}
        result = list(triples)
        for t in triples:
            relation = atom(t["relation"])
            flipped = inverse(relation)
            if flipped not in relations and relation not in SYMMETRIC and relation not in INVERSES:
                continue
            key = (atom(t["object"]), flipped, atom(t["subject"]))
            if key not in seen:
                seen.add(key)
                result.append({**t, "subject": t["object"], "relation": flipped,
                               "object": t["subject"], "inverse": True})
        return result

    def plan(self, question):
        """The question as numbered hops, from the question alone (no facts, no gold); "" if the reply is unusable."""
        try:
            raw = self._generate(PLAN.format(question=question), max_tokens=160)
        except Exception:
            return ""
        hops = [line.strip() for line in raw.splitlines() if re.match(r"\s*\d+[.)]\s+\S", line)]
        return "\n".join(hops[:5])

    def rules(self, question, triples, plan=""):
        lines = []
        for t in triples:
            line = f"{t['subject']} | {relation_name(t['relation'])} | {t['object']}"
            if line not in lines:
                lines.append(line)
        plan = f"Plan (write one condition per hop):\n{plan}\n" if plan else ""
        messages = [{"role": "user", "content": RULES.format(question=question, plan=plan, triples="\n".join(lines))}]
        self.rule_attempts = []
        self.answer_type = None
        fallback = []  # first valid rules, kept if repairs never reach the asked-for answer type
        raw = self._generate(messages, max_tokens=512)
        for attempt in range(self.repairs + 1):
            self.rules_raw = raw
            parsed = parse_json(raw)
            rules = parsed.get("rules") if isinstance(parsed, dict) else None
            if isinstance(parsed, dict) and str(parsed.get("answer_type")).lower() in (*ANSWER_TYPES, "other"):
                self.answer_type = parsed["answer_type"].lower()
            if question_type(question):
                self.answer_type = question_type(question)
            valid, problems = self._validate(rules, triples, question=question, auto_hop=attempt == self.repairs)
            if rules is None:
                problems = ['The reply did not contain a JSON object with a "rules" list.']
            if valid:
                answers = self._answer_names(valid, triples)
                wanted = ANSWER_TYPES.get(self.answer_type)
                if wanted is None or any(answer_fit(question, a, self.answer_type)[0] for a in answers):
                    self.rule_attempts.append((raw, problems))
                    return valid
                fallback = fallback or valid
                problems = [f"The answer_type is {wanted}, but the rules only return: "
                            f"{', '.join(answers[:6])}. Use relations that lead to {wanted}."]
            self.rule_attempts.append((raw, problems))
            if attempt == self.repairs:
                return fallback
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": REPAIR.format(problems="\n".join(f"- {p}" for p in problems))}]
            try:
                raw = self._generate(messages, max_tokens=512)
            except Exception:  # truncated or timed-out repair: keep what we have
                return fallback
        return fallback

    def _answer_names(self, rules, triples):
        """Entity names the rules bind to their answer variable, in order found."""
        facts = {(atom(t["subject"]), atom(t["relation"]), atom(t["object"])) for t in triples}
        display = {atom(t[k]): t[k] for t in triples for k in ("subject", "object")}
        names = []
        for rule in rules:
            for binding in self._bindings(rule["conditions"], facts):
                name = display[binding[rule["answer"]]]
                if name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _closest(name, options, display=None):
        """Exact atom match, else a near-identical spelling (small LLM typos only). Compares full readable names
        (display maps an atom to its full name; long atoms are truncated and hashed) at 90% similarity, and
        never swaps names that differ in a number or roman numeral: mother_of is not father_of, born_in is not
        born_on, Edward I is not Edward II."""
        key = atom(name)
        if key in options:
            return key
        full = {readable(display.get(o, o) if display else o): o for o in options}
        match = difflib.get_close_matches(readable(name), list(full), n=1, cutoff=0.9)
        if not match:
            return None
        marks = lambda text: set(re.findall(r"\b(?:\d+|[ivx]+)\b", text))  # numbers and roman numerals
        return full[match[0]] if marks(match[0]) == marks(readable(name)) else None

    @staticmethod
    def _unlinked(conditions, answer):
        """1-based numbers of conditions not chained to the answer. Conditions with variables must reach the
        answer's condition through shared variables; sharing a constant is not enough ("the_dynasty defeated ?B"
        would then mean anything the dynasty defeated). Fully constant conditions may attach by a shared entity."""
        variables = lambda c: {x for x in (c[0], c[2]) if is_var(x)}
        linked = {i for i, c in enumerate(conditions) if answer in variables(c)}
        changed = True
        while changed:
            changed = False
            linked_vars = {x for i in linked for x in variables(conditions[i])}
            linked_terms = {x for i in linked for x in (conditions[i][0], conditions[i][2])}
            for i, c in enumerate(conditions):
                if i in linked:
                    continue
                joins = variables(c) & linked_vars if variables(c) else {c[0], c[2]} & linked_terms
                if joins:
                    linked.add(i)
                    changed = True
        return [i + 1 for i in range(len(conditions)) if i not in linked]

    @staticmethod
    def _bindings(conditions, facts, return_used=False):
        """All variable assignments that satisfy every condition at once, without reusing the exact same
        ground fact twice in a single rule AND ensuring distinct variables bind to distinct entities
        (otherwise "?A and ?B both had hits on X" could bind ?A and ?B to the same person). A variable never binds
        to an entity the rule names itself: "the father of X" is never X."""
        constants = {x for s, _, o in conditions for x in (s, o) if not is_var(x)}
        results = [(dict(), set())]
        for s, r, o in conditions:
            extended = []
            for binding, used_facts in results:
                for fs, fr, fo in facts:
                    if fr != r or (fs, fr, fo) in used_facts:
                        continue
                    new, ok = dict(binding), True
                    for x, value in ((s, fs), (o, fo)):
                        if is_var(x):
                            if x in new:
                                ok = ok and new[x] == value
                            else:
                                # Distinct variables in the same rule should not bind to the same entity
                                ok = ok and value not in new.values() and value not in constants
                                new[x] = value
                        else:
                            ok = ok and x == value
                    if ok:
                        extended.append((new, used_facts | {(fs, fr, fo)}))
            results = extended
        return results if return_used else [b for b, _ in results]

    def _auto_bridge_path(self, fixed, answer, facts):
        """If a rule of 2-4 conditions stops binding at condition k because the LLM skipped a 1-hop link
        (e.g. "Captain Vos | played_by | Ilse Marren" between a role and its actor), insert the first link in `facts`
        that makes the whole rule bind. Counted as auto_hop, since it is a search rather than the LLM's rule."""
        if len(fixed) >= 5:
            return None
        partial = [{}]
        for idx_c, cond in enumerate(fixed):
            next_b = self._bindings(fixed[:idx_c + 1], facts)
            if not next_b and idx_c > 0:
                vars_in_cond = [x for x in (cond[0], cond[2]) if is_var(x) and any(x in b for b in partial)]
                if len(vars_in_cond) == 1:
                    v = vars_in_cond[0]
                    bound_vals = {b[v] for b in partial if v in b}
                    # Find all 1-hop neighbors of `bound_vals` in `facts`
                    candidate_rels = []
                    for fs, fr, fo in facts:
                        if fs in bound_vals and (fr, True) not in candidate_rels:
                            candidate_rels.append((fr, True))   # v | fr | ?BRIDGE_HOP
                        if fo in bound_vals and (fr, False) not in candidate_rels:
                            candidate_rels.append((fr, False))  # ?BRIDGE_HOP | fr | v
                    new_var = "?AUTO_HOP"
                    replaced_cond = [(new_var if x == v else x) for x in cond]
                    for fr, v_is_subj in candidate_rels:
                        if fr == cond[1]:  # repeating the failing relation turns a hop into "shares it with"
                            continue
                        bridge = [v, fr, new_var] if v_is_subj else [new_var, fr, v]
                        trial = fixed[:idx_c] + [bridge, replaced_cond] + fixed[idx_c + 1:]
                        # Also replace `v` in subsequent conditions if needed
                        trial_2 = fixed[:idx_c] + [bridge] + [[(new_var if x == v else x) for x in c] for c in fixed[idx_c:]]
                        for candidate_rule in (trial, trial_2):
                            if not self._unlinked(candidate_rule, answer) and self._bindings(candidate_rule, facts):
                                return candidate_rule
                break
            partial = next_b
        return None

    def _validate(self, rules, triples, question="", auto_hop=True):
        """Repair rules (aliases, spelling, argument order, literal answers, constant bridges, 1-hop path completion);
        return (valid rules, problems for the LLM). auto_hop: allow the 1-hop path search (rules() allows it only on
        the last attempt, so the LLM's own repair comes first)."""
        if not isinstance(rules, list):
            return [], []
        facts = {(atom(t["subject"]), atom(t["relation"]), atom(t["object"])) for t in triples}
        relations = sorted({r for _, r, _ in facts})
        entities = sorted({x for s, _, o in facts for x in (s, o)})
        display = {atom(t[k]): t[k] for t in triples for k in ("subject", "object")}
        rel_display = {atom(t["relation"]): relation_name(t["relation"]) for t in triples}
        aliases = {atom(alias): atom(canonical) for alias, canonical in self.aliases.items()}
        q_words = {w for a in question_anchors(question) for w in re.findall(r"[a-z0-9]+", a.lower()) if len(w) >= 3}

        def entity(x):
            found = aliases.get(atom(x)) or self._closest(x, entities, display)
            if found:
                return found
            sk = surface_key(x)
            if sk:
                sk_matches = [e for e in entities if surface_key(display.get(e, e)) == sk]
                if len(sk_matches) == 1:
                    return sk_matches[0]
            pa_matches = [e for e in entities if plausible_alias(x, display.get(e, e))]
            if len(pa_matches) == 1:
                return pa_matches[0]
            return None

        def matches(s, r, o):
            return any(r == fr and (is_var(s) or s == fs) and (is_var(o) or o == fo)
                       for fs, fr, fo in facts)

        def around(e):
            """Up to 14 triples of entity e, for the REPAIR message."""
            used = sorted({f"{display[e]} | {rel_display[fr]} | {display[fo]}" for fs, fr, fo in facts if fs == e}
                          | {f"{display[fs]} | {rel_display[fr]} | {display[e]}" for fs, fr, fo in facts if fo == e})
            return "; ".join(used[:14]) or "none"

        valid, seen, problems = [], set(), []
        for number, rule in enumerate(rules[:3], 1):
            if not isinstance(rule, dict):
                problems.append(f"Rule {number} is not a JSON object.")
                continue
            conditions, answer = rule.get("conditions"), rule.get("answer")
            # A literal "answer" in the answer field or a condition means the answer variable, not an entity.
            literal = lambda x: isinstance(x, str) and x.strip().lower() in ("answer", "?answer")
            if literal(answer):
                answer = "?ANSWER"
            if isinstance(conditions, list):
                conditions = [[("?ANSWER" if literal(x) else x) for x in c] if isinstance(c, list) else c
                              for c in conditions]

            # Auto-repair 0: Expand string-interpolated variables like "wife of?A" or "inaugural parades of Presidents?A"
            # into a clean variable "?EMBED_k" + an explicit bridge condition ["?EMBED_k", "of", "?A"].
            if isinstance(conditions, list):
                expanded_conds = []
                embed_idx = 0
                for c in conditions:
                    if isinstance(c, list) and len(c) == 3 and all(isinstance(x, str) for x in c):
                        s_c, r_c, o_c = c[0].strip(), c[1].strip(), c[2].strip()
                        new_pair = [s_c, o_c]
                        extra = []
                        for pos, val in enumerate(new_pair):
                            if not is_var(val):
                                var_match = re.search(r"(\?[A-Za-z][A-Za-z0-9_]*)", val)
                                if var_match:
                                    emb_var = var_match.group(1)
                                    embed_idx += 1
                                    holder_var = f"?EMBED_{embed_idx}"
                                    new_pair[pos] = holder_var
                                    extra.append([holder_var, "of", emb_var])
                                    self.auto_repairs["embedded_var"] += 1
                        expanded_conds.append([new_pair[0], r_c, new_pair[1]])
                        expanded_conds.extend(extra)
                    else:
                        expanded_conds.append(c)
                conditions = expanded_conds

            # Auto-repair: if LLM put a literal entity in "answer", convert it to ?ANSWER ONLY if that entity
            # already appears in a condition connected to another variable/entity AND fits answer_type.
            if isinstance(answer, str) and not is_var(answer) and isinstance(conditions, list):
                ans_ent = entity(answer)
                if ans_ent is not None and answer_fit(question, display.get(ans_ent, answer), self.answer_type)[0]:
                    replaced = False
                    new_conds = []
                    for c in conditions:
                        if isinstance(c, list) and len(c) == 3:
                            s_c, r_c, o_c = c
                            if not is_var(s_c) and entity(s_c) == ans_ent:
                                s_c, replaced = "?ANSWER", True
                            if not is_var(o_c) and entity(o_c) == ans_ent:
                                o_c, replaced = "?ANSWER", True
                            new_conds.append([s_c, r_c, o_c])
                        else:
                            new_conds.append(c)
                    if replaced:
                        conditions, answer = new_conds, "?ANSWER"
                        self.auto_repairs["literal_answer"] += 1

            # Up to 5 conditions; _chain splits long rules into binary steps so ONA never sees more than two.
            if isinstance(conditions, list) and len(conditions) > 5:
                problems.append(f"Rule {number} has {len(conditions)} conditions; the limit is 5. Keep only the chain "
                                f"from the known entity to the answer (one condition per hop) and drop conditions "
                                f"for descriptive details of the question.")
                continue
            if not (isinstance(conditions, list) and conditions):
                problems.append(f"Rule {number} needs 1 to 5 conditions.")
                continue
            if not (isinstance(answer, str) and is_var(answer)):
                problems.append(f"Rule {number}: \"answer\" must be a variable such as ?B.")
                continue
            prep_swap = {"born_in": "born_on", "born_on": "born_in", "died_in": "died_on", "died_on": "died_in"}
            fixed = []
            for c in conditions:
                if not (isinstance(c, list) and len(c) == 3
                        and all(isinstance(x, str) and x.strip() for x in c)):
                    problems.append(f"Rule {number}: condition {c!r} must be [subject, relation, object].")
                    break
                s, r, o = (x.strip() for x in c)
                r_fixed = self._closest(r, relations, rel_display)
                flipped_rel = False
                if r_fixed is None:
                    alt_r = prep_swap.get(atom(r))
                    if alt_r in relations:
                        r_fixed = alt_r
                    else:
                        # Check if the inverse relation is in the triples
                        inv_r = self._closest(inverse(relation_name(r)), relations, rel_display)
                        if inv_r is not None:
                            r_fixed, flipped_rel = inv_r, True
                            self.auto_repairs["inverse_relation"] += 1
                        else:
                            problems.append(f"Rule {number}: relation \"{r}\" is not in the triples.")
                            break
                missing = [x for x in (s, o) if not is_var(x) and entity(x) is None]
                if missing:
                    problems.append(f"Rule {number}: entity \"{missing[0]}\" is not in the triples.")
                    break
                s = s if is_var(s) else entity(s)
                o = o if is_var(o) else entity(o)
                if flipped_rel:
                    s, o = o, s
                if not matches(s, r_fixed, o) and matches(o, r_fixed, s):
                    s, o = o, s  # LLM reversed the relation direction
                if not matches(s, r_fixed, o) and prep_swap.get(r_fixed) in relations:
                    alt_r = prep_swap[r_fixed]
                    if matches(s, alt_r, o):
                        r_fixed = alt_r
                    elif matches(o, alt_r, s):
                        r_fixed, s, o = alt_r, o, s
                if not matches(s, r_fixed, o):
                    constant = s if not is_var(s) else o
                    if not is_var(constant) and constant in display:
                        problems.append(f"Rule {number}: no triple matches {c}. "
                                        f"Triples with {display[constant]}: {around(constant)}.")
                    else:
                        problems.append(f"Rule {number}: no triple matches {c}.")
                    break
                fixed.append([s, r_fixed, o])
            else:
                variables = {x for c in fixed for x in (c[0], c[2]) if is_var(x)}
                if answer not in variables:
                    problems.append(f"Rule {number}: answer {answer} does not appear in any condition.")
                    continue
                loose = self._unlinked(fixed, answer)
                if loose:
                    # Auto-repair 1 (Bridge Constants): Replace any constant shared between loose and linked
                    # conditions with a bridging variable (?BRIDGE_k), ONLY if the remaining constants still
                    # mention a question anchor (so the rule still starts from the question, not from a guess).
                    bridged = [list(c) for c in fixed]
                    loose_set = set(loose)
                    linked_consts = {x for idx, c in enumerate(bridged, 1) if idx not in loose_set for x in (c[0], c[2]) if not is_var(x)}
                    loose_consts = {x for idx, c in enumerate(bridged, 1) if idx in loose_set for x in (c[0], c[2]) if not is_var(x)}
                    shared_consts = linked_consts & loose_consts
                    remaining_consts = (linked_consts | loose_consts) - shared_consts
                    rem_text = " ".join(display.get(x, x).lower() for x in remaining_consts)
                    keeps_anchor = not q_words or any(w in rem_text for w in q_words)
                    if shared_consts and remaining_consts and keeps_anchor:
                        for b_idx, sc in enumerate(sorted(shared_consts), 1):
                            b_var = f"?BRIDGE_{b_idx}"
                            for c in bridged:
                                if c[0] == sc:
                                    c[0] = b_var
                                if c[2] == sc:
                                    c[2] = b_var
                        if not self._unlinked(bridged, answer) and self._bindings(bridged, facts):
                            fixed = bridged
                            self.auto_repairs["bridged_constant"] += 1
                            loose = []
                if loose:
                    # Auto-repair 2 (Prune Unlinked): Only prune unlinked conditions if the remaining linked conditions
                    # still mention at least one anchor word from the question AND are not sibling/parallel guesses.
                    linked_conds = [c for idx, c in enumerate(fixed, 1) if idx not in loose]
                    unlinked_conds = [c for idx, c in enumerate(fixed, 1) if idx in loose]
                    linked_consts = [display.get(x, x).lower() for c in linked_conds for x in (c[0], c[2]) if not is_var(x)]
                    keeps_anchor = not q_words or any(w in " ".join(linked_consts) for w in q_words)
                    same_target = ({x for c in linked_conds for x in (c[0], c[2]) if not is_var(x)}
                                   == {x for c in unlinked_conds for x in (c[0], c[2]) if not is_var(x)})
                    if linked_conds and linked_consts and keeps_anchor and not same_target and not self._unlinked(linked_conds, answer) and self._bindings(linked_conds, facts):
                        fixed = linked_conds
                        self.auto_repairs["pruned_unlinked"] += 1
                        loose = []
                if loose:
                    problems.append(f"Rule {number}: conditions {', '.join(map(str, loose))} are not chained to the "
                                    f"answer {answer} through shared variables, so the answer does not depend on "
                                    f"them. Replace any intermediate entity shared between conditions with a variable (?A, ?B, ?C).")
                    continue
                # Convert any intermediate constant shared across 2+ conditions into a variable if there is another anchor constant.
                const_counts = Counter(x for c in fixed for x in (c[0], c[2]) if not is_var(x))
                repeated_consts = [k for k, v in const_counts.items() if v >= 2]
                if len(fixed) >= 3 and repeated_consts and len(const_counts) > len(repeated_consts):
                    for b_idx, rc in enumerate(repeated_consts, 1):
                        b_var = f"?HOP_{b_idx}"
                        test_fixed = [[(b_var if x == rc else x) for x in c] for c in fixed]
                        if not self._unlinked(test_fixed, answer) and self._bindings(test_fixed, facts):
                            fixed = test_fixed
                if auto_hop and not self._bindings(fixed, facts):
                    # Try inserting one skipped 1-hop link before rejecting the rule.
                    completed = self._auto_bridge_path(fixed, answer, facts)
                    if completed is not None:
                        fixed = completed
                        self.auto_repairs["auto_hop"] += 1
                if not self._bindings(fixed, facts):
                    # Diagnostic join failure: find which condition broke the chain and show what the variable bound to.
                    partial = [{}]
                    diag = ""
                    for idx_c, cond in enumerate(fixed):
                        next_b = self._bindings(fixed[:idx_c + 1], facts)
                        if not next_b:
                            vars_in_cond = [x for x in (cond[0], cond[2]) if is_var(x) and any(x in b for b in partial)]
                            hints = []
                            for v in vars_in_cond:
                                vals = list(dict.fromkeys(b[v] for b in partial if v in b))[:2]
                                for val in vals:
                                    hints.append(f"{v} bound to \"{display.get(val, val)}\" (whose triples are: {around(val)})")
                            if hints:
                                diag = f" At condition {idx_c + 1} ({cond[0]} | {cond[1]} | {cond[2]}), {'; '.join(hints)}."
                            break
                        partial = next_b
                    problems.append(f"Rule {number}: each condition matches some triple, but no entities satisfy "
                                    f"all of them together.{diag} Check that the variables chain the hops correctly.")
                    continue
                # Same rule with different variable letters is a duplicate.
                order = {}
                for x in (x for c in fixed for x in (c[0], c[2])):
                    if is_var(x):
                        order.setdefault(x, len(order))
                key = (tuple(tuple(order.get(x, x) for x in c) for c in fixed), order[answer])
                if key not in seen:
                    seen.add(key)
                    valid.append({"conditions": fixed, "answer": answer})
        return valid, problems

    @staticmethod
    def _chain(rule, tag):
        """A rule as Narsese implications of at most two conditions each.
        ONA drops the 3rd variable of a product like `<($b * $a * $c) --> r0_s2>`, so an intermediate step
        carries at most 2 variables."""
        conditions = list(rule["conditions"])
        ans_var = rule["answer"]
        c_vars = lambda c: {x for x in (c[0], c[2]) if is_var(x)}

        # Drop dead-end existential conditions containing a non-answer variable that appears only once in the rule,
        # provided the remaining conditions still connect to ans_var (ONA cannot unify dangling existential variables).
        changed = True
        while changed and len(conditions) > 1:
            changed = False
            var_counts = Counter(x for c in conditions for x in c_vars(c))
            for i, c in enumerate(conditions):
                if any(v != ans_var and var_counts[v] == 1 for v in c_vars(c)):
                    rem = conditions[:i] + conditions[i + 1:]
                    if any(ans_var in c_vars(r) for r in rem) and not OnaReasoner._unlinked(rem, ans_var):
                        conditions = rem
                        changed = True
                        break

        # Greedily order 3+ conditions starting from a constant-bearing condition and minimizing carried variables.
        if len(conditions) > 2:
            rem = list(conditions)
            first = min(rem, key=lambda c: (len(c_vars(c)), ans_var in c_vars(c)))
            rem.remove(first)
            ordered = [first]
            seen_v = set(c_vars(first))
            while len(rem) > 1:
                def score_next(c):
                    cv = c_vars(c)
                    connected = 0 if (cv & seen_v) else 1
                    new_seen = seen_v | cv
                    later_v = {x for r in rem if r is not c for x in c_vars(r)} | {ans_var}
                    carried = len(new_seen & later_v)
                    has_ans = 1 if ans_var in cv else 0
                    return (connected, carried, has_ans, len(cv))
                nxt = min(rem, key=score_next)
                rem.remove(nxt)
                ordered.append(nxt)
                seen_v |= c_vars(nxt)
            ordered.extend(rem)
            conditions = ordered

        variables = {}

        def term(value):
            return variables.setdefault(value, "$" + chr(ord("a") + len(variables))) if is_var(value) else atom(value)

        statement = lambda c: f"<({term(c[0])} * {term(c[2])}) --> {atom(c[1])}>"
        answer = f"<{term(rule['answer'])} --> answer>"
        if len(conditions) <= 2:
            body = " && ".join(statement(c) for c in conditions)
            return [f"<{'(' + body + ')' if len(conditions) == 2 else body} ==> {answer}>. {{1.0 0.9}}"]

        order = list(dict.fromkeys(x for c in conditions for x in (c[0], c[2]) if is_var(x)))
        lines, body, seen = [], statement(conditions[0]), {x for x in (conditions[0][0], conditions[0][2]) if is_var(x)}
        for k, c in enumerate(conditions[1:-1], 1):
            seen |= {x for x in (c[0], c[2]) if is_var(x)}
            later = {x for rest in conditions[k + 1:] for x in (rest[0], rest[2]) if is_var(x)} | {rule["answer"]}
            carry = [term(x) for x in order if x in seen & later]
            # ONA product terms inside implication heads work reliably with 1 or 2 variables; if 3 occur,
            # prioritize the variable needed by the immediately next condition and the final answer variable.
            if len(carry) > 2:
                next_vars = {term(x) for x in (conditions[k + 1][0], conditions[k + 1][2]) if is_var(x)}
                ans_term = term(rule["answer"])
                carry = [v for v in carry if v == ans_term or v in next_vars][:2]
            if not carry:
                head = f"<{tag} --> {tag}_s{k}>"
            elif len(carry) == 1:
                head = f"<{carry[0]} --> {tag}_s{k}>"
            else:
                head = f"<({' * '.join(carry)}) --> {tag}_s{k}>"
            lines.append(f"<({body} && {statement(c)}) ==> {head}>. {{1.0 0.9}}")
            body = head
        lines.append(f"<({body} && {statement(conditions[-1])}) ==> {answer}>. {{1.0 0.9}}")
        return lines

    def program(self, triples, rules):
        # ONA's attention is limited: enter the exact facts that participate in valid bindings FIRST
        # and cap unrelated distractors per relation so ONA never drops the gold derivation.
        position = {}
        for rule in rules:
            for i, c in enumerate(rule["conditions"]):
                position[atom(c[1])] = min(position.get(atom(c[1]), i), i)
        used = set(position)
        facts = {(atom(t["subject"]), atom(t["relation"]), atom(t["object"])) for t in triples}
        active_triples = set()
        self.candidates = []
        # Sort rules by specificity (more conditions = more specific constraints first) when collecting candidates
        for rule in sorted(rules, key=lambda r: len(r["conditions"]), reverse=True):
            ans_var = rule["answer"]
            raw_bindings = self._bindings(rule["conditions"], facts, return_used=True)
            for binding, used_facts in raw_bindings:
                active_triples.update(used_facts)
                if binding[ans_var] not in self.candidates:
                    self.candidates.append(binding[ans_var])

        names, lines = {}, ["*volume=0"]
        rel_distractor_count = Counter()
        sorted_triples = sorted(
            triples,
            key=lambda t: (
                0 if (atom(t["subject"]), atom(t["relation"]), atom(t["object"])) in active_triples else 1,
                position.get(atom(t["relation"]), len(position)),
            ),
        )
        for triple in sorted_triples:
            subject, obj, relation = (atom(triple[k]) for k in ("subject", "object", "relation"))
            names.setdefault(subject, triple["subject"])
            names.setdefault(obj, triple["object"])
            if relation not in used:
                continue
            is_active = (subject, relation, obj) in active_triples
            if not is_active:
                if rel_distractor_count[relation] >= 3:
                    continue
                rel_distractor_count[relation] += 1
            line = f"<({subject} * {obj}) --> {relation}>. {{{triple['f']:.3f} {triple['c']:.3f}}}"
            if line not in lines:
                lines.append(line)
        for number, rule in enumerate(rules):
            for line in self._chain(rule, f"r{number}"):
                if line not in lines:
                    lines.append(line)
        lines.extend([str(self.cycles), "<?1 --> answer>?", str(self.cycles)])
        if self.candidates:
            lines.extend(f"<{c} --> answer>?" for c in self.candidates)
            lines.append(str(max(10, self.cycles // 10)))
        return "\n".join(lines) + "\n", names

    def answer(self, output, question):
        """Pick ONA's best-fitting answer for the question this reasoner translated last. The pick prefers answers
        the rules bind over the triples; self.ona_top is ONA's pick without that preference, so a run can show
        how often ONA's own derivations alone are right."""
        inside = ({(atom(t["subject"]), atom(t["object"])) for t in self.last_triples
                   if relation_name(t["relation"]) in INSIDE}
                  | {(atom(t["object"]), atom(t["subject"])) for t in self.last_triples
                     if relation_name(t["relation"]) in ("contains", "has_part", "includes", "consists_of")})
        self.ona_top = pick_answer(output, question, self.names, self.answer_type, inside=inside)
        return pick_answer(output, question, self.names, self.answer_type, self.candidates, inside=inside)

    def _prepare(self, facts, question):
        """Triples for the question (aliases, merged relations, inverses) and its hop plan, shared by every rule set."""
        if not isinstance(facts, list) or not facts:
            raise ValueError("Pass a nonempty list of facts.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Pass the question text so the answer rule can be generated.")
        self.missing_terms, self.auto_repairs, self.ona_top, self.samples = [], Counter(), None, []
        self.relation_groups, self.temperature = [], 0.0
        self.last_raw_triples = self.triples(facts)
        triples = self.canonicalize(self.last_raw_triples)
        if self.merge:
            triples = self.merge_relations(triples)
        self.last_triples = self.add_inverses(triples)
        self.plan_text = self.plan(question) if self.use_plan else ""

    def run(self, facts, question):
        """Translate a fact list and question through DeepInfra; return Narsese text (one rule set, no voting)."""
        self._prepare(facts, question)
        self.last_rules = self.rules(question, self.last_triples, self.plan_text)
        if not self.last_rules:
            raise ValueError("No valid rule generated. Inspect ona.rule_attempts.")
        program, self.names = self.program(self.last_triples, self.last_rules)
        self.last_narsese = program
        return program

    def solve(self, facts, question, ona_path="ONA/NAR"):
        """Facts + question -> answer (or None). Writes up to self.votes rule sets over the same triples (the first at
        temperature 0, the others at 0.7), runs ONA on each, and returns the answer most rule sets agree on; ties go to
        the earliest. Stops early once two rule sets agree. Afterwards last_rules, rule_attempts, auto_repairs,
        candidates, names, last_narsese and last_output describe the winning rule set; self.samples lists them all."""
        self._prepare(facts, question)
        for k in range(max(1, self.votes)):
            self.temperature = 0.0 if k == 0 else 0.7
            self.auto_repairs, self.candidates, self.names = Counter(), [], {}
            sample = {"rules": [], "answer": None, "ona_top": None, "narsese": None, "output": None}
            try:
                sample["rules"] = self.rules(question, self.last_triples, self.plan_text)
            except Exception as error:  # an API failure on a sampled rule set only loses that vote
                if k == 0 and self.votes <= 1:
                    raise
                sample["error"] = error_details(error)
            sample.update(rule_attempts=self.rule_attempts, answer_type=self.answer_type,
                          auto_repairs=self.auto_repairs)
            if sample["rules"]:
                sample["narsese"], self.names = self.program(self.last_triples, sample["rules"])
                sample["output"] = self.execute(sample["narsese"], ona_path)
                sample["answer"] = self.answer(sample["output"], question)
                sample.update(ona_top=self.ona_top, candidates=self.candidates, names=self.names)
            self.samples.append(sample)
            votes = Counter(squash(s["answer"]) for s in self.samples if s["answer"])
            if votes and votes.most_common(1)[0][1] >= 2:
                break
        self.temperature = 0.0
        votes = Counter(squash(s["answer"]) for s in self.samples if s["answer"])
        top = votes.most_common(1)[0][0] if votes else None
        winner = next((s for s in self.samples if s["answer"] and squash(s["answer"]) == top), None) \
            or next((s for s in self.samples if s["rules"]), self.samples[0])
        self.last_rules, self.rule_attempts, self.answer_type = winner["rules"], winner["rule_attempts"], winner["answer_type"]
        self.auto_repairs, self.ona_top = winner["auto_repairs"], winner["ona_top"]
        self.candidates, self.names = winner.get("candidates", []), winner.get("names", {})
        self.last_narsese, self.last_output = winner["narsese"], winner["output"]
        return winner["answer"]

    @staticmethod
    def execute(narsese, ona_path="ONA/NAR"):
        """Run the returned Narsese locally; make no API requests or file writes."""
        binary = Path(ona_path).expanduser().resolve()
        if binary.is_dir():
            binary = binary / "NAR"
        # start_new_session: Ctrl-C in the terminal stops the Python run, not the ONA processes it is waiting on.
        result = subprocess.run(
            [str(binary), "shell"], input=narsese,
            capture_output=True, text=True, timeout=120, start_new_session=True,
        )
        if result.returncode:
            # ONA prints its own errors to stdout, so show the tail of it.
            raise RuntimeError(f"ONA exited with code {result.returncode}:\n"
                               f"{result.stderr}{result.stdout[-400:]}")
        return result.stdout



