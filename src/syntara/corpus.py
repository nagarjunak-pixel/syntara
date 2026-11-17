"""Deterministic demo corpus, schema and pre-trained tokenizer/LM bundle."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from syntara.lm import NGramLM
from syntara.tokenizer import BPETokenizer

PERSON_SCHEMA: dict[str, Any] = {
    "title": "Employee",
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 24},
        "age": {"type": "integer"},
        "role": {"enum": ["engineer", "manager", "analyst", "scientist"]},
        "email": {"type": "string", "format": "email"},
        "skills": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 16},
            "maxItems": 3,
        },
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "role", "active"],
}

_FIRST = ["Asha", "Ravi", "Lena", "Omar", "Mei", "Tomas", "Priya", "Noor", "Kofi", "Sara"]
_LAST = ["Rao", "Khan", "Silva", "Ng", "Haddad", "Ivanova", "Mensah", "Fischer"]
_ROLES = ["engineer", "manager", "analyst", "scientist", "intern", "director"]
_SKILLS = ["python", "c++", "sql", "rust", "pytorch", "kubernetes", "go", "spark"]
_DOMAINS = ["example.com", "corp.ae", "lab.io"]


def _record(rng: random.Random) -> dict[str, Any]:
    first, last = rng.choice(_FIRST), rng.choice(_LAST)
    rec: dict[str, Any] = {
        "name": f"{first} {last}",
        "age": rng.randint(21, 64),
        "role": rng.choice(_ROLES),
    }
    if rng.random() < 0.6:
        rec["email"] = f"{first.lower()}.{last.lower()}@{rng.choice(_DOMAINS)}"
    if rng.random() < 0.7:
        rec["skills"] = rng.sample(_SKILLS, rng.randint(1, 4))
    rec["active"] = rng.random() < 0.8
    if rng.random() < 0.25:
        rec["id"] = rng.randint(1000, 9999)  # schema-violating extra key
    if rng.random() < 0.2:
        keys = list(rec)
        rng.shuffle(keys)
        rec = {k: rec[k] for k in keys}
    return rec


def build_corpus(size: int = 400, seed: int = 7) -> list[str]:
    """Mixture of JSON records (some schema-violating) and free-text descriptions.

    Real chat models also mix prose and almost-JSON; constrained decoding is what turns
    that mixture into guaranteed-valid output.
    """
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(size):
        rec = _record(rng)
        roll = rng.random()
        if roll < 0.55:
            lines.append(json.dumps(rec, separators=(",", ":")))
        elif roll < 0.75:
            lines.append(json.dumps(rec))
        else:
            lines.append(
                f"{rec['name']} is a {rec['age']} year old {rec['role']}"
                + (f" who knows {', '.join(rec['skills'])}." if "skills" in rec else ".")
            )
    return lines


@dataclass(frozen=True, slots=True)
class DemoBundle:
    tokenizer: BPETokenizer
    lm: NGramLM
    corpus: tuple[str, ...]


@lru_cache(maxsize=4)
def demo_bundle(vocab_size: int = 512, order: int = 4, corpus_size: int = 400, seed: int = 7) -> DemoBundle:
    """Train (and cache) the demo BPE tokenizer and n-gram LM."""
    corpus = build_corpus(corpus_size, seed)
    tok = BPETokenizer.train(corpus, vocab_size)
    lm = NGramLM(tok.vocab_size, order=order, bos_id=tok.eos_id)
    lm.fit(tok.encode(line) + [tok.eos_id] for line in corpus)
    return DemoBundle(tokenizer=tok, lm=lm, corpus=tuple(corpus))
