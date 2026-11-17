"""Reproducible benchmarks: native vs Python index build, and validity with/without constraints."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from syntara.corpus import PERSON_SCHEMA, demo_bundle
from syntara.decode import generate
from syntara.earley import JSON_GRAMMAR, GrammarConstraint
from syntara.index import RegexConstraint
from syntara.kernels import KernelBridge, vocab_layout
from syntara.regex import compile_regex
from syntara.schema import schema_to_regex, validate_text
from syntara.tokenizer import Vocabulary


@dataclass(frozen=True, slots=True)
class IndexBenchmark:
    regex_chars: int
    dfa_states: int
    dfa_transitions: int
    vocab_size: int
    allowed_pairs: int
    native_backend: str
    native_seconds: float
    python_seconds: float
    speedup: float
    bit_exact: bool


@dataclass(frozen=True, slots=True)
class ValidityBenchmark:
    samples: int
    temperature: float
    unconstrained_valid: int
    schema_constrained_valid: int
    grammar_constrained_json: int
    grammar_truncated: int
    mean_allowed_fraction: float
    unconstrained_example: str
    constrained_example: str


def substring_vocabulary(corpus: Sequence[str], size: int, max_len: int = 8) -> Vocabulary:
    """A large, realistic-shaped vocabulary: all 256 bytes plus the most frequent substrings.

    Used to stress the index kernel at production vocabulary scale (16k-32k tokens) without
    downloading a tokenizer.
    """
    counts: Counter[bytes] = Counter()
    for line in corpus:
        data = line.encode("utf-8")
        for i in range(len(data)):
            for j in range(i + 2, min(len(data), i + max_len) + 1):
                counts[data[i:j]] += 1
    singles = [bytes((b,)) for b in range(256)]
    ranked = sorted(counts, key=lambda t: (-counts[t], t))[: max(size - 257, 0)]
    return Vocabulary(tokens=(*singles, *ranked, b""), eos_id=256 + len(ranked))


def benchmark_index(vocab: Vocabulary | None = None, repeats: int = 3) -> IndexBenchmark:
    """Time the index kernel alone (vocabulary layout and DFA table prepared up-front)."""
    vocab = vocab or demo_bundle().tokenizer.vocab
    pattern = schema_to_regex(PERSON_SCHEMA)
    dfa = compile_regex(pattern)
    table = dfa.to_array()
    accepting = np.zeros(dfa.num_states, dtype=np.uint8)
    accepting[list(dfa.accepting)] = 1
    layout = vocab_layout(vocab)
    native = KernelBridge()
    python = KernelBridge(use_native=False)

    def best(bridge: KernelBridge, reps: int) -> tuple[float, npt.NDArray[np.int32]]:
        times: list[float] = []
        out = bridge.build_token_index(table, accepting, layout)
        for _ in range(reps):
            t0 = time.perf_counter()
            out = bridge.build_token_index(table, accepting, layout)
            times.append(time.perf_counter() - t0)
        return min(times), out

    nat_s, nat_idx = best(native, repeats)
    py_s, py_idx = best(python, 1)
    return IndexBenchmark(
        regex_chars=len(pattern),
        dfa_states=dfa.num_states,
        dfa_transitions=dfa.num_transitions,
        vocab_size=len(vocab),
        allowed_pairs=int((nat_idx >= 0).sum()),
        native_backend=native.backend,
        native_seconds=nat_s,
        python_seconds=py_s,
        speedup=py_s / nat_s if nat_s > 0 else float("inf"),
        bit_exact=bool(np.array_equal(nat_idx, py_idx)),
    )


def benchmark_validity(
    samples: int = 20, temperature: float = 0.8, max_tokens: int = 96
) -> ValidityBenchmark:
    bundle = demo_bundle()
    tok, lm = bundle.tokenizer, bundle.lm
    vocab = tok.vocab
    schema_c = RegexConstraint.from_schema(PERSON_SCHEMA, vocab)
    grammar_c = GrammarConstraint(JSON_GRAMMAR, vocab)
    unc_ok = sch_ok = gram_ok = truncated = 0
    fractions: list[float] = []
    unc_example = con_example = ""
    for seed in range(samples):
        unc = generate(lm, vocab, None, max_tokens=max_tokens, temperature=temperature, seed=seed)
        con = generate(lm, vocab, schema_c, max_tokens=max_tokens, temperature=temperature, seed=seed)
        gram = generate(lm, vocab, grammar_c, max_tokens=max_tokens, temperature=temperature, seed=seed)
        unc_ok += validate_text(unc.text, PERSON_SCHEMA)
        sch_ok += con.finished and validate_text(con.text, PERSON_SCHEMA)
        gram_ok += gram.finished and _is_json(gram.text)
        truncated += not gram.finished
        fractions.append(con.mean_allowed_fraction)
        if seed == 0:
            unc_example, con_example = unc.text, con.text
    return ValidityBenchmark(
        samples=samples,
        temperature=temperature,
        unconstrained_valid=unc_ok,
        schema_constrained_valid=sch_ok,
        grammar_constrained_json=gram_ok,
        grammar_truncated=truncated,
        mean_allowed_fraction=float(np.mean(fractions)),
        unconstrained_example=unc_example,
        constrained_example=con_example,
    )


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def run_all(samples: int = 20, large_vocab: int = 16384) -> dict[str, Any]:
    bundle = demo_bundle()
    return {
        "index": asdict(benchmark_index(bundle.tokenizer.vocab)),
        "index_large_vocab": asdict(
            benchmark_index(substring_vocabulary(bundle.corpus, large_vocab), repeats=3)
        ),
        "validity": asdict(benchmark_validity(samples=samples)),
    }
