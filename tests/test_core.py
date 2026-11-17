from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from syntara.corpus import PERSON_SCHEMA, build_corpus, demo_bundle
from syntara.index import RegexConstraint, TokenIndex
from syntara.kernels import (
    DEFAULT_SOURCE,
    KernelBridge,
    get_bridge,
    py_masked_argmax,
    sorted_order_and_lcp,
    vocab_layout,
)
from syntara.lm import NGramLM
from syntara.regex import Dfa, compile_regex
from syntara.schema import schema_to_regex
from syntara.tokenizer import BPETokenizer, Vocabulary

NATIVE = KernelBridge(use_native=True)
PYTHON = KernelBridge(use_native=False)


def _naive_index(dfa: Dfa, vocab: Vocabulary) -> np.ndarray:
    out = np.full((dfa.num_states, len(vocab)), -1, dtype=np.int32)
    for s in range(dfa.num_states):
        for t, tok in enumerate(vocab.tokens):
            if t == vocab.eos_id:
                out[s, t] = s if dfa.is_accepting(s) else -1
            elif tok:
                out[s, t] = dfa.walk(s, tok)
    return out


# ---------------------------------------------------------------------------- tokenizer


def test_bpe_roundtrip_and_multibyte_tokens() -> None:
    tok = demo_bundle().tokenizer
    assert tok.vocab_size == 512
    assert any(len(t) > 3 for t in tok.vocab.tokens)
    for text in ['{"name":"Asha Rao","age":30}', "naïve café ✓", ""]:
        assert tok.decode(tok.encode(text)) == text
    assert tok.vocab.tokens[tok.eos_id] == b""


def test_bpe_training_edge_cases() -> None:
    with pytest.raises(ValueError):
        BPETokenizer.train(["abc"], 256)
    single = BPETokenizer.train(["a", "b"], 300)  # no adjacent pairs at all
    assert single.merges == ()
    rare = BPETokenizer.train(["abcdef"], 300)  # every pair occurs once -> stop
    assert rare.merges == ()
    tok = BPETokenizer.train(["abab abab"] * 3, 260)
    assert len(tok.merges) == 3
    assert tok.encode("abab") != list(b"abab")


def test_vocabulary_helpers() -> None:
    v = Vocabulary.from_strings(["a", "bc", "<|eos|>"])
    assert v.tokens == (b"a", b"bc", b"")
    assert v.eos_id == 2 and len(v) == 3
    assert v.decode([0, 1, 2]) == "abc"
    with pytest.raises(ValueError):
        Vocabulary(tokens=(b"a",), eos_id=3)


# ------------------------------------------------------------------------------ kernels


def test_native_kernel_compiles() -> None:
    assert NATIVE.is_native, "clang++/g++ must be available to build the C++20 kernels"
    assert NATIVE.backend.startswith("c++20-native")
    assert not PYTHON.is_native
    assert get_bridge() is get_bridge()


@pytest.mark.parametrize(
    "pattern",
    [r"(a|b)*abb", r"[0-9]{1,3}(\.[0-9]+)?", r'"[a-z ]*"', r"\{(\"k\":[0-9]+,?)*\}", "x"],
)
def test_index_bit_exact_native_python_naive(pattern: str) -> None:
    dfa = compile_regex(pattern)
    vocab = Vocabulary(
        tokens=(
            *[bytes((b,)) for b in range(256)],
            b"ab",
            b"abb",
            b"ba",
            b'"k"',
            b'":',
            b"12",
            b"1.",
            b"",
            b"zz",
        ),
        eos_id=263,
    )
    expected = _naive_index(dfa, vocab)
    assert np.array_equal(TokenIndex(dfa, vocab, NATIVE).next, expected)
    assert np.array_equal(TokenIndex(dfa, vocab, PYTHON).next, expected)


def test_index_on_bpe_vocab_with_schema() -> None:
    vocab = demo_bundle().tokenizer.vocab
    dfa = compile_regex(schema_to_regex(PERSON_SCHEMA))
    native = TokenIndex(dfa, vocab, NATIVE)
    assert np.array_equal(native.next, _naive_index(dfa, vocab))
    assert native.stats.num_states == dfa.num_states
    assert native.stats.allowed_pairs == int((native.next >= 0).sum())
    assert 0 < native.stats.mean_allowed_per_state < len(vocab)
    # Every allowed transition lands on a live state: the decoder can always finish.
    assert native.next.max() < dfa.num_states


@given(
    tokens=st.lists(st.binary(min_size=0, max_size=4), min_size=1, max_size=40),
    pattern=st.sampled_from([r"[ab]*c", r"(ab|ba)+", r"a?b?c?", r"[^c]{0,3}"]),
)
@settings(max_examples=60, deadline=None)
def test_index_property_random_vocabularies(tokens: list[bytes], pattern: str) -> None:
    vocab = Vocabulary(tokens=(*tokens, b""), eos_id=len(tokens))
    dfa = compile_regex(pattern)
    expected = _naive_index(dfa, vocab)
    assert np.array_equal(TokenIndex(dfa, vocab, NATIVE).next, expected)
    assert np.array_equal(TokenIndex(dfa, vocab, PYTHON).next, expected)


def test_sorted_order_and_lcp() -> None:
    order, lcp = sorted_order_and_lcp((b"abc", b"ab", b"b", b""))
    assert order.tolist() == [3, 1, 0, 2]
    assert lcp.tolist() == [0, 0, 2, 0]
    v = Vocabulary(tokens=(b"x", b""), eos_id=1)
    assert vocab_layout(v) is vocab_layout(v)


@pytest.mark.parametrize("bridge", [NATIVE, PYTHON], ids=["native", "python"])
def test_masked_argmax_and_counts(bridge: KernelBridge) -> None:
    logits = np.array([0.5, 3.0, 2.0, 3.0], dtype=np.float64)
    assert bridge.masked_argmax(logits, np.array([0, 0, 0, 0], dtype=np.int32)) == 1  # tie -> lowest
    assert bridge.masked_argmax(logits, np.array([0, -1, 1, -1], dtype=np.int32)) == 2
    assert bridge.masked_argmax(logits, np.array([-1, -1, -1, -1], dtype=np.int32)) == -1
    idx = np.array([[0, -1, 2], [-1, -1, -1]], dtype=np.int32)
    assert bridge.count_allowed(idx).tolist() == [2, 0]


def test_masked_argmax_handles_negative_infinity() -> None:
    logits = np.array([-np.inf, -np.inf], dtype=np.float64)
    nr = np.array([-1, 0], dtype=np.int32)
    assert NATIVE.masked_argmax(logits, nr) == 1
    assert py_masked_argmax(logits, nr) == 1


def test_bridge_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not KernelBridge(source=tmp_path / "missing.cpp").is_native
    broken = tmp_path / "csrc" / "broken.cpp"
    broken.parent.mkdir()
    broken.write_text("this is not C++")
    assert not KernelBridge(source=broken).is_native
    monkeypatch.setenv("SYNTARA_DISABLE_NATIVE", "1")
    assert not KernelBridge().is_native
    monkeypatch.setenv("SYNTARA_DISABLE_NATIVE", "0")
    monkeypatch.setenv("PATH", "")
    assert not KernelBridge().is_native
    assert DEFAULT_SOURCE.exists()
    assert os.environ["PATH"] == ""


def test_regex_constraint_api() -> None:
    vocab = Vocabulary(tokens=(b"a", b"b", b"ab", b""), eos_id=3)
    rc = RegexConstraint.from_regex("ab", vocab, NATIVE)
    s0 = rc.initial()
    assert rc.allowed(s0).tolist() == [True, False, True, False]
    s1 = rc.advance(s0, 0)
    assert not rc.is_complete(s1)
    s2 = rc.advance(s1, 1)
    assert rc.is_complete(s2)
    assert rc.allowed(s2).tolist() == [False, False, False, True]
    with pytest.raises(ValueError):
        rc.advance(s0, 1)
    with pytest.raises(ValueError):
        RegexConstraint.from_choices([], vocab)


# ----------------------------------------------------------------------------------- LM


def test_ngram_lm_is_normalised_and_consistent() -> None:
    lm = NGramLM(vocab_size=5, order=3, bos_id=4).fit([[0, 1, 2], [0, 1, 3]])
    for ctx in ([], [0], [0, 1], [3, 3, 3]):
        assert np.isclose(np.exp(lm.log_probs(ctx)).sum(), 1.0)
    assert lm.log_probs([0])[1] > lm.log_probs([0])[2]
    total = lm.sequence_log_prob([0], [1, 2])
    assert np.isclose(total, lm.log_probs([0])[1] + lm.log_probs([0, 1])[2])
    with pytest.raises(ValueError):
        NGramLM(vocab_size=3, order=0)


def test_corpus_is_deterministic_and_mixed() -> None:
    a, b = build_corpus(50, seed=3), build_corpus(50, seed=3)
    assert a == b
    assert any(line.startswith("{") for line in a)
    assert any(" year old " in line for line in a)
