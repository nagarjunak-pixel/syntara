from __future__ import annotations

import re

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from syntara.regex import (
    ALPHABET_SIZE,
    RegexError,
    compile_regex,
    escape_literal,
    parse_regex,
    prune_dead_states,
)

DIFF_PATTERNS = [
    r"a*b",
    r"(a|b)*abb",
    r"[a-c]+x?",
    r"(?:ab|a)(?:bc|c)",
    r"a{2,4}",
    r"a{3}",
    r"(ab){2,}",
    r"[^a]*",
    r"\d+(\.\d+)?",
    r"\w+ \w+",
    r"x|",
    r"(a|)b",
    r".a.",
    r"[0-9a-f]{2}-?[0-9]",
    r"(a*)*b",
    r"((a|b)c)+|d?",
    r"-?(?:0|[1-9][0-9]*)",
    r"a\.b\*c",
    r"[\d_]+",
    r"\s*\S\s*",
]
ALPHABET = "abcdx01 .-_*\n9f"


@pytest.mark.parametrize("pattern", DIFF_PATTERNS)
@given(text=st.text(alphabet=ALPHABET, max_size=8))
@settings(max_examples=120, deadline=None)
def test_dfa_agrees_with_python_re(pattern: str, text: str) -> None:
    dfa = compile_regex(pattern)
    assert dfa.matches(text) == (re.fullmatch(pattern, text) is not None)


def test_minimal_dfa_for_textbook_example() -> None:
    # The minimal DFA for (a|b)*abb has exactly 4 states (Aho, Sethi & Ullman, Fig. 3.36).
    dfa = compile_regex("(a|b)*abb")
    assert dfa.num_states == 4
    assert len(dfa.accepting) == 1
    assert dfa.start == 0


def test_equivalent_patterns_minimise_to_identical_tables() -> None:
    a = compile_regex("(a|b)*")
    b = compile_regex("(a*b*)*")
    assert a.table == b.table and a.accepting == b.accepting


def test_utf8_literals_and_raw_bytes() -> None:
    dfa = compile_regex("é+")
    assert dfa.matches("éé")
    assert not dfa.matches("e")
    assert compile_regex(r"\u00e9").matches("é")
    raw = compile_regex(r"\xff\x41")
    assert raw.matches(b"\xffA")
    assert compile_regex(r"[\x80-\xff]").matches(b"\x90")
    assert compile_regex(r"\n\t\r\f\v\0").matches(b"\n\t\r\f\v\x00")


def test_dot_excludes_newline_and_negated_class_covers_high_bytes() -> None:
    dot = compile_regex(".")
    assert not dot.matches("\n")
    assert dot.matches(b"\xc3")
    assert compile_regex("[^a]").matches(b"\xfe")


@given(st.text(max_size=12))
@settings(max_examples=150, deadline=None)
def test_escape_literal_roundtrip(text: str) -> None:
    dfa = compile_regex(escape_literal(text))
    assert dfa.matches(text)
    assert not dfa.matches(text + "z")


def test_dfa_helpers() -> None:
    dfa = compile_regex("ab")
    arr = dfa.to_array()
    assert arr.shape == (dfa.num_states, ALPHABET_SIZE)
    assert arr.dtype == np.int32
    assert dfa.step(-1, ord("a")) == -1
    assert dfa.walk(0, b"zz") == -1
    assert dfa.num_transitions == 2
    end = dfa.walk(0, b"ab")
    assert dfa.is_accepting(end)
    assert not dfa.is_accepting(0)


def test_parse_regex_structure() -> None:
    node = parse_regex("a|b")
    assert type(node).__name__ == "Alt"
    assert parse_regex("") == parse_regex("(?:)")


@pytest.mark.parametrize(
    "pattern",
    [
        "(",
        "(a",
        "a)",
        "*a",
        "a{3,2}",
        "a{1001}",
        "a{,3}",
        "a{2",
        "a{2x",
        "[",
        "[abc",
        r"[^\x00-\xff]",
        r"\q",
        r"\x4",
        r"\u12",
        "^a",
        "a$",
        "[é]",
        r"[\u00e9]",
        "[z-a]",
        r"[a-\d]",
        "]",
        "}",
    ],
)
def test_malformed_patterns_raise(pattern: str) -> None:
    with pytest.raises(RegexError):
        compile_regex(pattern)


def test_unbounded_and_open_ranges() -> None:
    assert compile_regex("a{2,}").matches("aaaaa")
    assert not compile_regex("a{2,}").matches("a")
    assert compile_regex("[a-]").matches("-")
    assert compile_regex("[]a]").matches("]")
    assert compile_regex(r"[\]]").matches("]")


def test_prune_rejects_empty_language() -> None:
    table = [[-1] * ALPHABET_SIZE]
    with pytest.raises(RegexError, match="no string"):
        prune_dead_states(table, set())


def test_prune_removes_dead_and_unreachable_states() -> None:
    table = [[-1] * ALPHABET_SIZE for _ in range(4)]
    table[0][ord("a")] = 1  # 1 accepting
    table[0][ord("b")] = 2  # 2 is a dead end
    table[3][ord("a")] = 1  # 3 unreachable
    new_table, acc = prune_dead_states(table, {1})
    assert len(new_table) == 2
    assert acc == {1}
    assert new_table[0][ord("b")] == -1
