from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from syntara.earley import (
    ARITHMETIC_GRAMMAR,
    JSON_GRAMMAR,
    EarleyParser,
    GrammarConstraint,
    GrammarError,
    parse_grammar,
)
from syntara.tokenizer import Vocabulary

JSON_PARSER = EarleyParser(parse_grammar(JSON_GRAMMAR))

json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-(10**6), 10**6) | st.text(max_size=6),
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=4), children, max_size=3)
    ),
    max_leaves=8,
)


@given(value=json_values, indent=st.sampled_from([None, 2]))
@settings(max_examples=80, deadline=None)
def test_earley_accepts_serialised_json(value: object, indent: int | None) -> None:
    assert JSON_PARSER.recognize(json.dumps(value, indent=indent, ensure_ascii=False))


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


@given(text=st.text(alphabet='{}[]",:0123456789.-e tfrulsa\\', max_size=10))
@settings(max_examples=300, deadline=None)
def test_earley_agrees_with_json_loads_on_fuzz(text: str) -> None:
    assert JSON_PARSER.recognize(text) == _is_json(text)


def test_left_recursive_arithmetic() -> None:
    parser = EarleyParser(parse_grammar(ARITHMETIC_GRAMMAR))
    for good in ["1", "1+2", "1 + 2 * (3 - 4) / 5", "((7))", "10*20-3"]:
        assert parser.recognize(good), good
    for bad in ["", "1+", "(1", "1++2", "+1", "1 +  2"]:
        assert not parser.recognize(bad), bad


def test_grammar_syntax_features() -> None:
    g = parse_grammar(
        r"""
        # comment line
        root ::= greet ( " " name )+ "!"? tail   # trailing comment
        greet ::= "hi" | "hey"
               | "yo"
        name ::= [A-Z] [a-z]*
        tail ::= ("ab")* "\n"? "\x41"? "\u00e9"? . ?
        """
    )
    p = EarleyParser(g)
    assert p.recognize("hi Ann")
    assert p.recognize("yo Bo Cy!ababé")
    assert p.recognize(b"hey Al\nA\xc3\xa9")
    assert p.recognize("hi Ann!z")
    assert not p.recognize("hi ann")
    assert not p.recognize("hello Ann")
    assert g.num_rules > 5


def test_unproductive_and_unreachable_rules_are_pruned() -> None:
    g = parse_grammar('root ::= "a" | bad\nbad ::= bad "x"\nunused ::= "u"')
    p = EarleyParser(g)
    assert p.recognize("a")
    assert not p.recognize("x")
    names = {g.names[prod.lhs] for prod in g.productions}
    assert names == {"root"}


def test_custom_start_symbol() -> None:
    g = parse_grammar('a ::= "x" b\nb ::= "y"', start="a")
    assert EarleyParser(g).recognize("xy")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('::= "a"', "rule name"),
        ('root "a"', "::="),
        ('root ::= "a"\nroot ::= "b"', "twice"),
        ("root ::= missing", "undefined"),
        ('other ::= "a"', "start rule"),
        ('root ::= root "a"', "derives no"),
        ('root ::= ( "a"', "parenthesis"),
        ('root ::= "abc', "unterminated"),
        ("root ::= [z-a]", "range"),
        (r'root ::= "\q"', "escape"),
        (r'root ::= "\d"', "class"),
        ("root ::= @", "unexpected"),
    ],
)
def test_grammar_errors(text: str, message: str) -> None:
    with pytest.raises(GrammarError, match=message):
        parse_grammar(text)


def test_grammar_constraint_masks_and_transitions() -> None:
    vocab = Vocabulary(tokens=(b"(", b")", b"()", b"x", b""), eos_id=4)
    gc = GrammarConstraint('root ::= "(" root ")" | "x"', vocab)
    s0 = gc.initial()
    assert gc.allowed(s0).tolist() == [True, False, False, True, False]
    assert gc.allowed(s0).tolist() == [True, False, False, True, False]  # cached path
    s1 = gc.advance(s0, 0)
    s2 = gc.advance(s1, 3)
    assert not gc.is_complete(s2)
    assert gc.allowed(s2).tolist() == [False, True, False, False, False]
    s3 = gc.advance(s2, 1)
    assert gc.is_complete(s3)
    assert gc.allowed(s3)[vocab.eos_id]
    assert gc.advance(s3, vocab.eos_id) is s3
    with pytest.raises(ValueError, match="EOS"):
        gc.advance(s0, vocab.eos_id)
    with pytest.raises(ValueError):
        gc.advance(s0, 1)


def test_grammar_constraint_multi_byte_tokens_cross_terminals() -> None:
    tokens = [b'{"', b'":', b"a", b"1", b"}", b"{", b'"', b":", b'"a', b",", b""]
    vocab = Vocabulary(tokens=tuple(tokens), eos_id=len(tokens) - 1)
    gc = GrammarConstraint(JSON_GRAMMAR, vocab)
    state = gc.initial()
    for tok in [0, 2, 1, 3, 4]:  # {"  a  ":  1  }
        assert gc.allowed(state)[tok]
        state = gc.advance(state, tok)
    assert gc.is_complete(state)
    assert np.flatnonzero(gc.allowed(state)).tolist() == [vocab.eos_id]
