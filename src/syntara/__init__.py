"""Syntara: grammar-constrained structured decoding engine.

JSON Schema / regex -> minimal byte DFA -> native C++20 token index, plus a byte-level
Earley parser for context-free grammars and exact closed-set choice scoring.
"""

from syntara.decode import ChoiceScore, Generation, generate, score_choices
from syntara.earley import EarleyParser, GrammarConstraint, parse_grammar
from syntara.index import RegexConstraint, TokenIndex
from syntara.regex import Dfa, compile_regex
from syntara.schema import schema_to_regex, validate, validate_text
from syntara.tokenizer import BPETokenizer, Vocabulary

__version__ = "1.0.0"

__all__ = [
    "BPETokenizer",
    "ChoiceScore",
    "Dfa",
    "EarleyParser",
    "Generation",
    "GrammarConstraint",
    "RegexConstraint",
    "TokenIndex",
    "Vocabulary",
    "__version__",
    "compile_regex",
    "generate",
    "parse_grammar",
    "schema_to_regex",
    "score_choices",
    "validate",
    "validate_text",
]
