"""Syntara: grammar-constrained structured decoding & cost-matrix decision engine.

JSON Schema / regex -> minimal byte DFA -> native C++20 token index, plus a byte-level
Earley parser for context-free grammars, exact closed-set choice scoring, and
amount-scaled institutional cost-matrix decision optimization.
"""

from syntara.cost_matrix import (
    CostDecisionResult,
    CostMatrixDecisionHead,
    LedgerStreamSummary,
    expected_calibration_error,
    kl_divergence_nats,
)
from syntara.decode import ChoiceScore, Generation, generate, score_choices
from syntara.earley import EarleyParser, GrammarConstraint, parse_grammar
from syntara.index import RegexConstraint, TokenIndex
from syntara.regex import Dfa, compile_regex
from syntara.schema import schema_to_regex, validate, validate_text
from syntara.tokenizer import BPETokenizer, Vocabulary

__version__ = "1.1.0"

__all__ = [
    "BPETokenizer",
    "ChoiceScore",
    "CostDecisionResult",
    "CostMatrixDecisionHead",
    "Dfa",
    "EarleyParser",
    "Generation",
    "GrammarConstraint",
    "LedgerStreamSummary",
    "RegexConstraint",
    "TokenIndex",
    "Vocabulary",
    "__version__",
    "compile_regex",
    "expected_calibration_error",
    "generate",
    "kl_divergence_nats",
    "parse_grammar",
    "schema_to_regex",
    "score_choices",
    "validate",
    "validate_text",
]
