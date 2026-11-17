"""Token-level constraint index over a byte DFA (regex / JSON-Schema constraints)."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from syntara.kernels import KernelBridge, get_bridge, vocab_layout
from syntara.regex import Dfa, compile_regex
from syntara.schema import schema_to_regex
from syntara.tokenizer import Vocabulary

BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class IndexStats:
    num_states: int
    vocab_size: int
    allowed_pairs: int
    mean_allowed_per_state: float
    build_seconds: float
    backend: str


class TokenIndex:
    """``next[s, t]`` = DFA state after emitting token ``t`` in state ``s`` (``-1`` = masked).

    Because the DFA is pruned to live states, any token the mask allows keeps the output
    completable: constrained decoding can never reach a dead end.
    """

    def __init__(self, dfa: Dfa, vocab: Vocabulary, bridge: KernelBridge | None = None) -> None:
        self.dfa = dfa
        self.vocab = vocab
        self.bridge = bridge or get_bridge()
        accepting = np.zeros(dfa.num_states, dtype=np.uint8)
        for a in dfa.accepting:
            accepting[a] = 1
        layout = vocab_layout(vocab)
        table = dfa.to_array()
        start = time.perf_counter()
        self.next: npt.NDArray[np.int32] = self.bridge.build_token_index(table, accepting, layout)
        elapsed = time.perf_counter() - start
        counts = self.bridge.count_allowed(self.next)
        self.stats = IndexStats(
            num_states=dfa.num_states,
            vocab_size=len(vocab),
            allowed_pairs=int(counts.sum()),
            mean_allowed_per_state=float(counts.mean()),
            build_seconds=elapsed,
            backend=self.bridge.backend,
        )

    def allowed(self, state: int) -> BoolArray:
        mask: BoolArray = self.next[state] >= 0
        return mask


class RegexConstraint:
    """Constraint whose state is a DFA state id."""

    def __init__(self, index: TokenIndex) -> None:
        self.index = index

    @classmethod
    def from_regex(
        cls, pattern: str, vocab: Vocabulary, bridge: KernelBridge | None = None
    ) -> RegexConstraint:
        return cls(TokenIndex(compile_regex(pattern), vocab, bridge))

    @classmethod
    def from_schema(
        cls,
        schema: Mapping[str, Any],
        vocab: Vocabulary,
        whitespace: str = "",
        bridge: KernelBridge | None = None,
    ) -> RegexConstraint:
        return cls.from_regex(schema_to_regex(schema, whitespace), vocab, bridge)

    @classmethod
    def from_choices(
        cls, choices: list[str], vocab: Vocabulary, bridge: KernelBridge | None = None
    ) -> RegexConstraint:
        """Closed-set decision constraint: the output must be exactly one of ``choices``."""
        from syntara.regex import escape_literal

        if not choices:
            raise ValueError("choices must be non-empty")
        return cls.from_regex("(?:" + "|".join(escape_literal(c) for c in choices) + ")", vocab, bridge)

    def initial(self) -> int:
        return self.index.dfa.start

    def allowed(self, state: int) -> BoolArray:
        return self.index.allowed(state)

    def next_row(self, state: int) -> npt.NDArray[np.int32]:
        row: npt.NDArray[np.int32] = self.index.next[state]
        return row

    def advance(self, state: int, token: int) -> int:
        nxt = int(self.index.next[state, token])
        if nxt < 0:
            raise ValueError(f"token {token} is not allowed in state {state}")
        return nxt

    def is_complete(self, state: int) -> bool:
        return self.index.dfa.is_accepting(state)
