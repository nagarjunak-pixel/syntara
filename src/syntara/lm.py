"""Interpolated n-gram language model over token ids (deterministic, dependency-free).

The model is deliberately small and fully reproducible so that every benchmark and test runs
on CPU in seconds without downloading weights. What it demonstrates is model-independent:
masking happens on the logits, so the same constraint objects drive any LLM whose
vocabulary is supplied as a :class:`~syntara.tokenizer.Vocabulary`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

F64Array = npt.NDArray[np.float64]


class NGramLM:
    """Jelinek-Mercer interpolated n-gram model with a uniform floor.

    ``P(t | h) = sum_k w_k * P_ML(t | h[-k:]) / sum_k w_k`` over the context orders ``k`` whose
    history has been observed, plus a uniform component, so every token has non-zero mass.
    """

    def __init__(self, vocab_size: int, order: int = 4, bos_id: int = 0) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.vocab_size = vocab_size
        self.order = order
        self.bos_id = bos_id
        self.counts: list[dict[tuple[int, ...], Counter[int]]] = [{} for _ in range(order)]
        # Higher orders get geometrically larger weights; index 0 = unigram.
        self.weights = [2.0**k for k in range(order)]
        self.uniform_weight = 0.01

    def fit(self, sequences: Iterable[Sequence[int]]) -> NGramLM:
        for seq in sequences:
            padded = [self.bos_id] * (self.order - 1) + list(seq)
            for i in range(self.order - 1, len(padded)):
                tok = padded[i]
                for k in range(self.order):
                    ctx = tuple(padded[i - k : i])
                    self.counts[k].setdefault(ctx, Counter())[tok] += 1
        return self

    def log_probs(self, context: Sequence[int]) -> F64Array:
        padded = [self.bos_id] * (self.order - 1) + list(context)
        probs = np.full(self.vocab_size, self.uniform_weight / self.vocab_size, dtype=np.float64)
        total_w = self.uniform_weight
        for k in range(self.order):
            ctx = tuple(padded[len(padded) - k :]) if k else ()
            counter = self.counts[k].get(ctx)
            if not counter:
                continue
            denom = float(sum(counter.values()))
            w = self.weights[k]
            ids = np.fromiter(counter.keys(), dtype=np.int64)
            vals = np.fromiter(counter.values(), dtype=np.float64)
            probs[ids] += w * vals / denom
            total_w += w
        result: F64Array = np.log(probs / total_w)
        return result

    def sequence_log_prob(self, context: Sequence[int], continuation: Sequence[int]) -> float:
        ctx = list(context)
        total = 0.0
        for tok in continuation:
            total += float(self.log_probs(ctx)[tok])
            ctx.append(tok)
        return total
