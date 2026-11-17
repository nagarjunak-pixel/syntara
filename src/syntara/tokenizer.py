"""Byte-level BPE tokenizer and the tokenizer-agnostic :class:`Vocabulary` used by the index.

Real LLM vocabularies contain multi-byte tokens such as ``'":"'`` or ``'ame'`` that straddle
the boundaries of grammar terminals. Token-level constraint indexes exist precisely to handle
that mismatch, so the demo tokenizer here is a genuine byte-level BPE (Sennrich et al., 2016;
GPT-2 style byte alphabet) trained on a corpus, not a character tokenizer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

EOS_TEXT = "<|eos|>"


@dataclass(frozen=True, slots=True, eq=False)
class Vocabulary:
    """Ordered token byte strings plus the id of the end-of-sequence token.

    The EOS token carries no bytes; it is allowed exactly when the constraint accepts.
    Equality and hashing are by identity so per-vocabulary preprocessing can be cached.
    """

    tokens: tuple[bytes, ...]
    eos_id: int

    def __post_init__(self) -> None:
        if not 0 <= self.eos_id < len(self.tokens):
            raise ValueError("eos_id out of range")

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_strings(cls, tokens: Iterable[str], eos: str = EOS_TEXT) -> Vocabulary:
        items = [t.encode("utf-8") for t in tokens if t != eos]
        return cls(tokens=(*items, b""), eos_id=len(items))

    def decode(self, ids: Sequence[int]) -> str:
        return b"".join(self.tokens[i] for i in ids if i != self.eos_id).decode("utf-8", errors="replace")


class BPETokenizer:
    """Byte-level BPE with deterministic, rank-ordered merges."""

    def __init__(self, merges: Sequence[tuple[int, int]]) -> None:
        self.merges: tuple[tuple[int, int], ...] = tuple(merges)
        self.ranks: dict[tuple[int, int], int] = {pair: r for r, pair in enumerate(self.merges)}
        pieces: list[bytes] = [bytes((b,)) for b in range(256)]
        for left, right in self.merges:
            pieces.append(pieces[left] + pieces[right])
        self.vocab = Vocabulary(tokens=(*pieces, b""), eos_id=len(pieces))

    @property
    def eos_id(self) -> int:
        return self.vocab.eos_id

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, corpus: Iterable[str], vocab_size: int) -> BPETokenizer:
        """Learn ``vocab_size - 257`` merges (256 bytes + EOS are always present)."""
        if vocab_size < 257:
            raise ValueError("vocab_size must be >= 257 (256 bytes + EOS)")
        words = Counter(tuple(text.encode("utf-8")) for text in corpus)
        seqs: dict[tuple[int, ...], int] = dict(words)
        merges: list[tuple[int, int]] = []
        next_id = 256
        while next_id < vocab_size - 1:
            pairs: Counter[tuple[int, int]] = Counter()
            for seq, freq in seqs.items():
                for a, b in zip(seq, seq[1:], strict=False):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            # Highest count wins; ties broken by the smallest pair for determinism.
            best = min(pairs, key=lambda p: (-pairs[p], p))
            if pairs[best] < 2:
                break
            merges.append(best)
            seqs = {_merge(seq, best, next_id): f for seq, f in seqs.items()}
            next_id += 1
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            ranked = [
                (self.ranks.get(pair, len(self.ranks)), i)
                for i, pair in enumerate(zip(ids, ids[1:], strict=False))
            ]
            rank, _ = min(ranked)
            if rank == len(self.ranks):
                break
            ids = list(_merge(tuple(ids), self.merges[rank], 256 + rank))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.vocab.decode(ids)


def _merge(seq: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)
