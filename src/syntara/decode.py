"""Constrained decoding loop and exact closed-set choice scoring."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from syntara.index import RegexConstraint
from syntara.kernels import KernelBridge, get_bridge
from syntara.lm import NGramLM
from syntara.tokenizer import BPETokenizer, Vocabulary

BoolArray = npt.NDArray[np.bool_]


class Constraint[S](Protocol):
    """Anything that can mask a vocabulary step by step."""

    def initial(self) -> S: ...

    def allowed(self, state: S) -> BoolArray: ...

    def advance(self, state: S, token: int) -> S: ...

    def is_complete(self, state: S) -> bool: ...


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    token_ids: tuple[int, ...]
    finished: bool  # emitted EOS (and, when constrained, the constraint accepted)
    steps: int
    mean_allowed_fraction: float
    seconds: float


def generate[S](
    lm: NGramLM,
    vocab: Vocabulary,
    constraint: Constraint[S] | None = None,
    prompt: Sequence[int] = (),
    max_tokens: int = 128,
    temperature: float = 0.0,
    seed: int = 0,
    bridge: KernelBridge | None = None,
) -> Generation:
    """Greedy (``temperature == 0``) or sampled decoding with optional logit masking."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    br = bridge or get_bridge()
    rng = np.random.default_rng(seed)
    ctx = list(prompt)
    out: list[int] = []
    state: S | None = constraint.initial() if constraint is not None else None
    fractions: list[float] = []
    finished = False
    start = time.perf_counter()
    for _ in range(max_tokens):
        logits = lm.log_probs(ctx)
        if constraint is not None:
            assert state is not None
            mask = constraint.allowed(state)
            fractions.append(float(mask.mean()))
            if not mask.any():
                break
            if temperature == 0 and isinstance(constraint, RegexConstraint):
                tok = br.masked_argmax(logits, constraint.next_row(state))
            else:
                tok = _pick(np.where(mask, logits, -np.inf), temperature, rng)
        else:
            tok = _pick(logits, temperature, rng)
        if tok == vocab.eos_id:
            finished = True
            break
        if constraint is not None:
            assert state is not None
            state = constraint.advance(state, tok)
        out.append(tok)
        ctx.append(tok)
    return Generation(
        text=vocab.decode(out),
        token_ids=tuple(out),
        finished=finished,
        steps=len(out),
        mean_allowed_fraction=float(np.mean(fractions)) if fractions else 1.0,
        seconds=time.perf_counter() - start,
    )


def _pick(logits: npt.NDArray[np.float64], temperature: float, rng: np.random.Generator) -> int:
    if temperature == 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


@dataclass(frozen=True, slots=True)
class ChoiceScore:
    choice: str
    log_prob: float
    mean_log_prob: float
    num_tokens: int


def score_choices(
    lm: NGramLM, tokenizer: BPETokenizer, choices: Sequence[str], prompt: str = ""
) -> list[ChoiceScore]:
    """Rank a closed set of answers by exact sequence log-likelihood (including EOS).

    Unlike greedy constrained decoding, which commits to the locally best first token,
    this evaluates every candidate in full, so it returns the true maximum-likelihood choice.
    """
    if not choices:
        raise ValueError("choices must be non-empty")
    ctx = tokenizer.encode(prompt) if prompt else []
    scores: list[ChoiceScore] = []
    for choice in choices:
        ids = tokenizer.encode(choice) + [tokenizer.eos_id]
        lp = lm.sequence_log_prob(ctx, ids)
        scores.append(ChoiceScore(choice, lp, lp / len(ids), len(ids)))
    return sorted(scores, key=lambda s: (-s.log_prob, s.choice))
