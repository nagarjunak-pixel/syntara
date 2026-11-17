"""SHA-256-cached C++20 compiler + ctypes bridge with a bit-exact pure-Python fallback."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

from syntara.tokenizer import Vocabulary

I32Array = npt.NDArray[np.int32]
F64Array = npt.NDArray[np.float64]

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = _REPO_ROOT / "csrc" / "syntara_kernels.cpp"


def sorted_order_and_lcp(tokens: tuple[bytes, ...]) -> tuple[I32Array, I32Array]:
    """Lexicographic token order and the longest-common-prefix with each predecessor."""
    order = sorted(range(len(tokens)), key=lambda i: tokens[i])
    lcp = np.zeros(len(order), dtype=np.int32)
    for k in range(1, len(order)):
        a, b = tokens[order[k - 1]], tokens[order[k]]
        n = min(len(a), len(b))
        j = 0
        while j < n and a[j] == b[j]:
            j += 1
        lcp[k] = j
    return np.asarray(order, dtype=np.int32), lcp


@dataclass(frozen=True, slots=True, eq=False)
class VocabLayout:
    """Per-vocabulary preprocessing shared by every index build (computed once, cached)."""

    tokens: tuple[bytes, ...]
    eos_id: int
    order: I32Array
    lcp: I32Array
    blob: npt.NDArray[np.uint8]
    offsets: npt.NDArray[np.int64]


@lru_cache(maxsize=32)
def vocab_layout(vocab: Vocabulary) -> VocabLayout:
    """Sorted order, LCP array and a flat byte blob for ``vocab`` (cached by identity)."""
    order, lcp = sorted_order_and_lcp(vocab.tokens)
    blob = np.frombuffer(b"".join(vocab.tokens) or b"\x00", dtype=np.uint8).copy()
    offsets = np.zeros(len(vocab.tokens) + 1, dtype=np.int64)
    np.cumsum([len(t) for t in vocab.tokens], out=offsets[1:])
    return VocabLayout(vocab.tokens, vocab.eos_id, order, lcp, blob, offsets)


def py_build_token_index(
    table: I32Array,
    accepting: npt.NDArray[np.uint8],
    tokens: tuple[bytes, ...],
    eos_id: int,
    order: I32Array,
    lcp: I32Array,
) -> I32Array:
    """Pure-Python twin of ``syntara_build_token_index`` (same prefix-sharing algorithm)."""
    num_states = int(table.shape[0])
    vocab = len(tokens)
    out = np.full((num_states, vocab), -1, dtype=np.int32)
    rows: list[list[int]] = table.tolist()
    max_len = max((len(t) for t in tokens), default=0)
    for s in range(num_states):
        stack = [-1] * (max_len + 1)
        stack[0] = s
        valid = 0
        row = out[s]
        for k in range(vocab):
            t = int(order[k])
            tok = tokens[t]
            if t == eos_id:
                row[t] = s if accepting[s] else -1
                continue
            if not tok:
                continue
            shared = int(lcp[k])
            if shared > valid:
                continue
            depth = shared
            dead = False
            while depth < len(tok):
                nxt = rows[stack[depth]][tok[depth]]
                if nxt < 0:
                    dead = True
                    break
                stack[depth + 1] = nxt
                depth += 1
            valid = depth
            if not dead:
                row[t] = stack[len(tok)]
    return out


def py_masked_argmax(logits: F64Array, next_row: I32Array) -> int:
    allowed = np.flatnonzero(next_row >= 0)
    if allowed.size == 0:
        return -1
    return int(allowed[int(np.argmax(logits[allowed]))])


class KernelBridge:
    """Compiles ``csrc/syntara_kernels.cpp`` with ``-O3 -std=c++20`` and loads it via ctypes.

    Set ``SYNTARA_DISABLE_NATIVE=1`` (or pass ``use_native=False``) to force the Python path.
    """

    def __init__(self, source: Path | None = None, use_native: bool | None = None) -> None:
        self._src = source or DEFAULT_SOURCE
        self._lib: ctypes.CDLL | None = None
        self.backend = "python-fallback"
        if use_native is None:
            use_native = os.environ.get("SYNTARA_DISABLE_NATIVE", "0") != "1"
        if use_native:
            self._try_load()

    @property
    def is_native(self) -> bool:
        return self._lib is not None

    def _try_load(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None or not self._src.exists():
            return
        cache = self._src.parent.parent / ".syntara_cache"
        cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(self._src.read_bytes()).hexdigest()[:12]
        ext = "dylib" if platform.system() == "Darwin" else "so"
        lib_path = cache / f"libsyntara_{digest}.{ext}"
        if not lib_path.exists():
            tmp = lib_path.with_suffix(f".{os.getpid()}.tmp")
            cmd = [compiler, "-O3", "-std=c++20", "-shared", "-fPIC", str(self._src), "-o", str(tmp)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return
            tmp.replace(lib_path)
        lib = ctypes.CDLL(str(lib_path))
        i32p = ctypes.POINTER(ctypes.c_int32)
        lib.syntara_build_token_index.argtypes = [
            i32p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int64),
            i32p,
            i32p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint8),
            i32p,
        ]
        lib.syntara_build_token_index.restype = None
        lib.syntara_masked_argmax.argtypes = [ctypes.POINTER(ctypes.c_double), i32p, ctypes.c_int32]
        lib.syntara_masked_argmax.restype = ctypes.c_int32
        lib.syntara_count_allowed.argtypes = [i32p, ctypes.c_int32, ctypes.c_int32, i32p]
        lib.syntara_count_allowed.restype = None
        self._lib = lib
        self.backend = f"c++20-native ({Path(compiler).name} -O3)"

    def build_token_index(
        self,
        table: I32Array,
        accepting: npt.NDArray[np.uint8],
        layout: VocabLayout,
    ) -> I32Array:
        """Compute ``next[state, token]`` for a dense DFA ``table`` over ``layout``'s vocabulary."""
        table_c = np.ascontiguousarray(table, dtype=np.int32)
        acc_c = np.ascontiguousarray(accepting, dtype=np.uint8)
        if self._lib is None:
            return py_build_token_index(
                table_c, acc_c, layout.tokens, layout.eos_id, layout.order, layout.lcp
            )
        num_states = int(table_c.shape[0])
        vocab = len(layout.tokens)
        out = np.empty((num_states, vocab), dtype=np.int32)
        i32p = ctypes.POINTER(ctypes.c_int32)
        self._lib.syntara_build_token_index(
            table_c.ctypes.data_as(i32p),
            num_states,
            layout.blob.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            layout.offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            layout.order.ctypes.data_as(i32p),
            layout.lcp.ctypes.data_as(i32p),
            vocab,
            layout.eos_id,
            acc_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            out.ctypes.data_as(i32p),
        )
        return out

    def masked_argmax(self, logits: F64Array, next_row: I32Array) -> int:
        if self._lib is None:
            return py_masked_argmax(logits, next_row)
        lg = np.ascontiguousarray(logits, dtype=np.float64)
        nr = np.ascontiguousarray(next_row, dtype=np.int32)
        return int(
            self._lib.syntara_masked_argmax(
                lg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                nr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                int(nr.shape[0]),
            )
        )

    def count_allowed(self, index: I32Array) -> I32Array:
        idx = np.ascontiguousarray(index, dtype=np.int32)
        if self._lib is None:
            return (idx >= 0).sum(axis=1).astype(np.int32)
        out = np.empty(idx.shape[0], dtype=np.int32)
        i32p = ctypes.POINTER(ctypes.c_int32)
        self._lib.syntara_count_allowed(
            idx.ctypes.data_as(i32p), int(idx.shape[0]), int(idx.shape[1]), out.ctypes.data_as(i32p)
        )
        return out


_BRIDGE: KernelBridge | None = None


def get_bridge() -> KernelBridge:
    """Process-wide cached bridge (compiles once per source digest)."""
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = KernelBridge()
    return _BRIDGE
