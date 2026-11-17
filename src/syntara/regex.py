"""Byte-level regular-expression compiler: recursive-descent parser -> Thompson NFA -> DFA.

The pipeline is the classic one used by structured-generation engines (Outlines, XGrammar):

1. :func:`parse_regex` turns a pattern into a small AST whose leaves are *byte sets*.
   Non-ASCII literals are expanded into their UTF-8 byte sequences, so every automaton below
   runs over the 256-symbol byte alphabet that LLM byte-level tokenizers use.
2. :func:`build_nfa` performs Thompson's construction.
3. :func:`compile_regex` runs the subset construction, prunes states that cannot reach an
   accepting state (so a constrained decoder can never paint itself into a corner), and
   minimises the result with Moore's partition-refinement algorithm.

Supported syntax: literals, ``.``, ``[...]`` / ``[^...]`` classes with ranges, ``\\d \\D \\w \\W
\\s \\S``, ``\\n \\t \\r \\f \\v``, ``\\xHH``, ``\\uHHHH``, escaped metacharacters, groups ``(...)``
and ``(?:...)``, alternation ``|`` and the quantifiers ``* + ? {m} {m,} {m,n}``. Patterns are
implicitly anchored (full match).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

ALPHABET_SIZE = 256
MAX_REPEAT = 1000

_DIGITS = frozenset(range(ord("0"), ord("9") + 1))
_WORD = frozenset(
    list(range(ord("a"), ord("z") + 1)) + list(range(ord("A"), ord("Z") + 1)) + list(_DIGITS) + [ord("_")]
)
_SPACE = frozenset(ord(c) for c in " \t\n\r\f\v")
_ALL = frozenset(range(ALPHABET_SIZE))
_SIMPLE_ESCAPES = {"n": 0x0A, "t": 0x09, "r": 0x0D, "f": 0x0C, "v": 0x0B, "0": 0x00}
_CLASS_ESCAPES = {
    "d": _DIGITS,
    "D": _ALL - _DIGITS,
    "w": _WORD,
    "W": _ALL - _WORD,
    "s": _SPACE,
    "S": _ALL - _SPACE,
}
_META = set("\\.*+?()[]{}|^$")


class RegexError(ValueError):
    """Raised for malformed or unsupported patterns."""


# --------------------------------------------------------------------------------------- AST


@dataclass(frozen=True, slots=True)
class ByteSet:
    """Matches exactly one byte drawn from ``members``."""

    members: frozenset[int]


@dataclass(frozen=True, slots=True)
class Concat:
    parts: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Alt:
    options: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Repeat:
    node: Node
    low: int
    high: int | None


Node = ByteSet | Concat | Alt | Repeat

EMPTY: Node = Concat(())


def _literal(text: str) -> Node:
    data = text.encode("utf-8")
    if len(data) == 1:
        return ByteSet(frozenset(data))
    return Concat(tuple(ByteSet(frozenset((b,))) for b in data))


def escape_literal(text: str) -> str:
    """Escape ``text`` so that :func:`parse_regex` matches it verbatim."""
    return "".join("\\" + ch if ch in _META else ch for ch in text)


class _Parser:
    def __init__(self, pattern: str) -> None:
        self.p = pattern
        self.i = 0

    def peek(self) -> str | None:
        return self.p[self.i] if self.i < len(self.p) else None

    def take(self) -> str:
        if self.i >= len(self.p):
            raise RegexError("unexpected end of pattern")
        ch = self.p[self.i]
        self.i += 1
        return ch

    def parse(self) -> Node:
        node = self.alt()
        if self.i != len(self.p):
            raise RegexError(f"unexpected {self.p[self.i]!r} at offset {self.i}")
        return node

    def alt(self) -> Node:
        options = [self.concat()]
        while self.peek() == "|":
            self.i += 1
            options.append(self.concat())
        return options[0] if len(options) == 1 else Alt(tuple(options))

    def concat(self) -> Node:
        parts: list[Node] = []
        while (ch := self.peek()) is not None and ch not in "|)":
            parts.append(self.repeat())
        return parts[0] if len(parts) == 1 else Concat(tuple(parts))

    def repeat(self) -> Node:
        node = self.atom()
        while (ch := self.peek()) is not None and ch in "*+?{":
            self.i += 1
            if ch == "*":
                node = Repeat(node, 0, None)
            elif ch == "+":
                node = Repeat(node, 1, None)
            elif ch == "?":
                node = Repeat(node, 0, 1)
            else:
                low, high = self.bounds()
                node = Repeat(node, low, high)
        return node

    def number(self) -> int:
        start = self.i
        while (ch := self.peek()) is not None and ch.isdigit():
            self.i += 1
        if start == self.i:
            raise RegexError(f"expected a number at offset {start}")
        return int(self.p[start : self.i])

    def bounds(self) -> tuple[int, int | None]:
        low = self.number()
        high: int | None = low
        if self.peek() == ",":
            self.i += 1
            high = None if self.peek() == "}" else self.number()
        if self.take() != "}":
            raise RegexError("unterminated {m,n} quantifier")
        if high is not None and high < low:
            raise RegexError(f"invalid repeat bounds {{{low},{high}}}")
        if max(low, high or 0) > MAX_REPEAT:
            raise RegexError(f"repeat bound exceeds {MAX_REPEAT}")
        return low, high

    def atom(self) -> Node:
        ch = self.take()
        if ch == "(":
            if self.p.startswith("?:", self.i):
                self.i += 2
            node = self.alt()
            if self.peek() != ")":
                raise RegexError("unbalanced parenthesis")
            self.i += 1
            return node
        if ch == "[":
            return self.char_class()
        if ch == ".":
            return ByteSet(_ALL - {0x0A})
        if ch == "\\":
            return self.escape()
        if ch in "*+?{":
            raise RegexError(f"nothing to repeat at offset {self.i - 1}")
        if ch in "^$]}":
            raise RegexError(f"unsupported metacharacter {ch!r}; escape it")
        return _literal(ch)

    def escape_char(self) -> tuple[str, frozenset[int] | None, bool]:
        """Decode one escape. Returns (literal, predefined class or None, is_raw_byte).

        ``\\xHH`` denotes a raw byte (so ``\\xff`` matches the single byte 0xFF); every other
        literal is a Unicode character that is UTF-8 encoded when matched.
        """
        ch = self.take()
        if ch in _CLASS_ESCAPES:
            return "", _CLASS_ESCAPES[ch], False
        if ch in _SIMPLE_ESCAPES:
            return chr(_SIMPLE_ESCAPES[ch]), None, True
        if ch in "xu":
            width = 2 if ch == "x" else 4
            digits = self.p[self.i : self.i + width]
            if len(digits) != width or any(c not in "0123456789abcdefABCDEF" for c in digits):
                raise RegexError(f"invalid \\{ch} escape at offset {self.i}")
            self.i += width
            return chr(int(digits, 16)), None, ch == "x"
        if ch.isalnum():
            raise RegexError(f"unsupported escape \\{ch}")
        return ch, None, False

    def escape(self) -> Node:
        literal, klass, raw = self.escape_char()
        if klass is not None:
            return ByteSet(klass)
        if raw:
            return ByteSet(frozenset((ord(literal),)))
        return _literal(literal)

    def class_member(self) -> tuple[int | None, frozenset[int] | None]:
        ch = self.take()
        raw = False
        if ch == "\\":
            literal, klass, raw = self.escape_char()
            if klass is not None:
                return None, klass
            ch = literal
        code = ord(ch)
        if code >= 0x80 and not (raw and code < ALPHABET_SIZE):
            raise RegexError("non-ASCII characters are not supported inside [...] classes")
        return code, None

    def char_class(self) -> ByteSet:
        negate = False
        if self.peek() == "^":
            negate = True
            self.i += 1
        members: set[int] = set()
        first = True
        while True:
            ch = self.peek()
            if ch is None:
                raise RegexError("unterminated character class")
            if ch == "]" and not first:
                self.i += 1
                break
            first = False
            low, klass = self.class_member()
            if klass is not None:
                members |= klass
                continue
            assert low is not None
            if self.peek() == "-" and self.p[self.i + 1 : self.i + 2] not in ("]", ""):
                self.i += 1
                high, hk = self.class_member()
                if hk is not None or high is None or high < low:
                    raise RegexError("invalid range in character class")
                members |= set(range(low, high + 1))
            else:
                members.add(low)
        final = frozenset(_ALL - members) if negate else frozenset(members)
        if not final:
            raise RegexError("empty character class")
        return ByteSet(final)


def parse_regex(pattern: str) -> Node:
    """Parse ``pattern`` into a byte-level AST."""
    return _Parser(pattern).parse()


# --------------------------------------------------------------------------------------- NFA


@dataclass(slots=True)
class Nfa:
    """Thompson NFA over bytes: ``edges[s]`` holds (byte set, target); ``eps[s]`` epsilon arcs."""

    edges: list[list[tuple[frozenset[int], int]]] = field(default_factory=list)
    eps: list[list[int]] = field(default_factory=list)
    start: int = 0
    accept: int = 0

    def new_state(self) -> int:
        self.edges.append([])
        self.eps.append([])
        return len(self.edges) - 1

    @property
    def num_states(self) -> int:
        return len(self.edges)


def _emit(nfa: Nfa, node: Node) -> tuple[int, int]:
    """Emit a fragment for ``node`` and return its (entry, exit) states."""
    if isinstance(node, ByteSet):
        s, t = nfa.new_state(), nfa.new_state()
        nfa.edges[s].append((node.members, t))
        return s, t
    if isinstance(node, Concat):
        s = nfa.new_state()
        cur = s
        for part in node.parts:
            ps, pt = _emit(nfa, part)
            nfa.eps[cur].append(ps)
            cur = pt
        return s, cur
    if isinstance(node, Alt):
        s, t = nfa.new_state(), nfa.new_state()
        for option in node.options:
            os_, ot = _emit(nfa, option)
            nfa.eps[s].append(os_)
            nfa.eps[ot].append(t)
        return s, t
    # Repeat
    s = nfa.new_state()
    cur = s
    for _ in range(node.low):
        ps, pt = _emit(nfa, node.node)
        nfa.eps[cur].append(ps)
        cur = pt
    if node.high is None:
        ps, pt = _emit(nfa, node.node)
        nfa.eps[cur].append(ps)
        nfa.eps[pt].append(ps)
        t = nfa.new_state()
        nfa.eps[cur].append(t)
        nfa.eps[pt].append(t)
        return s, t
    t = nfa.new_state()
    for _ in range(node.high - node.low):
        nfa.eps[cur].append(t)
        ps, pt = _emit(nfa, node.node)
        nfa.eps[cur].append(ps)
        cur = pt
    nfa.eps[cur].append(t)
    return s, t


def build_nfa(node: Node) -> Nfa:
    """Thompson construction for ``node``."""
    nfa = Nfa()
    nfa.start, nfa.accept = _emit(nfa, node)
    return nfa


# --------------------------------------------------------------------------------------- DFA


@dataclass(frozen=True, slots=True)
class Dfa:
    """Dense byte DFA. ``table[s][b]`` is the successor state or ``-1`` (dead).

    Invariants established by :func:`compile_regex`: state ``0`` is the start state, every
    state is reachable from the start and can reach an accepting state, and the automaton
    is minimal.
    """

    table: tuple[tuple[int, ...], ...]
    accepting: frozenset[int]
    start: int = 0

    @property
    def num_states(self) -> int:
        return len(self.table)

    def step(self, state: int, byte: int) -> int:
        if state < 0:
            return -1
        return self.table[state][byte]

    def walk(self, state: int, data: bytes) -> int:
        for byte in data:
            state = self.step(state, byte)
            if state < 0:
                return -1
        return state

    def matches(self, data: bytes | str) -> bool:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        end = self.walk(self.start, raw)
        return end in self.accepting

    def is_accepting(self, state: int) -> bool:
        return state in self.accepting

    def to_array(self) -> npt.NDArray[np.int32]:
        return np.asarray(self.table, dtype=np.int32).reshape(self.num_states, ALPHABET_SIZE)

    @property
    def num_transitions(self) -> int:
        return sum(1 for row in self.table for t in row if t >= 0)


def _closure(nfa: Nfa, states: set[int]) -> frozenset[int]:
    stack = list(states)
    seen = set(states)
    while stack:
        s = stack.pop()
        for t in nfa.eps[s]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return frozenset(seen)


def subset_construction(nfa: Nfa) -> tuple[list[list[int]], set[int]]:
    """Rabin-Scott powerset construction. Returns (table, accepting set); start is ``0``."""
    start = _closure(nfa, {nfa.start})
    ids: dict[frozenset[int], int] = {start: 0}
    order = [start]
    table: list[list[int]] = []
    accepting: set[int] = set()
    idx = 0
    while idx < len(order):
        current = order[idx]
        if nfa.accept in current:
            accepting.add(idx)
        moves: dict[int, set[int]] = {}
        for s in current:
            for members, target in nfa.edges[s]:
                for b in members:
                    moves.setdefault(b, set()).add(target)
        row = [-1] * ALPHABET_SIZE
        cache: dict[frozenset[int], int] = {}
        for b, targets in moves.items():
            key = frozenset(targets)
            if key not in cache:
                closed = _closure(nfa, set(key))
                if closed not in ids:
                    ids[closed] = len(order)
                    order.append(closed)
                cache[key] = ids[closed]
            row[b] = cache[key]
        table.append(row)
        idx += 1
    return table, accepting


def prune_dead_states(table: list[list[int]], accepting: set[int]) -> tuple[list[list[int]], set[int]]:
    """Drop states that are unreachable from ``0`` or cannot reach an accepting state."""
    n = len(table)
    reverse: list[set[int]] = [set() for _ in range(n)]
    for s, row in enumerate(table):
        for t in row:
            if t >= 0:
                reverse[t].add(s)
    live = set(accepting)
    stack = list(accepting)
    while stack:
        t = stack.pop()
        for s in reverse[t]:
            if s not in live:
                live.add(s)
                stack.append(s)
    if 0 not in live:
        raise RegexError("pattern matches no string")
    reach = {0}
    stack = [0]
    while stack:
        s = stack.pop()
        for t in table[s]:
            if t >= 0 and t in live and t not in reach:
                reach.add(t)
                stack.append(t)
    keep = sorted(reach)
    remap = {old: new for new, old in enumerate(keep)}
    new_table = [[remap.get(t, -1) if t >= 0 else -1 for t in table[old]] for old in keep]
    return new_table, {remap[a] for a in accepting if a in remap}


def minimize(table: list[list[int]], accepting: set[int]) -> tuple[list[list[int]], set[int]]:
    """Moore partition refinement; returns a minimal DFA with the start state renumbered 0."""
    n = len(table)
    block = [1 if s in accepting else 0 for s in range(n)]
    num_blocks = len(set(block))
    while True:
        signatures: dict[tuple[int, ...], int] = {}
        new_block = [0] * n
        for s in range(n):
            sig = (block[s], *(block[t] if t >= 0 else -1 for t in table[s]))
            new_block[s] = signatures.setdefault(sig, len(signatures))
        if len(signatures) == num_blocks:
            break
        block, num_blocks = new_block, len(signatures)
    # Renumber blocks in BFS order from the start state for canonical, deterministic output.
    order: dict[int, int] = {block[0]: 0}
    rep: dict[int, int] = {}
    for s in range(n):
        rep.setdefault(block[s], s)
    queue = [block[0]]
    while queue:
        b = queue.pop(0)
        for t in table[rep[b]]:
            if t >= 0 and block[t] not in order:
                order[block[t]] = len(order)
                queue.append(block[t])
    new_table = [[-1] * ALPHABET_SIZE for _ in order]
    for b, new_id in order.items():
        new_table[new_id] = [order[block[t]] if t >= 0 else -1 for t in table[rep[b]]]
    new_accepting = {order[block[a]] for a in accepting}
    return new_table, new_accepting


def compile_ast(node: Node) -> Dfa:
    table, accepting = subset_construction(build_nfa(node))
    table, accepting = prune_dead_states(table, accepting)
    table, accepting = minimize(table, accepting)
    return Dfa(table=tuple(tuple(r) for r in table), accepting=frozenset(accepting))


def compile_regex(pattern: str) -> Dfa:
    """Compile ``pattern`` into a pruned, minimal byte-level :class:`Dfa`."""
    return compile_ast(parse_regex(pattern))
