"""Context-free grammar constraints: a GBNF-style EBNF front-end and a byte-level Earley parser.

Regular languages cannot express arbitrarily nested JSON, code or DSLs. For those, Syntara
runs an incremental Earley recogniser (Earley, 1970) directly over bytes, with the
Aycock-Horspool (2002) treatment of nullable non-terminals. Every byte fed produces a new
chart column; a prefix is *viable* iff its last column is non-empty. Unproductive rules are
pruned up-front, so every viable prefix can be completed to a sentence of the grammar.

Grammar syntax (a subset of llama.cpp GBNF)::

    root   ::= "{" ws pair ( ws "," ws pair )* ws "}"
    pair   ::= string ws ":" ws value
    string ::= "\\"" [a-z]+ "\\""
    ws     ::= [ \\t\\n]*

Terminals are ``"literals"`` (with ``\\n \\t \\" \\\\ \\xHH \\uHHHH`` escapes), byte classes
``[...]`` / ``[^...]`` and ``.``; operators are grouping ``( )``, alternation ``|`` and the
postfix quantifiers ``* + ?``. ``#`` starts a comment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from syntara.regex import ALPHABET_SIZE, RegexError, _Parser
from syntara.tokenizer import Vocabulary

BoolArray = npt.NDArray[np.bool_]


class GrammarError(ValueError):
    """Raised for malformed grammars or grammars whose start symbol derives nothing."""


@dataclass(frozen=True, slots=True)
class Production:
    lhs: int
    rhs: tuple[int, ...]  # >= 0: non-terminal id; < 0: terminal id encoded as -(k + 1)


@dataclass(slots=True)
class Grammar:
    names: list[str]
    terminals: list[frozenset[int]]
    productions: list[Production]
    start: int
    by_lhs: dict[int, list[int]] = field(default_factory=dict)
    nullable: frozenset[int] = frozenset()

    @property
    def num_rules(self) -> int:
        return len(self.productions)


# ------------------------------------------------------------------------------------ front-end


class _GrammarParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.i = 0
        self.names: list[str] = []
        self.ids: dict[str, int] = {}
        self.defined: set[int] = set()
        self.terminals: list[frozenset[int]] = []
        self.term_ids: dict[frozenset[int], int] = {}
        self.productions: list[tuple[int, tuple[int, ...]]] = []
        self.fresh = 0

    # -- lexing helpers
    def skip(self, newlines: bool = True) -> None:
        while self.i < len(self.text):
            ch = self.text[self.i]
            if ch == "#":
                while self.i < len(self.text) and self.text[self.i] != "\n":
                    self.i += 1
            elif ch in " \t\r" or (newlines and ch == "\n"):
                self.i += 1
            else:
                break

    def ident(self) -> str | None:
        start = self.i
        while self.i < len(self.text) and (self.text[self.i].isalnum() or self.text[self.i] in "-_"):
            self.i += 1
        return self.text[start : self.i] if self.i > start else None

    def at_rule_start(self) -> bool:
        save = self.i
        self.skip()
        name = self.ident()
        self.skip(newlines=False)
        ok = name is not None and self.text.startswith("::=", self.i)
        self.i = save
        return ok

    def nt(self, name: str) -> int:
        if name not in self.ids:
            self.ids[name] = len(self.names)
            self.names.append(name)
        return self.ids[name]

    def term(self, members: frozenset[int]) -> int:
        if members not in self.term_ids:
            self.term_ids[members] = len(self.terminals)
            self.terminals.append(members)
        return -(self.term_ids[members] + 1)

    def new_nt(self, base: str) -> int:
        self.fresh += 1
        nid = self.nt(f"{base}__{self.fresh}")
        self.defined.add(nid)
        return nid

    # -- grammar
    def parse(self) -> None:
        self.skip()
        while self.i < len(self.text):
            name = self.ident()
            if name is None:
                raise GrammarError(f"expected rule name at offset {self.i}")
            self.skip(newlines=False)
            if not self.text.startswith("::=", self.i):
                raise GrammarError(f"expected '::=' after {name!r}")
            self.i += 3
            lhs = self.nt(name)
            if lhs in self.defined:
                raise GrammarError(f"rule {name!r} defined twice")
            self.defined.add(lhs)
            for alt in self.alternatives(name, top=True):
                self.productions.append((lhs, tuple(alt)))
            self.skip()

    def alternatives(self, base: str, top: bool) -> list[list[int]]:
        alts = [self.sequence(base, top)]
        while True:
            self.skip()
            if self.i < len(self.text) and self.text[self.i] == "|":
                self.i += 1
                alts.append(self.sequence(base, top))
            else:
                return alts

    def sequence(self, base: str, top: bool) -> list[int]:
        seq: list[int] = []
        while True:
            if top and self.at_rule_start():
                return seq
            self.skip()
            if self.i >= len(self.text) or self.text[self.i] in "|)":
                return seq
            symbols = self.atom(base)
            self.skip(newlines=not top)
            if self.i < len(self.text) and self.text[self.i] in "*+?":
                op = self.text[self.i]
                self.i += 1
                seq.append(self.quantify(base, symbols, op))
            else:
                seq.extend(symbols)

    def quantify(self, base: str, symbols: list[int], op: str) -> int:
        if len(symbols) == 1:
            inner = symbols[0]
        else:
            inner = self.new_nt(base)
            self.productions.append((inner, tuple(symbols)))
        rep = self.new_nt(base)
        if op == "*":
            self.productions += [(rep, ()), (rep, (rep, inner))]
        elif op == "+":
            self.productions += [(rep, (inner,)), (rep, (rep, inner))]
        else:
            self.productions += [(rep, ()), (rep, (inner,))]
        return rep

    def atom(self, base: str) -> list[int]:
        ch = self.text[self.i]
        if ch == "(":
            self.i += 1
            group = self.new_nt(base)
            for alt in self.alternatives(base, top=False):
                self.productions.append((group, tuple(alt)))
            self.skip()
            if self.i >= len(self.text) or self.text[self.i] != ")":
                raise GrammarError("unbalanced parenthesis in grammar")
            self.i += 1
            return [group]
        if ch == '"':
            return self.literal()
        if ch == "[":
            rp = _Parser(self.text)
            rp.i = self.i + 1
            try:
                node = rp.char_class()
            except RegexError as exc:
                raise GrammarError(str(exc)) from exc
            self.i = rp.i
            return [self.term(node.members)]
        if ch == ".":
            self.i += 1
            return [self.term(frozenset(range(ALPHABET_SIZE)) - {0x0A})]
        name = self.ident()
        if name is None:
            raise GrammarError(f"unexpected {ch!r} at offset {self.i}")
        return [self.nt(name)]

    def literal(self) -> list[int]:
        self.i += 1
        out: list[int] = []
        rp = _Parser(self.text)
        while True:
            if self.i >= len(self.text):
                raise GrammarError("unterminated string literal")
            ch = self.text[self.i]
            if ch == '"':
                self.i += 1
                return out
            if ch == "\\":
                rp.i = self.i + 1
                try:
                    lit, klass, raw = rp.escape_char()
                except RegexError as exc:
                    raise GrammarError(str(exc)) from exc
                if klass is not None:
                    raise GrammarError("character-class escapes are not allowed in literals")
                self.i = rp.i
                data = bytes((ord(lit),)) if raw else lit.encode("utf-8")
            else:
                self.i += 1
                data = ch.encode("utf-8")
            out.extend(self.term(frozenset((b,))) for b in data)


def parse_grammar(text: str, start: str = "root") -> Grammar:
    """Parse EBNF ``text``, prune unproductive/unreachable rules and precompute nullability."""
    gp = _GrammarParser(text)
    gp.parse()
    if start not in gp.ids or gp.ids[start] not in gp.defined:
        raise GrammarError(f"start rule {start!r} is not defined")
    undefined = [gp.names[i] for i in range(len(gp.names)) if i not in gp.defined]
    if undefined:
        raise GrammarError(f"undefined rules: {sorted(undefined)}")
    prods = gp.productions
    # Productive non-terminals (derive at least one terminal string).
    productive: set[int] = set()
    changed = True
    while changed:
        changed = False
        for lhs, rhs in prods:
            if lhs not in productive and all(s < 0 or s in productive for s in rhs):
                productive.add(lhs)
                changed = True
    root = gp.ids[start]
    if root not in productive:
        raise GrammarError(f"start rule {start!r} derives no finite string")
    prods = [(lhs, rhs) for lhs, rhs in prods if all(s < 0 or s in productive for s in rhs)]
    # Reachable from the start symbol.
    reach = {root}
    stack = [root]
    while stack:
        cur = stack.pop()
        for lhs, rhs in prods:
            if lhs == cur:
                for s in rhs:
                    if s >= 0 and s not in reach:
                        reach.add(s)
                        stack.append(s)
    final = [Production(lhs, rhs) for lhs, rhs in prods if lhs in reach]
    by_lhs: dict[int, list[int]] = {}
    for idx, p in enumerate(final):
        by_lhs.setdefault(p.lhs, []).append(idx)
    nullable: set[int] = set()
    changed = True
    while changed:
        changed = False
        for p in final:
            if p.lhs not in nullable and all(s >= 0 and s in nullable for s in p.rhs):
                nullable.add(p.lhs)
                changed = True
    return Grammar(
        names=gp.names,
        terminals=gp.terminals,
        productions=final,
        start=root,
        by_lhs=by_lhs,
        nullable=frozenset(nullable),
    )


# --------------------------------------------------------------------------------------- Earley

Item = tuple[int, int, int]  # (production, dot, origin)


@dataclass(frozen=True, slots=True)
class Column:
    items: frozenset[Item]
    waiting: dict[int, tuple[Item, ...]]
    scans: tuple[tuple[int, Item], ...]


Chart = tuple[Column, ...]


class EarleyParser:
    """Incremental, persistent byte-level Earley recogniser."""

    def __init__(self, grammar: Grammar) -> None:
        self.g = grammar

    def _close(self, chart: Chart, seed: Sequence[Item]) -> Column:
        g = self.g
        k = len(chart)
        items: set[Item] = set()
        waiting: dict[int, list[Item]] = {}
        scans: list[tuple[int, Item]] = []
        agenda: list[Item] = []

        def add(it: Item) -> None:
            if it not in items:
                items.add(it)
                agenda.append(it)

        for it in seed:
            add(it)
        while agenda:
            p, d, o = agenda.pop()
            rhs = g.productions[p].rhs
            if d == len(rhs):
                lhs = g.productions[p].lhs
                source = chart[o].waiting.get(lhs, ()) if o < k else tuple(waiting.get(lhs, ()))
                for wp, wd, wo in source:
                    add((wp, wd + 1, wo))
                continue
            sym = rhs[d]
            if sym < 0:
                scans.append((-sym - 1, (p, d, o)))
                continue
            waiting.setdefault(sym, []).append((p, d, o))
            for q in g.by_lhs.get(sym, ()):
                add((q, 0, k))
            if sym in g.nullable:
                add((p, d + 1, o))
        return Column(
            items=frozenset(items),
            waiting={nt: tuple(v) for nt, v in waiting.items()},
            scans=tuple(scans),
        )

    def initial(self) -> Chart:
        seed = [(q, 0, 0) for q in self.g.by_lhs.get(self.g.start, ())]
        return (self._close((), seed),)

    def feed(self, chart: Chart, byte: int) -> Chart | None:
        """Return the chart after consuming ``byte`` or ``None`` if the prefix dies."""
        terms = self.g.terminals
        seed = [(p, d + 1, o) for t, (p, d, o) in chart[-1].scans if byte in terms[t]]
        if not seed:
            return None
        return (*chart, self._close(chart, seed))

    def feed_bytes(self, chart: Chart, data: bytes) -> Chart | None:
        cur: Chart | None = chart
        for b in data:
            if cur is None:
                return None
            cur = self.feed(cur, b)
        return cur

    def accepts(self, chart: Chart) -> bool:
        g = self.g
        return any(
            o == 0 and d == len(g.productions[p].rhs) and g.productions[p].lhs == g.start
            for p, d, o in chart[-1].items
        )

    def recognize(self, data: bytes | str) -> bool:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        chart = self.feed_bytes(self.initial(), raw)
        return chart is not None and self.accepts(chart)


class GrammarConstraint:
    """Token-level constraint whose state is an Earley chart.

    Masks are computed by walking the vocabulary in lexicographic order and re-using the
    chart reached by the longest common prefix of consecutive tokens (an implicit trie DFS),
    mirroring the native DFA index kernel.
    """

    def __init__(self, grammar: Grammar | str, vocab: Vocabulary, start: str = "root") -> None:
        self.grammar = parse_grammar(grammar, start) if isinstance(grammar, str) else grammar
        self.parser = EarleyParser(self.grammar)
        self.vocab = vocab
        order = sorted(range(len(vocab)), key=lambda i: vocab.tokens[i])
        self._order = order
        self._lcp = [0] * len(order)
        for k in range(1, len(order)):
            a, b = vocab.tokens[order[k - 1]], vocab.tokens[order[k]]
            j = 0
            while j < min(len(a), len(b)) and a[j] == b[j]:
                j += 1
            self._lcp[k] = j
        self._cache_key: Chart | None = None
        self._cache: dict[int, Chart] = {}

    def initial(self) -> Chart:
        return self.parser.initial()

    def _successors(self, state: Chart) -> dict[int, Chart]:
        if self._cache_key is state:
            return self._cache
        tokens = self.vocab.tokens
        succ: dict[int, Chart] = {}
        stack: list[Chart | None] = [state]
        for k, t in enumerate(self._order):
            tok = tokens[t]
            if not tok:
                continue
            shared = min(self._lcp[k], len(stack) - 1)
            del stack[shared + 1 :]
            cur = stack[shared]
            for b in tok[shared:]:
                cur = None if cur is None else self.parser.feed(cur, b)
                stack.append(cur)
            if cur is not None:
                succ[t] = cur
        self._cache_key, self._cache = state, succ
        return succ

    def allowed(self, state: Chart) -> BoolArray:
        mask = np.zeros(len(self.vocab), dtype=np.bool_)
        for t in self._successors(state):
            mask[t] = True
        mask[self.vocab.eos_id] = self.parser.accepts(state)
        return mask

    def advance(self, state: Chart, token: int) -> Chart:
        if token == self.vocab.eos_id:
            if not self.parser.accepts(state):
                raise ValueError("EOS is not allowed before the grammar is satisfied")
            return state
        succ = self._successors(state)
        if token not in succ:
            raise ValueError(f"token {token} is not allowed here")
        return succ[token]

    def is_complete(self, state: Chart) -> bool:
        return self.parser.accepts(state)


JSON_GRAMMAR = r"""
# RFC 8259 JSON (byte-level; strings accept any non-control, non-quote byte).
root    ::= ws value ws
value   ::= object | array | string | number | "true" | "false" | "null"
object  ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
member  ::= string ws ":" ws value
array   ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string  ::= "\"" ( [^"\\\x00-\x1f] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] ) )* "\""
number  ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
ws      ::= [ \t\n\r]*
"""

ARITHMETIC_GRAMMAR = r"""
# Left-recursive arithmetic expressions (Earley handles left recursion natively).
root   ::= expr
expr   ::= expr ws [-+] ws term | term
term   ::= term ws [*/] ws factor | factor
factor ::= number | "(" ws expr ws ")"
number ::= [0-9]+
ws     ::= " "?
"""
