"""JSON Schema -> regular expression compiler and a matching reference validator.

Compiling a schema to a regex (and from there to a DFA and a token index) is the approach
popularised by Outlines (Willard & Louf, 2023). The supported subset is the one that
structured-output APIs actually use for tool calls and extraction:

``type`` (string / integer / number / boolean / null / array / object, or a list of these),
``enum``, ``const``, ``anyOf`` / ``oneOf``, local ``$ref`` into ``$defs`` / ``definitions``,
``properties`` + ``required`` (emitted in declaration order, closed objects), ``items`` +
``minItems`` / ``maxItems``, ``minLength`` / ``maxLength`` / ``pattern`` / ``format``
(``date``, ``time``, ``date-time``, ``uuid``, ``email``) on strings.

Unknown *assertion* keywords raise :class:`SchemaError` instead of being silently ignored, so a
constraint that cannot be enforced is never reported as enforced.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from syntara.regex import escape_literal

STRING_CHAR = r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'
INTEGER = r"-?(?:0|[1-9][0-9]*)"
NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?"
BOOLEAN = r"(?:true|false)"
NULL = r"null"
FORMATS = {
    "date": r'"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"',
    "time": r'"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]+)?(?:Z|[+\-][0-9]{2}:[0-9]{2})?"',
    "date-time": (
        r'"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T'
        r'(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]+)?(?:Z|[+\-][0-9]{2}:[0-9]{2})"'
    ),
    "uuid": r'"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"',
    "email": r'"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)*\.[a-zA-Z]{2,}"',
}
_ANNOTATIONS = {
    "title",
    "description",
    "$schema",
    "$id",
    "$comment",
    "default",
    "examples",
    "$defs",
    "definitions",
    "additionalProperties",
}
_KEYWORDS = {
    "type",
    "enum",
    "const",
    "anyOf",
    "oneOf",
    "$ref",
    "properties",
    "required",
    "items",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "format",
}
MAX_REF_DEPTH = 16


class SchemaError(ValueError):
    """Raised for schemas outside the supported, enforceable subset."""


def _strip_anchors(pattern: str) -> str:
    """Patterns are always full-match; drop redundant ``^`` / ``$`` anchors."""
    if pattern.startswith("^"):
        pattern = pattern[1:]
    if pattern.endswith("$") and not pattern.endswith("\\$"):
        pattern = pattern[:-1]
    return pattern


def _json_literal(value: Any) -> str:
    return escape_literal(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


class SchemaCompiler:
    """Translate a JSON Schema into an anchored regex over its compact-or-spaced JSON text."""

    def __init__(self, root: Mapping[str, Any], whitespace: str = "") -> None:
        self.root = root
        self.ws = whitespace

    def compile(self) -> str:
        return self._node(self.root, depth=0)

    def _resolve(self, ref: str) -> Mapping[str, Any]:
        for prefix in ("#/$defs/", "#/definitions/"):
            if ref.startswith(prefix):
                bucket = self.root.get(prefix[2:-1], {})
                name = ref[len(prefix) :]
                if name in bucket:
                    target = bucket[name]
                    if not isinstance(target, Mapping):
                        raise SchemaError(f"$ref target {ref} is not an object")
                    return target
        raise SchemaError(f"unresolvable $ref {ref!r}")

    def _node(self, schema: Any, depth: int) -> str:
        if depth > MAX_REF_DEPTH:
            raise SchemaError("schema nesting/recursion exceeds the supported depth")
        if schema is True or schema == {}:
            raise SchemaError("unconstrained schemas ({} / true) are not supported; use a grammar")
        if not isinstance(schema, Mapping):
            raise SchemaError(f"schema must be an object, got {type(schema).__name__}")
        unknown = set(schema) - _KEYWORDS - _ANNOTATIONS
        if unknown:
            raise SchemaError(f"unsupported keywords: {sorted(unknown)}")
        if schema.get("additionalProperties", False) not in (False,) and "properties" in schema:
            raise SchemaError("only closed objects (additionalProperties: false) are supported")
        if "$ref" in schema:
            return self._node(self._resolve(schema["$ref"]), depth + 1)
        if "const" in schema:
            return _json_literal(schema["const"])
        if "enum" in schema:
            values = schema["enum"]
            if not isinstance(values, list) or not values:
                raise SchemaError("enum must be a non-empty list")
            return "(?:" + "|".join(_json_literal(v) for v in values) + ")"
        for key in ("anyOf", "oneOf"):
            if key in schema:
                options = schema[key]
                if not isinstance(options, list) or not options:
                    raise SchemaError(f"{key} must be a non-empty list")
                return "(?:" + "|".join(self._node(o, depth + 1) for o in options) + ")"
        kind = schema.get("type")
        if kind is None:
            if "properties" in schema:
                kind = "object"
            elif "items" in schema:
                kind = "array"
            else:
                raise SchemaError("schema has no 'type'")
        if isinstance(kind, list):
            if not kind:
                raise SchemaError("type list must be non-empty")
            parts = [self._typed(dict(schema, type=k), k, depth) for k in kind]
            return "(?:" + "|".join(parts) + ")"
        return self._typed(schema, kind, depth)

    def _typed(self, schema: Mapping[str, Any], kind: Any, depth: int) -> str:
        if kind == "string":
            return self._string(schema)
        if kind == "integer":
            return INTEGER
        if kind == "number":
            return NUMBER
        if kind == "boolean":
            return BOOLEAN
        if kind == "null":
            return NULL
        if kind == "array":
            return self._array(schema, depth)
        if kind == "object":
            return self._object(schema, depth)
        raise SchemaError(f"unsupported type {kind!r}")

    def _string(self, schema: Mapping[str, Any]) -> str:
        if "format" in schema:
            fmt = schema["format"]
            if fmt not in FORMATS:
                raise SchemaError(f"unsupported string format {fmt!r}")
            return FORMATS[fmt]
        if "pattern" in schema:
            return '"(?:' + _strip_anchors(str(schema["pattern"])) + ')"'
        low = int(schema.get("minLength", 0))
        high = schema.get("maxLength")
        if high is None:
            quant = "*" if low == 0 else ("+" if low == 1 else f"{{{low},}}")
        else:
            if int(high) < low:
                raise SchemaError("maxLength < minLength")
            quant = f"{{{low},{int(high)}}}"
        return f'"{STRING_CHAR}{quant}"'

    def _array(self, schema: Mapping[str, Any], depth: int) -> str:
        items = schema.get("items")
        if items is None:
            raise SchemaError("arrays require an 'items' schema")
        item = self._node(items, depth + 1)
        low = int(schema.get("minItems", 0))
        high_raw = schema.get("maxItems")
        high = None if high_raw is None else int(high_raw)
        if high is not None and high < low:
            raise SchemaError("maxItems < minItems")
        ws = self.ws
        sep = f"{ws},{ws}"
        if high == 0:
            return rf"\[{ws}\]"
        rest_low = max(low - 1, 0)
        rest = f"{{{rest_low},}}" if high is None else f"{{{rest_low},{high - 1}}}"
        body = f"{item}(?:{sep}{item}){rest}"
        if low == 0:
            body = f"(?:{body})?"
        return rf"\[{ws}{body}{ws}\]"

    def _object(self, schema: Mapping[str, Any], depth: int) -> str:
        props = schema.get("properties", {})
        if not isinstance(props, Mapping):
            raise SchemaError("properties must be an object")
        required = set(schema.get("required", []))
        missing = required - set(props)
        if missing:
            raise SchemaError(f"required properties without schemas: {sorted(missing)}")
        ws = self.ws
        names = list(props)
        kvs = [f"{_json_literal(name)}{ws}:{ws}{self._node(props[name], depth + 1)}" for name in names]
        sep = f"{ws},{ws}"
        n = len(names)
        # tail[i]: properties i.. after at least one property has been emitted (linear size).
        tail = [""] * (n + 1)
        for i in range(n - 1, -1, -1):
            piece = f"{sep}{kvs[i]}"
            tail[i] = (piece if names[i] in required else f"(?:{piece})?") + tail[i + 1]
        # head[i]: properties i.. when nothing has been emitted yet (quadratic worst case).
        head = [""] * (n + 1)
        for i in range(n - 1, -1, -1):
            emitted = kvs[i] + tail[i + 1]
            if names[i] in required:
                head[i] = emitted
            else:
                head[i] = f"(?:{emitted}|{head[i + 1]})" if head[i + 1] else f"(?:{emitted})?"
        return rf"\{{{ws}{head[0]}{ws}\}}"


def schema_to_regex(schema: Mapping[str, Any], whitespace: str = "") -> str:
    """Compile ``schema`` into an anchored regex. ``whitespace`` is emitted around separators."""
    return SchemaCompiler(schema, whitespace).compile()


# ------------------------------------------------------------------------- reference validator


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def validate(instance: Any, schema: Mapping[str, Any], root: Mapping[str, Any] | None = None) -> bool:
    """Validate ``instance`` against the supported schema subset (closed objects)."""
    root = schema if root is None else root
    if "$ref" in schema:
        return validate(instance, SchemaCompiler(root)._resolve(schema["$ref"]), root)
    if "const" in schema:
        return bool(instance == schema["const"]) and type(instance) is type(schema["const"])
    if "enum" in schema:
        return any(instance == v and type(instance) is type(v) for v in schema["enum"])
    for key in ("anyOf", "oneOf"):
        if key in schema:
            hits = sum(validate(instance, o, root) for o in schema[key])
            return hits >= 1 if key == "anyOf" else hits == 1
    kind = schema.get("type", "object" if "properties" in schema else "array")
    kinds = kind if isinstance(kind, list) else [kind]
    return any(_validate_typed(instance, schema, k, root) for k in kinds)


def _validate_typed(instance: Any, schema: Mapping[str, Any], kind: str, root: Mapping[str, Any]) -> bool:
    if kind == "string":
        if not isinstance(instance, str):
            return False
        if "format" in schema:
            return re.fullmatch(FORMATS[schema["format"]][1:-1], instance) is not None
        if "pattern" in schema:
            return re.fullmatch(_strip_anchors(str(schema["pattern"])), instance) is not None
        size = len(instance.encode("utf-8"))
        return size >= int(schema.get("minLength", 0)) and size <= int(schema.get("maxLength", size))
    if kind == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if kind == "number":
        return _is_number(instance)
    if kind == "boolean":
        return isinstance(instance, bool)
    if kind == "null":
        return instance is None
    if kind == "array":
        if not isinstance(instance, list):
            return False
        if len(instance) < int(schema.get("minItems", 0)):
            return False
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            return False
        return all(validate(x, schema["items"], root) for x in instance)
    if kind == "object":
        if not isinstance(instance, dict):
            return False
        props = schema.get("properties", {})
        if set(instance) - set(props):
            return False
        if set(schema.get("required", [])) - set(instance):
            return False
        return all(validate(v, props[k], root) for k, v in instance.items())
    return False


def validate_text(text: str, schema: Mapping[str, Any]) -> bool:
    """Parse ``text`` as JSON and validate it; malformed JSON is simply invalid."""
    try:
        instance = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return False
    return validate(instance, schema)
