from __future__ import annotations

import json
import random
from typing import Any

import pytest

from syntara.corpus import PERSON_SCHEMA, _record
from syntara.regex import compile_regex
from syntara.schema import SchemaError, schema_to_regex, validate, validate_text


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _ordered(rec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    order = list(schema["properties"])
    known = [k for k in order if k in rec]
    extra = [k for k in rec if k not in order]
    return {k: rec[k] for k in known + extra}


def test_person_schema_regex_agrees_with_reference_validator() -> None:
    dfa = compile_regex(schema_to_regex(PERSON_SCHEMA))
    rng = random.Random(1)
    seen_valid = seen_invalid = 0
    for _ in range(300):
        rec = _ordered(_record(rng), PERSON_SCHEMA)
        ok = validate(rec, PERSON_SCHEMA)
        assert dfa.matches(_compact(rec)) == ok, rec
        seen_valid += ok
        seen_invalid += not ok
    assert seen_valid > 30 and seen_invalid > 30


def test_whitespace_policy() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "array", "items": {"type": "boolean"}}},
    }
    spaced = compile_regex(schema_to_regex(schema, whitespace="[ ]?"))
    assert spaced.matches(json.dumps({"a": 1, "b": [True, False]}))
    assert spaced.matches(_compact({"a": 1, "b": [True, False]}))
    compact = compile_regex(schema_to_regex(schema))
    assert not compact.matches(json.dumps({"a": 1, "b": [True]}))


@pytest.mark.parametrize(
    ("schema", "good", "bad"),
    [
        ({"type": "integer"}, ["0", "-12", "907"], ["01", "1.5", "-", ""]),
        ({"type": "number"}, ["0", "-1.25", "3e10", "2.5E-3"], ["1.", ".5", "1e", "+1"]),
        ({"type": "boolean"}, ["true", "false"], ["True", "1"]),
        ({"type": "null"}, ["null"], ["None", ""]),
        ({"type": "string"}, ['""', '"a\\"b"', '"\\u00e9"', '"é"'], ['"a', '"\\x"', '"\n"']),
        ({"type": "string", "minLength": 2, "maxLength": 3}, ['"ab"', '"abc"'], ['"a"', '"abcd"']),
        ({"type": "string", "minLength": 1}, ['"a"'], ['""']),
        ({"type": "string", "minLength": 3}, ['"abc"', '"abcdef"'], ['"ab"']),
        ({"type": "string", "pattern": "^[A-Z]{2}[0-9]+$"}, ['"AB12"'], ['"ab12"', '"AB"']),
        ({"type": "string", "format": "date"}, ['"2026-11-17"'], ['"2026-13-01"']),
        ({"type": "string", "format": "time"}, ['"13:45:00Z"', '"09:00:00.5+04:00"'], ['"24:00:00"']),
        ({"type": "string", "format": "date-time"}, ['"2026-11-17T09:18:18Z"'], ['"2026-11-17 09:18:18"']),
        ({"type": "string", "format": "uuid"}, ['"123e4567-e89b-12d3-a456-426614174000"'], ['"123"']),
        ({"type": "string", "format": "email"}, ['"a.b@corp.ae"'], ['"a@b"', '"@x.io"']),
        ({"enum": ["red", 3, None, True]}, ['"red"', "3", "null", "true"], ['"blue"', "false"]),
        ({"const": {"k": [1, 2]}}, ['{"k":[1,2]}'], ['{"k":[1]}']),
        ({"type": ["integer", "null"]}, ["4", "null"], ['"4"']),
        ({"anyOf": [{"type": "integer"}, {"type": "string", "maxLength": 1}]}, ["7", '"x"'], ['"xy"']),
        ({"oneOf": [{"const": 1}, {"const": 2}]}, ["1", "2"], ["3"]),
        ({"type": "array", "items": {"type": "integer"}}, ["[]", "[1]", "[1,2,3]"], ["[1,]", "[,]"]),
        (
            {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 3},
            ["[1,2]", "[1,2,3]"],
            ["[1]", "[1,2,3,4]"],
        ),
        ({"type": "array", "items": {"type": "integer"}, "minItems": 1}, ["[5]"], ["[]"]),
        ({"type": "array", "items": {"type": "integer"}, "maxItems": 0}, ["[]"], ["[1]"]),
        ({"items": {"type": "boolean"}}, ["[true]"], ["[1]"]),
        ({"properties": {"x": {"type": "integer"}}, "required": ["x"]}, ['{"x":1}'], ["{}"]),
        ({"type": "object"}, ["{}"], ['{"a":1}']),
        (
            {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            ["{}", '{"a":1}', '{"b":2}', '{"a":1,"b":2}'],
            ['{"b":2,"a":1}', '{"a":1,}', '{,"b":2}'],
        ),
        (
            {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}, "c": {"type": "integer"}},
                "required": ["b"],
            },
            ['{"b":1}', '{"a":1,"b":2}', '{"b":2,"c":3}', '{"a":1,"b":2,"c":3}'],
            ['{"a":1}', '{"a":1,"c":3}', "{}"],
        ),
        (
            {
                "$defs": {"pt": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
                "$ref": "#/$defs/pt",
            },
            ["[1,2.5]"],
            ["[1]"],
        ),
        (
            {
                "definitions": {"flag": {"type": "boolean"}},
                "type": "object",
                "properties": {"f": {"$ref": "#/definitions/flag"}},
                "required": ["f"],
            },
            ['{"f":true}'],
            ['{"f":1}'],
        ),
    ],
)
def test_schema_keywords(schema: dict[str, Any], good: list[str], bad: list[str]) -> None:
    dfa = compile_regex(schema_to_regex(schema))
    for text in good:
        assert dfa.matches(text), (schema, text)
        assert validate_text(text, schema), (schema, text)
    for text in bad:
        assert not dfa.matches(text), (schema, text)


@pytest.mark.parametrize(
    "schema",
    [
        {},
        True,
        [],
        {"type": "string", "minimum": 3},
        {"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": True},
        {"$ref": "#/$defs/missing"},
        {"$ref": "http://example.com/schema"},
        {"$defs": {"x": 5}, "$ref": "#/$defs/x"},
        {"$defs": {"n": {"type": "array", "items": {"$ref": "#/$defs/n"}}}, "$ref": "#/$defs/n"},
        {"enum": []},
        {"anyOf": []},
        {"description": "no type"},
        {"type": []},
        {"type": "tuple"},
        {"type": "string", "format": "ipv6"},
        {"type": "string", "minLength": 4, "maxLength": 2},
        {"type": "array"},
        {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 1},
        {"type": "object", "properties": ["a"]},
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["b"]},
    ],
)
def test_unsupported_or_malformed_schemas_raise(schema: Any) -> None:
    with pytest.raises(SchemaError):
        schema_to_regex(schema)


def test_reference_validator_branches() -> None:
    assert not validate(True, {"const": 1})
    assert not validate(1, {"enum": [True]})
    assert not validate(1, {"oneOf": [{"type": "integer"}, {"type": "number"}]})
    assert validate(1.5, {"oneOf": [{"type": "integer"}, {"type": "number"}]})
    assert not validate(True, {"type": "integer"})
    assert not validate(True, {"type": "number"})
    assert not validate(1, {"type": "string"})
    assert not validate("x", {"type": "array", "items": {"type": "integer"}})
    assert not validate([1, 2], {"type": "array", "items": {"type": "integer"}, "maxItems": 1})
    assert not validate([], {"type": "array", "items": {"type": "integer"}, "minItems": 1})
    assert not validate([], {"type": "object"})
    assert not validate({"a": 1}, {"type": "object", "properties": {"a": {"type": "string"}}})
    assert not validate("x", {"type": "mystery"})
    assert validate(
        {"p": [1, 2]},
        {
            "$defs": {"v": {"type": "array", "items": {"type": "integer"}}},
            "properties": {"p": {"$ref": "#/$defs/v"}},
        },
    )
    assert not validate_text("{not json", {"type": "object"})
    assert not validate_text("[" * 5000 + "]" * 5000, {"type": "array", "items": {"type": "integer"}})
