from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from syntara import __version__
from syntara.api import create_app
from syntara.bench import benchmark_index, benchmark_validity, substring_vocabulary
from syntara.cli import app as cli_app
from syntara.corpus import PERSON_SCHEMA, demo_bundle
from syntara.decode import generate, score_choices
from syntara.earley import ARITHMETIC_GRAMMAR, GrammarConstraint
from syntara.index import RegexConstraint
from syntara.kernels import KernelBridge
from syntara.schema import validate_text
from syntara.tokenizer import Vocabulary

BUNDLE = demo_bundle()
VOCAB = BUNDLE.tokenizer.vocab


# ------------------------------------------------------------------------------- decoding


def test_constrained_generation_is_always_schema_valid() -> None:
    constraint = RegexConstraint.from_schema(PERSON_SCHEMA, VOCAB)
    for seed in range(8):
        gen = generate(BUNDLE.lm, VOCAB, constraint, max_tokens=128, temperature=0.9, seed=seed)
        assert gen.finished
        assert validate_text(gen.text, PERSON_SCHEMA), gen.text
        assert 0 < gen.mean_allowed_fraction < 1


def test_greedy_native_and_python_paths_agree() -> None:
    native = RegexConstraint.from_schema(PERSON_SCHEMA, VOCAB, bridge=KernelBridge(use_native=True))
    python = RegexConstraint.from_schema(PERSON_SCHEMA, VOCAB, bridge=KernelBridge(use_native=False))
    a = generate(BUNDLE.lm, VOCAB, native, bridge=KernelBridge(use_native=True))
    b = generate(BUNDLE.lm, VOCAB, python, bridge=KernelBridge(use_native=False))
    assert a.token_ids == b.token_ids
    assert a.finished and validate_text(a.text, PERSON_SCHEMA)


def test_unconstrained_generation_is_deterministic_per_seed() -> None:
    g1 = generate(BUNDLE.lm, VOCAB, None, max_tokens=40, temperature=0.8, seed=5)
    g2 = generate(BUNDLE.lm, VOCAB, None, max_tokens=40, temperature=0.8, seed=5)
    assert g1 == g2 or (g1.token_ids == g2.token_ids and g1.text == g2.text)
    greedy = generate(BUNDLE.lm, VOCAB, None, max_tokens=200)
    assert greedy.mean_allowed_fraction == 1.0
    with pytest.raises(ValueError):
        generate(BUNDLE.lm, VOCAB, None, temperature=-1.0)


def test_grammar_constrained_generation() -> None:
    gc = GrammarConstraint(ARITHMETIC_GRAMMAR, VOCAB)
    gen = generate(BUNDLE.lm, VOCAB, gc, max_tokens=12, temperature=1.0, seed=2)
    assert gen.steps <= 12
    gc2 = GrammarConstraint('root ::= "1" "+" "2"', VOCAB)
    out = generate(BUNDLE.lm, VOCAB, gc2, max_tokens=10)
    assert out.text == "1+2" and out.finished


def test_generation_stops_when_vocabulary_cannot_continue() -> None:
    vocab = Vocabulary(tokens=(b"a", b""), eos_id=1)
    from syntara.lm import NGramLM

    lm = NGramLM(vocab_size=2, order=1).fit([[0, 1]])
    gen = generate(lm, vocab, RegexConstraint.from_regex("b", vocab))
    assert not gen.finished and gen.steps == 0


def test_choice_constraint_and_exact_scoring() -> None:
    choices = ["engineer", "manager", "analyst"]
    rc = RegexConstraint.from_choices(choices, VOCAB)
    gen = generate(BUNDLE.lm, VOCAB, rc, prompt=BUNDLE.tokenizer.encode('{"role":"'))
    assert gen.text in choices and gen.finished
    ranked = score_choices(BUNDLE.lm, BUNDLE.tokenizer, choices, prompt="Asha Rao is a 30 year old ")
    assert [s.choice for s in ranked] != [] and len(ranked) == 3
    assert ranked[0].log_prob >= ranked[1].log_prob >= ranked[2].log_prob
    assert all(s.num_tokens >= 2 for s in ranked)
    assert score_choices(BUNDLE.lm, BUNDLE.tokenizer, ["x"])[0].choice == "x"
    with pytest.raises(ValueError):
        score_choices(BUNDLE.lm, BUNDLE.tokenizer, [])


# ------------------------------------------------------------------------------ benchmarks


def test_benchmarks_report_consistent_numbers() -> None:
    idx = benchmark_index(repeats=1)
    assert idx.bit_exact and idx.speedup > 1.0
    big = benchmark_index(substring_vocabulary(BUNDLE.corpus, 1024), repeats=1)
    assert big.vocab_size == 1024 and big.bit_exact
    val = benchmark_validity(samples=3, max_tokens=64)
    assert val.schema_constrained_valid == 3
    assert val.grammar_constrained_json + val.grammar_truncated == 3
    small = substring_vocabulary(["ab"], 258)
    assert len(small) == 258 and small.tokens[256] == b"ab"


# ------------------------------------------------------------------------------------- API


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_api_health_and_compile(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "version": __version__}
    r = client.post("/v1/compile", json={"json_schema": PERSON_SCHEMA}).json()
    assert r["kind"] == "json_schema" and r["dfa_states"] > 100
    r = client.post("/v1/compile", json={"regex": "(a|b)*abb"}).json()
    assert r["dfa_states"] == 4
    r = client.post("/v1/compile", json={"grammar": ARITHMETIC_GRAMMAR}).json()
    assert r["kind"] == "grammar" and r["grammar_rules"] > 3
    r = client.post("/v1/compile", json={"choices": ["yes", "no"]}).json()
    assert r["kind"] == "choices"
    assert client.post("/v1/compile", json={"regex": "("}).status_code == 422
    assert client.post("/v1/compile", json={"grammar": "root ::= nope"}).status_code == 422
    assert client.post("/v1/compile", json={"regex": "a", "choices": ["b"]}).status_code == 422
    assert client.post("/v1/compile", json={}).status_code == 422


def test_api_generate_and_score(client: TestClient) -> None:
    r = client.post("/v1/generate", json={"json_schema": PERSON_SCHEMA, "temperature": 0.7, "seed": 3}).json()
    assert r["schema_valid"] is True and r["finished"] is True
    r = client.post("/v1/generate", json={"regex": "[0-9]{3}", "prompt": "age "}).json()
    assert len(r["text"]) == 3 and r["schema_valid"] is None
    r = client.post("/v1/generate", json={"grammar": 'root ::= "ok"'}).json()
    assert r["text"] == "ok"
    r = client.post("/v1/generate", json={"choices": ["true", "false"]}).json()
    assert r["text"] in ("true", "false")
    assert client.post("/v1/generate", json={"json_schema": {"type": "tuple"}}).status_code == 422
    ranked = client.post("/v1/score", json={"choices": ["engineer", "zzz"]}).json()
    assert ranked[0]["choice"] == "engineer"


# ------------------------------------------------------------------------------------- CLI

runner = CliRunner()


def test_cli_compile(tmp_path: Path) -> None:
    res = runner.invoke(cli_app, ["compile"])
    assert res.exit_code == 0 and json.loads(res.stdout)["dfa_states"] > 100
    res = runner.invoke(cli_app, ["compile", "--regex", "(a|b)*abb"])
    assert json.loads(res.stdout)["dfa_states"] == 4
    schema = tmp_path / "s.json"
    schema.write_text(json.dumps({"type": "boolean"}))
    res = runner.invoke(cli_app, ["compile", "--schema", str(schema)])
    assert json.loads(res.stdout)["regex"] == "(?:true|false)"
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    assert runner.invoke(cli_app, ["compile", "--schema", str(bad)]).exit_code != 0


def test_cli_generate_variants(tmp_path: Path) -> None:
    res = runner.invoke(cli_app, ["generate", "--temperature", "0.5", "--seed", "1"])
    assert res.exit_code == 0 and json.loads(res.stdout)["schema_valid"] is True
    res = runner.invoke(cli_app, ["generate", "--unconstrained", "--max-tokens", "30"])
    assert res.exit_code == 0
    res = runner.invoke(cli_app, ["generate", "--json-grammar", "--max-tokens", "120"])
    assert res.exit_code == 0 and "text" in json.loads(res.stdout)
    grammar = tmp_path / "g.ebnf"
    grammar.write_text('root ::= "4" "2"')
    res = runner.invoke(cli_app, ["generate", "--grammar", str(grammar)])
    assert json.loads(res.stdout)["text"] == "42"


def test_cli_score_and_bench(tmp_path: Path) -> None:
    res = runner.invoke(cli_app, ["score", "engineer", "analyst", "--prompt", "Ravi Ng is a 40 year old "])
    assert res.exit_code == 0 and len(res.stdout.strip().splitlines()) == 2
    out = tmp_path / "bench.json"
    res = runner.invoke(cli_app, ["bench", "--samples", "2", "--large-vocab", "600", "--output", str(out)])
    assert res.exit_code == 0, res.stdout
    report = json.loads(out.read_text())
    assert report["index"]["bit_exact"] and report["index_large_vocab"]["vocab_size"] == 600
