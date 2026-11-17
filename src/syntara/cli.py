"""``syntara`` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from syntara.api import create_app
from syntara.bench import run_all
from syntara.corpus import PERSON_SCHEMA, demo_bundle
from syntara.decode import generate, score_choices
from syntara.earley import JSON_GRAMMAR, GrammarConstraint
from syntara.index import RegexConstraint
from syntara.regex import compile_regex
from syntara.schema import schema_to_regex, validate_text

app = typer.Typer(
    name="syntara",
    help="Syntara: grammar-constrained structured decoding (JSON Schema / regex / EBNF).",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _load_schema(path: Path | None) -> dict[str, object]:
    if path is None:
        return PERSON_SCHEMA
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise typer.BadParameter("schema file must contain a JSON object")
    return loaded


@app.command("compile")
def compile_cmd(
    schema: Path | None = typer.Option(None, help="JSON Schema file (default: demo schema)"),
    regex: str | None = typer.Option(None, help="Compile a regex instead of a schema"),
) -> None:
    """Compile a schema/regex to a minimal DFA and print its statistics."""
    pattern = regex if regex is not None else schema_to_regex(_load_schema(schema))
    dfa = compile_regex(pattern)
    typer.echo(
        json.dumps(
            {
                "regex": pattern,
                "dfa_states": dfa.num_states,
                "dfa_transitions": dfa.num_transitions,
                "accepting_states": len(dfa.accepting),
            },
            indent=2,
        )
    )


@app.command("generate")
def generate_cmd(
    schema: Path | None = typer.Option(None, help="JSON Schema file (default: demo schema)"),
    grammar: Path | None = typer.Option(None, help="EBNF grammar file (overrides --schema)"),
    json_grammar: bool = typer.Option(False, "--json-grammar", help="Use the built-in JSON CFG"),
    unconstrained: bool = typer.Option(False, help="Disable masking (for comparison)"),
    max_tokens: int = typer.Option(96, min=1),
    temperature: float = typer.Option(0.0, min=0.0),
    seed: int = typer.Option(0),
) -> None:
    """Generate text from the demo LM under a schema, grammar, or no constraint."""
    bundle = demo_bundle()
    vocab = bundle.tokenizer.vocab
    schema_obj = _load_schema(schema)
    if unconstrained:
        gen = generate(bundle.lm, vocab, None, max_tokens=max_tokens, temperature=temperature, seed=seed)
    elif grammar is not None or json_grammar:
        text = grammar.read_text(encoding="utf-8") if grammar is not None else JSON_GRAMMAR
        gen = generate(
            bundle.lm,
            vocab,
            GrammarConstraint(text, vocab),
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
    else:
        gen = generate(
            bundle.lm,
            vocab,
            RegexConstraint.from_schema(schema_obj, vocab),
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
    typer.echo(
        json.dumps(
            {
                "text": gen.text,
                "finished": gen.finished,
                "steps": gen.steps,
                "schema_valid": validate_text(gen.text, schema_obj),
                "mean_allowed_fraction": round(gen.mean_allowed_fraction, 4),
            },
            indent=2,
        )
    )


@app.command("score")
def score_cmd(
    choices: list[str] = typer.Argument(..., help="Candidate answers"),
    prompt: str = typer.Option("", help="Conditioning prefix"),
) -> None:
    """Rank a closed set of answers by exact log-likelihood under the demo LM."""
    bundle = demo_bundle()
    for s in score_choices(bundle.lm, bundle.tokenizer, choices, prompt):
        typer.echo(f"{s.log_prob:10.4f}  {s.mean_log_prob:8.4f}  {s.choice}")


@app.command("bench")
def bench_cmd(
    samples: int = typer.Option(20, min=1),
    large_vocab: int = typer.Option(16384, min=257, help="Size of the stress-test vocabulary"),
    output: Path | None = typer.Option(None, help="Write the JSON report here"),
) -> None:
    """Run the native-vs-Python index and validity benchmarks."""
    report = run_all(samples=samples, large_vocab=large_vocab)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    idx, val = report["index"], report["validity"]
    table = Table(title="Syntara benchmark")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("DFA states / transitions", f"{idx['dfa_states']} / {idx['dfa_transitions']}")
    table.add_row(
        "Vocabulary / allowed (state, token) pairs", f"{idx['vocab_size']} / {idx['allowed_pairs']}"
    )
    table.add_row(
        "Index build native / python (ms)",
        f"{idx['native_seconds'] * 1e3:.2f} / {idx['python_seconds'] * 1e3:.2f}",
    )
    table.add_row("Speedup / bit-exact", f"{idx['speedup']:.1f}x / {idx['bit_exact']}")
    big = report["index_large_vocab"]
    table.add_row(
        f"Index build @ {big['vocab_size']} tokens native / python (ms)",
        f"{big['native_seconds'] * 1e3:.2f} / {big['python_seconds'] * 1e3:.2f}",
    )
    table.add_row("Large-vocab speedup / bit-exact", f"{big['speedup']:.1f}x / {big['bit_exact']}")
    n = val["samples"]
    table.add_row("Schema-valid: unconstrained", f"{val['unconstrained_valid']}/{n}")
    table.add_row("Schema-valid: DFA-constrained", f"{val['schema_constrained_valid']}/{n}")
    table.add_row(
        "Parseable JSON: Earley-constrained (truncated)",
        f"{val['grammar_constrained_json']}/{n} ({val['grammar_truncated']})",
    )
    console.print(table)


@app.command("serve")
def serve_cmd(host: str = "127.0.0.1", port: int = 8090) -> None:  # pragma: no cover - blocking
    """Serve the HTTP API."""
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
