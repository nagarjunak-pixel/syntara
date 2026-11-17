"""FastAPI service: compile constraints and run constrained generation on the demo model."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from syntara import __version__
from syntara.corpus import demo_bundle
from syntara.decode import Constraint, Generation, generate, score_choices
from syntara.earley import GrammarConstraint, GrammarError
from syntara.index import RegexConstraint
from syntara.lm import NGramLM
from syntara.regex import RegexError, compile_regex
from syntara.schema import SchemaError, schema_to_regex, validate_text
from syntara.tokenizer import Vocabulary


class ConstraintSpec(BaseModel):
    """Exactly one of ``json_schema``, ``regex``, ``grammar`` or ``choices``."""

    json_schema: dict[str, Any] | None = None
    regex: str | None = None
    grammar: str | None = None
    choices: list[str] | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ConstraintSpec:
        given = [x for x in (self.json_schema, self.regex, self.grammar, self.choices) if x is not None]
        if len(given) != 1:
            raise ValueError("provide exactly one of json_schema, regex, grammar, choices")
        return self


class CompileResponse(BaseModel):
    kind: str
    regex: str | None
    dfa_states: int | None
    dfa_transitions: int | None
    grammar_rules: int | None


class GenerateRequest(ConstraintSpec):
    prompt: str = ""
    max_tokens: int = Field(default=96, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 0


class GenerateResponse(BaseModel):
    text: str
    finished: bool
    steps: int
    mean_allowed_fraction: float
    schema_valid: bool | None


class ScoreRequest(BaseModel):
    choices: list[str] = Field(min_length=1)
    prompt: str = ""


class ScoreItem(BaseModel):
    choice: str
    log_prob: float
    mean_log_prob: float


def _regex_for(spec: ConstraintSpec) -> str | None:
    if spec.json_schema is not None:
        return schema_to_regex(spec.json_schema)
    return spec.regex


def _run[S](
    lm: NGramLM,
    vocab: Vocabulary,
    constraint: Constraint[S],
    prompt: list[int],
    req: GenerateRequest,
) -> Generation:
    return generate(
        lm,
        vocab,
        constraint,
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        seed=req.seed,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Syntara", version=__version__)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/compile", response_model=CompileResponse)
    def compile_constraint(spec: ConstraintSpec) -> CompileResponse:
        try:
            if spec.grammar is not None:
                gc = GrammarConstraint(spec.grammar, demo_bundle().tokenizer.vocab)
                return CompileResponse(
                    kind="grammar",
                    regex=None,
                    dfa_states=None,
                    dfa_transitions=None,
                    grammar_rules=gc.grammar.num_rules,
                )
            if spec.choices is not None:
                rc = RegexConstraint.from_choices(spec.choices, demo_bundle().tokenizer.vocab)
                dfa = rc.index.dfa
                return CompileResponse(
                    kind="choices",
                    regex=None,
                    dfa_states=dfa.num_states,
                    dfa_transitions=dfa.num_transitions,
                    grammar_rules=None,
                )
            pattern = _regex_for(spec)
            assert pattern is not None
            dfa = compile_regex(pattern)
        except (RegexError, SchemaError, GrammarError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return CompileResponse(
            kind="json_schema" if spec.json_schema is not None else "regex",
            regex=pattern,
            dfa_states=dfa.num_states,
            dfa_transitions=dfa.num_transitions,
            grammar_rules=None,
        )

    @app.post("/v1/generate", response_model=GenerateResponse)
    def generate_endpoint(req: GenerateRequest) -> GenerateResponse:
        bundle = demo_bundle()
        vocab = bundle.tokenizer.vocab
        constraint: RegexConstraint | GrammarConstraint
        try:
            if req.grammar is not None:
                constraint = GrammarConstraint(req.grammar, vocab)
            elif req.choices is not None:
                constraint = RegexConstraint.from_choices(req.choices, vocab)
            else:
                pattern = _regex_for(req)
                assert pattern is not None
                constraint = RegexConstraint.from_regex(pattern, vocab)
        except (RegexError, SchemaError, GrammarError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        prompt = bundle.tokenizer.encode(req.prompt) if req.prompt else []
        if isinstance(constraint, GrammarConstraint):
            gen = _run(bundle.lm, vocab, constraint, prompt, req)
        else:
            gen = _run(bundle.lm, vocab, constraint, prompt, req)
        valid = validate_text(gen.text, req.json_schema) if req.json_schema is not None else None
        return GenerateResponse(
            text=gen.text,
            finished=gen.finished,
            steps=gen.steps,
            mean_allowed_fraction=gen.mean_allowed_fraction,
            schema_valid=valid,
        )

    @app.post("/v1/score", response_model=list[ScoreItem])
    def score_endpoint(req: ScoreRequest) -> list[ScoreItem]:
        bundle = demo_bundle()
        ranked = score_choices(bundle.lm, bundle.tokenizer, req.choices, req.prompt)
        return [
            ScoreItem(choice=s.choice, log_prob=s.log_prob, mean_log_prob=s.mean_log_prob) for s in ranked
        ]

    return app
