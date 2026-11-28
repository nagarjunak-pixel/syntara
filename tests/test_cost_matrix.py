"""Unit, API, and Hypothesis property tests for CostMatrixDecisionHead (arXiv:2611.34858)."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from syntara.api import create_app
from syntara.cost_matrix import (
    CostMatrixDecisionHead,
    expected_calibration_error,
    kl_divergence_nats,
)


def test_kl_divergence_and_ece_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        kl_divergence_nats([], [])
    with pytest.raises(ValueError, match="strictly positive"):
        kl_divergence_nats([0.5, 0.5], [1.0, 0.0])
    with pytest.raises(ValueError, match="strictly positive"):
        kl_divergence_nats([-0.1, 1.1], [0.5, 0.5])

    assert kl_divergence_nats([0.25, 0.75], [0.25, 0.75]) == pytest.approx(0.0, abs=1e-12)
    assert kl_divergence_nats([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2.0), rel=1e-9)

    with pytest.raises(ValueError, match="non-empty"):
        expected_calibration_error([], [])
    with pytest.raises(ValueError, match="num_bins"):
        expected_calibration_error([0.5], [True], num_bins=0)
    with pytest.raises(ValueError, match="confidence out of"):
        expected_calibration_error([1.2], [True])

    ece_zero = expected_calibration_error([1.0, 1.0, 0.0, 0.0], [True, True, False, False], num_bins=2)
    assert ece_zero == pytest.approx(0.0, abs=1e-12)


def test_cost_matrix_constructor_and_validation() -> None:
    with pytest.raises(ValueError, match="actions must be non-empty"):
        CostMatrixDecisionHead((), ())
    with pytest.raises(ValueError, match="rows must match"):
        CostMatrixDecisionHead(("a", "b"), ((1.0, 2.0),))
    with pytest.raises(ValueError, match="rectangular"):
        CostMatrixDecisionHead(("a", "b"), ((1.0, 2.0), (1.0,)))
    with pytest.raises(ValueError, match="non-negative"):
        CostMatrixDecisionHead(("a",), ((-1.0,),))
    with pytest.raises(ValueError, match="kl_weight"):
        CostMatrixDecisionHead(("a",), ((0.0,),), kl_weight=0.0)
    with pytest.raises(ValueError, match="shape must match"):
        CostMatrixDecisionHead(("a",), ((0.0, 1.0),), amount_scale_matrix=((0.0,),))
    with pytest.raises(ValueError, match="entries must be non-negative"):
        CostMatrixDecisionHead(("a",), ((0.0, 1.0),), amount_scale_matrix=((-0.1, 0.2),))

    default_scale_head = CostMatrixDecisionHead(("approve", "reject"), ((0.0, 10.0), (5.0, 0.0)))
    assert default_scale_head.cell_cost_usd(0, 0, 100.0) == pytest.approx(0.0)
    assert default_scale_head.cell_cost_usd(0, 1, 100.0) == pytest.approx(110.0)

    with pytest.raises(ValueError, match="amount_usd"):
        default_scale_head.cell_cost_usd(0, 0, -1.0)
    with pytest.raises(ValueError, match="outcome probabilities"):
        default_scale_head.decide([0.5], 100.0)
    with pytest.raises(ValueError, match="positive sum"):
        default_scale_head.decide([0.0, 0.0], 100.0)


def test_heteroscedastic_financial_ledger_savings() -> None:
    head = CostMatrixDecisionHead.default_financial_audit_head(kl_weight=50.0)
    # 88% clean, 7% compliance variance, 5% material misstatement:
    # Symmetric argmax chooses "auto_post" even on a $500,000 wire!
    high_value = head.decide((0.88, 0.07, 0.05), amount_usd=500_000.0)
    assert high_value.naive_argmax_action == "auto_post"
    assert high_value.selected_action != "auto_post"
    assert high_value.expected_savings_usd > 20_000.0
    assert sum(high_value.policy_distribution) == pytest.approx(1.0, abs=1e-9)

    low_value = head.decide((0.99, 0.008, 0.002), amount_usd=25.0)
    assert low_value.selected_action == "auto_post"

    with pytest.raises(ValueError, match="non-empty"):
        head.evaluate_ledger_stream([])
    with pytest.raises(ValueError, match="true_y"):
        head.evaluate_ledger_stream([((0.8, 0.1, 0.1), 100.0, 5)])

    events = [
        ((0.96, 0.03, 0.01), 40.0, 0),
        ((0.85, 0.08, 0.07), 400_000.0, 2),
        ((0.82, 0.12, 0.06), 250_000.0, 1),
        ((0.98, 0.01, 0.01), 80.0, 0),
    ]
    summary = head.evaluate_ledger_stream(events)
    assert summary.num_events == 4
    assert summary.cost_sensitive_loss_usd < summary.symmetric_argmax_loss_usd
    assert summary.realized_savings_usd > 300_000.0
    assert summary.savings_ratio > 0.9


@given(
    p0=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    p1=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    p2=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    amount=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=60)
def test_property_bayes_decision_never_exceeds_naive_expected_loss(
    p0: float,
    p1: float,
    p2: float,
    amount: float,
) -> None:
    head = CostMatrixDecisionHead.default_financial_audit_head(kl_weight=40.0)
    res = head.decide((p0, p1, p2), amount_usd=amount)
    assert res.expected_loss_usd <= res.naive_expected_loss_usd + 1e-9
    assert res.expected_savings_usd >= 0.0
    assert res.kl_to_uniform_nats >= 0.0
    assert sum(res.policy_distribution) == pytest.approx(1.0, abs=1e-9)


def test_fastapi_decision_endpoint() -> None:
    client = TestClient(create_app())
    ok = client.post(
        "/v1/decision",
        json={"outcome_probs": [0.85, 0.10, 0.05], "amount_usd": 300_000.0, "kl_weight": 50.0},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["naive_argmax_action"] == "auto_post"
    assert body["selected_action"] in ("escalate_audit", "block_settlement")
    assert body["expected_savings_usd"] > 10_000.0

    bad = client.post(
        "/v1/decision",
        json={"outcome_probs": [0.0, 0.0, 0.0], "amount_usd": 100.0},
    )
    assert bad.status_code == 422
