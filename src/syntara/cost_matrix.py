"""Amount-scaled financial cost-matrix decision head and KL-anchored calibration.

Implements closed-form institutional decision optimization over heteroscedastic
financial ledger streams (arXiv:2611.34858):
- Symmetric cross-entropy argmax treats a $50 variance and a $500,000 ledger
  misstatement identically.
- `CostMatrixDecisionHead` evaluates action-by-outcome institutional cost matrices
  scaled by transaction dollar magnitude `amount_usd`, selects minimum expected
  dollar-loss actions, computes closed-form KL-anchored variational action posteriors
  `q*(a) proportional to q0(a) * exp(-E_y[C(a, y, M)] / kl_weight)`, and audits
  Expected Calibration Error (ECE) on probability streams.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CostDecisionResult:
    """Optimal action, expected dollar loss, and KL-anchored policy diagnostics."""

    selected_action: str
    selected_index: int
    expected_loss_usd: float
    naive_argmax_action: str
    naive_expected_loss_usd: float
    expected_savings_usd: float
    kl_to_uniform_nats: float
    policy_distribution: tuple[float, ...]


@dataclass(frozen=True)
class LedgerStreamSummary:
    """Aggregate realized dollar loss comparison across a batch of ledger events."""

    num_events: int
    cost_sensitive_loss_usd: float
    symmetric_argmax_loss_usd: float
    realized_savings_usd: float
    savings_ratio: float
    ece: float


def kl_divergence_nats(p: list[float] | tuple[float, ...], q: list[float] | tuple[float, ...]) -> float:
    """Compute exact KL(p || q) in nats over normalized discrete distributions."""
    if len(p) != len(q) or not p:
        raise ValueError("Distributions p and q must be non-empty and equal length")
    total_p = sum(p)
    total_q = sum(q)
    if total_p <= 0.0 or total_q <= 0.0 or any(x < 0.0 for x in p) or any(y <= 0.0 for y in q):
        raise ValueError("p must be non-negative and q must be strictly positive")
    pn = [x / total_p for x in p]
    qn = [y / total_q for y in q]
    acc = 0.0
    for pi, qi in zip(pn, qn, strict=True):
        if pi > 0.0:
            acc += pi * math.log(pi / qi)
    return max(0.0, acc)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    num_bins: int = 10,
) -> float:
    """Compute binned Expected Calibration Error (ECE) over top-1 predictions."""
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidences and correct must be non-empty and equal length")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    n = len(confidences)
    bin_counts = [0] * num_bins
    bin_conf_sum = [0.0] * num_bins
    bin_acc_sum = [0.0] * num_bins
    for conf, is_ok in zip(confidences, correct, strict=True):
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"confidence out of [0, 1]: {conf}")
        idx = min(num_bins - 1, int(conf * num_bins))
        bin_counts[idx] += 1
        bin_conf_sum[idx] += conf
        bin_acc_sum[idx] += 1.0 if is_ok else 0.0
    ece = 0.0
    for count, c_sum, a_sum in zip(bin_counts, bin_conf_sum, bin_acc_sum, strict=True):
        if count > 0:
            ece += (count / n) * abs((a_sum / count) - (c_sum / count))
    return ece


class CostMatrixDecisionHead:
    """Amount-scaled minimum-expected-loss decision layer with KL policy anchoring."""

    def __init__(
        self,
        actions: tuple[str, ...],
        base_cost_matrix: tuple[tuple[float, ...], ...],
        amount_scale_matrix: tuple[tuple[float, ...], ...] | None = None,
        kl_weight: float = 25.0,
    ) -> None:
        if not actions:
            raise ValueError("actions must be non-empty")
        num_actions = len(actions)
        if len(base_cost_matrix) != num_actions:
            raise ValueError("base_cost_matrix rows must match len(actions)")
        num_outcomes = len(base_cost_matrix[0])
        if num_outcomes == 0 or any(len(row) != num_outcomes for row in base_cost_matrix):
            raise ValueError("base_cost_matrix must be rectangular and non-empty")
        if any(cell < 0.0 for row in base_cost_matrix for cell in row):
            raise ValueError("base_cost_matrix entries must be non-negative")
        if kl_weight <= 0.0:
            raise ValueError("kl_weight must be > 0")

        if amount_scale_matrix is not None:
            if len(amount_scale_matrix) != num_actions or any(
                len(row) != num_outcomes for row in amount_scale_matrix
            ):
                raise ValueError("amount_scale_matrix shape must match base_cost_matrix")
            if any(cell < 0.0 for row in amount_scale_matrix for cell in row):
                raise ValueError("amount_scale_matrix entries must be non-negative")
            self._scale = amount_scale_matrix
        else:
            self._scale = tuple(
                tuple(0.0 if a_idx == y_idx else 1.0 for y_idx in range(num_outcomes))
                for a_idx in range(num_actions)
            )

        self.actions = actions
        self.num_outcomes = num_outcomes
        self.base_cost_matrix = base_cost_matrix
        self.kl_weight = kl_weight

    @classmethod
    def default_financial_audit_head(cls, kl_weight: float = 50.0) -> CostMatrixDecisionHead:
        """Construct a 3-action x 3-outcome financial ledger gate (auto_post, review, block)."""
        actions = ("auto_post", "escalate_audit", "block_settlement")
        # Outcomes: 0 = clean_entry, 1 = compliance_variance, 2 = material_misstatement
        base_costs = (
            (0.0, 120.0, 600.0),
            (35.0, 25.0, 45.0),
            (180.0, 60.0, 10.0),
        )
        # Fraction of transaction principal lost under each (action, outcome) pair
        amount_scales = (
            (0.0, 0.12, 1.00),
            (0.0, 0.01, 0.02),
            (0.005, 0.002, 0.0),
        )
        return cls(
            actions=actions,
            base_cost_matrix=base_costs,
            amount_scale_matrix=amount_scales,
            kl_weight=kl_weight,
        )

    def cell_cost_usd(self, action_idx: int, outcome_idx: int, amount_usd: float) -> float:
        """Compute exact dollar loss C(a, y, M) for action `a`, outcome `y`, and amount `M`."""
        if amount_usd < 0.0:
            raise ValueError("amount_usd must be >= 0")
        base = self.base_cost_matrix[action_idx][outcome_idx]
        frac = self._scale[action_idx][outcome_idx]
        return base + frac * amount_usd

    def decide(
        self,
        outcome_probs: Sequence[float],
        amount_usd: float,
    ) -> CostDecisionResult:
        """Select Bayes-optimal action minimizing expected dollar loss with KL policy diagnostics."""
        if len(outcome_probs) != self.num_outcomes:
            raise ValueError(f"Expected {self.num_outcomes} outcome probabilities, got {len(outcome_probs)}")
        total = sum(outcome_probs)
        if total <= 0.0 or any(p < 0.0 for p in outcome_probs):
            raise ValueError("outcome_probs must be non-negative with positive sum")
        probs = [p / total for p in outcome_probs]

        expected_losses: list[float] = []
        for a_idx in range(len(self.actions)):
            loss = sum(
                probs[y_idx] * self.cell_cost_usd(a_idx, y_idx, amount_usd)
                for y_idx in range(self.num_outcomes)
            )
            expected_losses.append(loss)

        best_idx = min(range(len(self.actions)), key=lambda i: expected_losses[i])
        top_outcome = max(range(self.num_outcomes), key=lambda y: probs[y])
        naive_idx = min(top_outcome, len(self.actions) - 1)

        # Closed-form KL-anchored variational policy q*(a) proportional to exp(-L(a) / kl_weight)
        scaled = [-loss / self.kl_weight for loss in expected_losses]
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        z = sum(exps)
        q_dist = tuple(v / z for v in exps)
        uniform = tuple(1.0 / len(self.actions) for _ in self.actions)
        kl_nats = kl_divergence_nats(q_dist, uniform)

        naive_loss = expected_losses[naive_idx]
        best_loss = expected_losses[best_idx]
        return CostDecisionResult(
            selected_action=self.actions[best_idx],
            selected_index=best_idx,
            expected_loss_usd=best_loss,
            naive_argmax_action=self.actions[naive_idx],
            naive_expected_loss_usd=naive_loss,
            expected_savings_usd=max(0.0, naive_loss - best_loss),
            kl_to_uniform_nats=kl_nats,
            policy_distribution=q_dist,
        )

    def evaluate_ledger_stream(
        self,
        events: Sequence[tuple[Sequence[float], float, int]],
    ) -> LedgerStreamSummary:
        """Compare realized dollar losses across `(probs, amount_usd, true_outcome)` events."""
        if not events:
            raise ValueError("events must be non-empty")
        cost_loss = 0.0
        naive_loss = 0.0
        confs: list[float] = []
        correct_flags: list[bool] = []
        for probs, amount_usd, true_y in events:
            if not (0 <= true_y < self.num_outcomes):
                raise ValueError(f"true_y out of range: {true_y}")
            res = self.decide(probs, amount_usd)
            total = sum(probs)
            norm = [p / total for p in probs]
            pred_y = max(range(self.num_outcomes), key=lambda y: norm[y])
            naive_idx = min(pred_y, len(self.actions) - 1)

            cost_loss += self.cell_cost_usd(res.selected_index, true_y, amount_usd)
            naive_loss += self.cell_cost_usd(naive_idx, true_y, amount_usd)
            confs.append(norm[pred_y])
            correct_flags.append(pred_y == true_y)

        savings = naive_loss - cost_loss
        ratio = (savings / naive_loss) if naive_loss > 0.0 else 0.0
        return LedgerStreamSummary(
            num_events=len(events),
            cost_sensitive_loss_usd=cost_loss,
            symmetric_argmax_loss_usd=naive_loss,
            realized_savings_usd=savings,
            savings_ratio=ratio,
            ece=expected_calibration_error(confs, correct_flags),
        )
