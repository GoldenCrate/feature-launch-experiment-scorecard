"""
analytics.py — Framework-free statistics and compute for the experiment
scorecard. No Streamlit / plotting imports here so the logic is unit-testable
and reusable.

Conventions
-----------
* "Rate" metrics are binary per-user outcomes (sent_message, crashed,
  retained_d7). We compare arms with a two-proportion z-test and report a
  Wald 95% CI on the absolute difference (treatment - control).
* "Continuous" metrics (latency_ms) are compared with Welch's t-test and a
  CI on the difference in means.
* For guardrails, the caller declares a `direction`: "higher_better" or
  "lower_better". The recommendation logic uses that to decide whether a
  statistically significant move is good or bad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

Direction = Literal["higher_better", "lower_better"]
MetricKind = Literal["rate", "continuous"]

Z_95 = stats.norm.ppf(0.975)  # ~1.95996


@dataclass
class MetricResult:
    """Outcome of comparing one metric between control and treatment."""

    name: str
    kind: MetricKind
    direction: Direction
    control_value: float
    treatment_value: float
    abs_lift: float           # treatment - control (same units as the metric)
    rel_lift: float           # abs_lift / control_value (fraction, may be nan)
    ci_low: float             # 95% CI on abs_lift
    ci_high: float
    p_value: float
    n_control: int
    n_treatment: int

    @property
    def significant(self) -> bool:
        """Statistically significant at alpha = 0.05 (CI excludes 0)."""
        return self.p_value < 0.05

    @property
    def is_improvement(self) -> bool:
        """Did the metric move in the desired direction (sign only)?"""
        if self.direction == "higher_better":
            return self.abs_lift > 0
        return self.abs_lift < 0


def two_proportion_ztest(
    successes_c: int, n_c: int, successes_t: int, n_t: int
) -> tuple[float, float, float, float]:
    """Two-proportion z-test (treatment vs control).

    Returns (abs_diff, ci_low, ci_high, p_value) where abs_diff = p_t - p_c
    and the CI is the Wald interval on that difference. The p-value uses the
    pooled-variance z statistic (standard for A/B rate tests).
    """
    p_c = successes_c / n_c
    p_t = successes_t / n_t
    abs_diff = p_t - p_c

    # Pooled proportion for the hypothesis test.
    p_pool = (successes_c + successes_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    if se_pool == 0:
        z, p_value = 0.0, 1.0
    else:
        z = abs_diff / se_pool
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled SE for the confidence interval on the difference.
    se_unpooled = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    ci_low = abs_diff - Z_95 * se_unpooled
    ci_high = abs_diff + Z_95 * se_unpooled
    return abs_diff, ci_low, ci_high, p_value


def welch_ttest(control: np.ndarray, treatment: np.ndarray) -> tuple[float, float, float, float]:
    """Welch's t-test on means (treatment vs control).

    Returns (mean_diff, ci_low, ci_high, p_value) with mean_diff =
    mean_t - mean_c and a 95% CI on that difference using the Welch df.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    mean_c, mean_t = control.mean(), treatment.mean()
    mean_diff = mean_t - mean_c

    var_c, var_t = control.var(ddof=1), treatment.var(ddof=1)
    n_c, n_t = len(control), len(treatment)
    se = np.sqrt(var_c / n_c + var_t / n_t)

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
    # Welch-Satterthwaite degrees of freedom.
    df = (var_c / n_c + var_t / n_t) ** 2 / (
        (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
    )
    t_crit = stats.t.ppf(0.975, df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    return mean_diff, ci_low, ci_high, float(p_value)


def analyze_rate_metric(
    df: pd.DataFrame, column: str, name: str, direction: Direction
) -> MetricResult:
    """Compare a binary rate metric between arms."""
    control = df[df["variant"] == "control"]
    treatment = df[df["variant"] == "treatment"]
    n_c, n_t = len(control), len(treatment)
    s_c, s_t = int(control[column].sum()), int(treatment[column].sum())
    p_c, p_t = s_c / n_c, s_t / n_t

    abs_diff, ci_low, ci_high, p_value = two_proportion_ztest(s_c, n_c, s_t, n_t)
    rel_lift = abs_diff / p_c if p_c else float("nan")
    return MetricResult(
        name=name,
        kind="rate",
        direction=direction,
        control_value=p_c,
        treatment_value=p_t,
        abs_lift=abs_diff,
        rel_lift=rel_lift,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_control=n_c,
        n_treatment=n_t,
    )


def analyze_continuous_metric(
    df: pd.DataFrame, column: str, name: str, direction: Direction
) -> MetricResult:
    """Compare a continuous metric between arms with Welch's t-test."""
    control = df[df["variant"] == "control"][column].to_numpy()
    treatment = df[df["variant"] == "treatment"][column].to_numpy()
    mean_c, mean_t = control.mean(), treatment.mean()
    mean_diff, ci_low, ci_high, p_value = welch_ttest(control, treatment)
    rel_lift = mean_diff / mean_c if mean_c else float("nan")
    return MetricResult(
        name=name,
        kind="continuous",
        direction=direction,
        control_value=mean_c,
        treatment_value=mean_t,
        abs_lift=mean_diff,
        rel_lift=rel_lift,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_control=len(control),
        n_treatment=len(treatment),
    )


# --- Metric registry --------------------------------------------------------
# Primary metric + guardrails, declared once so app and tests agree.
PRIMARY_METRIC = {
    "column": "sent_message",
    "name": "7-day message-send rate",
    "kind": "rate",
    "direction": "higher_better",
}
GUARDRAIL_METRICS = [
    {"column": "crashed", "name": "Client crash rate", "kind": "rate", "direction": "lower_better"},
    {"column": "latency_ms", "name": "p95 client latency (ms)", "kind": "continuous", "direction": "lower_better"},
    {"column": "retained_d7", "name": "D7 retention rate", "kind": "rate", "direction": "higher_better"},
]


def _analyze_one(df: pd.DataFrame, spec: dict) -> MetricResult:
    if spec["kind"] == "rate":
        return analyze_rate_metric(df, spec["column"], spec["name"], spec["direction"])
    return analyze_continuous_metric(df, spec["column"], spec["name"], spec["direction"])


def analyze_primary(df: pd.DataFrame) -> MetricResult:
    return _analyze_one(df, PRIMARY_METRIC)


def analyze_guardrails(df: pd.DataFrame) -> list[MetricResult]:
    return [_analyze_one(df, spec) for spec in GUARDRAIL_METRICS]


# --- Funnel -----------------------------------------------------------------
FUNNEL_STAGES = [
    ("installed", "Install"),
    ("activated", "Activate"),
    ("engaged", "Engage"),
    ("retained_stage", "Retain"),
]


def funnel_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy funnel table: variant, stage, users, and conversion rates.

    `overall_conv` is relative to Install (top of funnel); `step_conv` is the
    stage-over-previous-stage conversion.
    """
    rows = []
    for variant in ["control", "treatment"]:
        sub = df[df["variant"] == variant]
        prev_count = None
        top_count = None
        for i, (col, label) in enumerate(FUNNEL_STAGES):
            count = int(sub[col].sum())
            if i == 0:
                top_count = count
            overall = count / top_count if top_count else float("nan")
            step = count / prev_count if prev_count else 1.0
            rows.append(
                {
                    "variant": variant,
                    "stage": label,
                    "stage_order": i,
                    "users": count,
                    "overall_conv": overall,
                    "step_conv": step,
                }
            )
            prev_count = count
    return pd.DataFrame(rows)


# --- Segment breakdown ------------------------------------------------------
def segment_lift(df: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """Per-platform lift for one metric (default: the primary metric).

    Returns a tidy frame with the control/treatment value, absolute & relative
    lift, CI and p-value for each platform, plus a flag for whether that
    segment moved the desired way. Surfaces interaction effects (a platform
    that regresses against the aggregate win).
    """
    spec = spec or PRIMARY_METRIC
    rows = []
    for plat in sorted(df["platform"].unique()):
        sub = df[df["platform"] == plat]
        res = _analyze_one(sub, spec)
        rows.append(
            {
                "platform": plat,
                "control_value": res.control_value,
                "treatment_value": res.treatment_value,
                "abs_lift": res.abs_lift,
                "rel_lift": res.rel_lift,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "p_value": res.p_value,
                "significant": res.significant,
                "is_improvement": res.is_improvement,
                "n": res.n_control + res.n_treatment,
            }
        )
    return pd.DataFrame(rows)


# --- Recommendation engine --------------------------------------------------
@dataclass
class Recommendation:
    """A ship/iterate/hold verdict with plain-English reasoning."""

    verdict: Literal["Ship", "Iterate", "Hold"]
    headline: str
    reasons: list[str] = field(default_factory=list)


# A guardrail move is only "practically" concerning if it clears this
# relative-change threshold AND is statistically significant. This separates
# statistical significance from practical significance.
GUARDRAIL_PRACTICAL_REL_THRESHOLD = 0.10  # 10% relative change
PRIMARY_PRACTICAL_REL_THRESHOLD = 0.01     # 1% relative lift on primary


def recommend(
    primary: MetricResult,
    guardrails: list[MetricResult],
    segments: pd.DataFrame,
) -> Recommendation:
    """Weigh primary vs guardrails and statistical vs practical significance.

    Logic (deliberately conservative, as experimentation review would be):
      * Hold if the primary is a significant REGRESSION, or if a guardrail has
        a significant AND practically large regression.
      * Ship if the primary is a significant, practically meaningful WIN, no
        guardrail is meaningfully breached, and no major segment regresses.
      * Otherwise Iterate (flat/underpowered primary, or mixed signals).
    """
    reasons: list[str] = []

    primary_sig_win = primary.significant and primary.is_improvement and (
        abs(primary.rel_lift) >= PRIMARY_PRACTICAL_REL_THRESHOLD
    )
    primary_sig_regression = primary.significant and not primary.is_improvement

    # Identify breached guardrails (significant + practically large + wrong way).
    breached = []
    for g in guardrails:
        wrong_way = not g.is_improvement
        practical = abs(g.rel_lift) >= GUARDRAIL_PRACTICAL_REL_THRESHOLD
        if g.significant and wrong_way and practical:
            breached.append(g)

    # Segments that significantly regress on the primary metric.
    regressing_segments = segments[
        (segments["significant"]) & (~segments["is_improvement"])
    ]["platform"].tolist()

    # Describe the primary outcome.
    if primary_sig_win:
        reasons.append(
            f"Primary metric ({primary.name}) is up "
            f"{primary.rel_lift * 100:+.1f}% "
            f"({primary.abs_lift * 100:+.2f} pts), statistically significant "
            f"(p={primary.p_value:.4f})."
        )
    elif primary_sig_regression:
        reasons.append(
            f"Primary metric ({primary.name}) REGRESSED "
            f"{primary.rel_lift * 100:+.1f}% and is statistically significant "
            f"(p={primary.p_value:.4f})."
        )
    else:
        reasons.append(
            f"Primary metric ({primary.name}) moved {primary.rel_lift * 100:+.1f}% "
            f"but is not a clear, practically meaningful win "
            f"(p={primary.p_value:.4f})."
        )

    if breached:
        for g in breached:
            reasons.append(
                f"Guardrail breached: {g.name} moved {g.rel_lift * 100:+.1f}% "
                f"the wrong way (p={g.p_value:.4f}) — past the "
                f"{int(GUARDRAIL_PRACTICAL_REL_THRESHOLD * 100)}% practical limit."
            )
    else:
        reasons.append("No guardrail breached its practical limit.")

    if regressing_segments:
        reasons.append(
            "Segment interaction: primary metric significantly REGRESSES on "
            + ", ".join(regressing_segments)
            + " even though the aggregate is positive."
        )

    # Decide the verdict.
    if primary_sig_regression or breached:
        verdict = "Hold"
        headline = (
            "Hold the launch — a guardrail or the primary metric regressed "
            "beyond what the win justifies."
        )
    elif primary_sig_win and not regressing_segments:
        verdict = "Ship"
        headline = "Ship it — clear primary win with guardrails intact."
    else:
        verdict = "Iterate"
        if regressing_segments:
            headline = (
                "Iterate — the feature wins overall but regresses on "
                + ", ".join(regressing_segments)
                + "; fix that segment before a full rollout."
            )
        else:
            headline = (
                "Iterate — the primary signal isn't a decisive, practical win yet."
            )

    return Recommendation(verdict=verdict, headline=headline, reasons=reasons)
