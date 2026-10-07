"""
Unit tests for analytics.py — run with:  python -m pytest -q
(No Streamlit needed; analytics is framework-free.)
"""

import numpy as np
import pandas as pd

import analytics as A
from data_gen import generate


def test_two_proportion_ztest_detects_known_difference():
    # 60% vs 50% over 1000 each should be clearly significant.
    diff, lo, hi, p = A.two_proportion_ztest(500, 1000, 600, 1000)
    assert abs(diff - 0.10) < 1e-9
    assert p < 0.001
    assert lo < 0.10 < hi  # CI brackets the true diff


def test_two_proportion_ztest_no_difference_is_insignificant():
    diff, lo, hi, p = A.two_proportion_ztest(500, 1000, 500, 1000)
    assert diff == 0.0
    assert p > 0.9
    assert lo < 0 < hi


def test_welch_ttest_sign_and_significance():
    rng = np.random.default_rng(0)
    c = rng.normal(100, 10, 5000)
    t = rng.normal(105, 10, 5000)  # +5 shift
    diff, lo, hi, p = A.welch_ttest(c, t)
    assert diff > 0
    assert p < 0.001
    assert lo < diff < hi


def test_rate_metric_direction_flags():
    df = generate(seed=42, n_users=20000)
    crash = A.analyze_rate_metric(df, "crashed", "crash", "lower_better")
    # Synthetic world raises crash rate in treatment -> a regression.
    assert crash.abs_lift > 0
    assert crash.is_improvement is False


def test_funnel_is_monotonic_nonincreasing():
    df = generate(seed=42, n_users=20000)
    f = A.funnel_by_variant(df)
    for variant in ["control", "treatment"]:
        counts = f[f["variant"] == variant].sort_values("stage_order")["users"].tolist()
        assert counts == sorted(counts, reverse=True)  # install >= ... >= retain


def test_segment_lift_surfaces_mobile_regression():
    df = generate(seed=42, n_users=48000)
    seg = A.segment_lift(df).set_index("platform")
    # By design: desktop/web positive, mobile negative on the primary metric.
    assert seg.loc["desktop", "abs_lift"] > 0
    assert seg.loc["web", "abs_lift"] > 0
    assert seg.loc["mobile", "abs_lift"] < 0


def test_recommendation_holds_on_breached_guardrail():
    df = generate(seed=42, n_users=48000)
    primary = A.analyze_primary(df)
    guardrails = A.analyze_guardrails(df)
    seg = A.segment_lift(df)
    rec = A.recommend(primary, guardrails, seg)
    # Crash-rate guardrail is breached -> Hold.
    assert rec.verdict == "Hold"
    assert rec.reasons


def test_recommendation_ships_on_clean_win():
    # Construct a clean win: primary up, all guardrails fine, no bad segment.
    primary = A.MetricResult(
        name="p", kind="rate", direction="higher_better",
        control_value=0.50, treatment_value=0.55, abs_lift=0.05,
        rel_lift=0.10, ci_low=0.03, ci_high=0.07, p_value=1e-6,
        n_control=10000, n_treatment=10000,
    )
    guardrails = [
        A.MetricResult("g", "rate", "lower_better", 0.02, 0.0201, 0.0001,
                       0.005, -0.001, 0.0011, 0.8, 10000, 10000),
    ]
    seg = pd.DataFrame([
        {"platform": "desktop", "significant": True, "is_improvement": True},
        {"platform": "mobile", "significant": True, "is_improvement": True},
    ])
    rec = A.recommend(primary, guardrails, seg)
    assert rec.verdict == "Ship"
