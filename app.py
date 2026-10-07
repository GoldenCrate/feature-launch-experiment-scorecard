"""
Feature-Launch Experiment Scorecard — Streamlit app.

A multi-view experiment readout for a (synthetic) Discord feature launch,
"Threaded Replies v2". Built as a portfolio piece for a Data Scientist role on
Discord's Client Platform team: A/B testing, guardrail monitoring, funnel
analytics and a plain-English ship decision for PMs.

All data is SYNTHETIC (see data_gen.py). Run with:
    streamlit run app.py

Author: Leo Chan — github.com/GoldenCrate
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

import analytics as A
from data_gen import EXPERIMENT_NAME

# --- Storytelling-with-Data palette (restrained; color = meaning) -----------
BLURPLE = "#5865F2"   # treatment / "the thing we're testing"
GRAY = "#B9BBBE"      # control / context
INK = "#2E3338"       # text
GOOD = "#248046"      # improvement
BAD = "#D83C3E"       # regression
MUTE = "#6D7075"      # de-emphasised

DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "experiment.csv")

st.set_page_config(
    page_title="Feature-Launch Experiment Scorecard",
    page_icon="🧪",
    layout="wide",
)

alt.data_transformers.disable_max_rows()


# --- Data loading -----------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_CSV):
        # Generate on first run if the CSV isn't present.
        from data_gen import main as gen_main

        gen_main()
    return pd.read_csv(DATA_CSV)


@st.cache_data
def compute_all(df: pd.DataFrame):
    primary = A.analyze_primary(df)
    guardrails = A.analyze_guardrails(df)
    segments = A.segment_lift(df)
    funnel = A.funnel_by_variant(df)
    rec = A.recommend(primary, guardrails, segments)
    return primary, guardrails, segments, funnel, rec


# --- Formatting helpers -----------------------------------------------------
def fmt_metric_value(res: A.MetricResult) -> tuple[str, str]:
    """Return (control_str, treatment_str) formatted per metric kind."""
    if res.kind == "rate":
        return f"{res.control_value * 100:.2f}%", f"{res.treatment_value * 100:.2f}%"
    return f"{res.control_value:.1f}", f"{res.treatment_value:.1f}"


def fmt_lift(res: A.MetricResult) -> str:
    if res.kind == "rate":
        return f"{res.abs_lift * 100:+.2f} pts ({res.rel_lift * 100:+.1f}%)"
    return f"{res.abs_lift:+.1f} ({res.rel_lift * 100:+.1f}%)"


def fmt_ci(res: A.MetricResult) -> str:
    if res.kind == "rate":
        return f"[{res.ci_low * 100:+.2f}, {res.ci_high * 100:+.2f}] pts"
    return f"[{res.ci_low:+.1f}, {res.ci_high:+.1f}]"


def verdict_color(v: str) -> str:
    return {"Ship": GOOD, "Iterate": "#C27C0E", "Hold": BAD}.get(v, MUTE)


# --- Chart builders (declutter: no gridlines clutter, direct labels) --------
def arm_bar_chart(res: A.MetricResult) -> alt.Chart:
    """Two-bar control vs treatment with direct value labels and a CI note."""
    if res.kind == "rate":
        vals = [res.control_value * 100, res.treatment_value * 100]
        unit = "%"
    else:
        vals = [res.control_value, res.treatment_value]
        unit = ""
    d = pd.DataFrame({"arm": ["Control", "Treatment"], "value": vals})
    base = alt.Chart(d).encode(
        x=alt.X("arm:N", title=None, sort=["Control", "Treatment"],
                axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        y=alt.Y("value:Q", title=None, axis=None),
        color=alt.Color("arm:N", scale=alt.Scale(domain=["Control", "Treatment"],
                                                  range=[GRAY, BLURPLE]), legend=None),
    )
    bars = base.mark_bar(size=64, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    labels = base.mark_text(dy=-10, fontSize=15, fontWeight="bold", color=INK).encode(
        text=alt.Text("value:Q", format=".1f" if unit else ".1f")
    )
    return (bars + labels).properties(height=240)


def segment_lift_chart(segments: pd.DataFrame, metric_name: str) -> alt.Chart:
    """Horizontal lift-by-platform bars, colored by win/regress, with CI rules."""
    d = segments.copy()
    d["rel_pct"] = d["rel_lift"] * 100
    d["lo"] = d["ci_low"] * 100
    d["hi"] = d["ci_high"] * 100
    d["Platform"] = d["platform"].str.capitalize()
    d["outcome"] = d.apply(
        lambda r: "Regression" if r["rel_pct"] < 0 else "Win", axis=1
    )
    base = alt.Chart(d).encode(
        y=alt.Y("Platform:N", title=None, sort="-x",
                axis=alt.Axis(labelFontSize=13)),
    )
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(
        color=MUTE, strokeDash=[4, 3]
    ).encode(x="z:Q")
    ci = base.mark_rule(color=MUTE, strokeWidth=2).encode(
        x=alt.X("lo:Q", title="Relative lift on primary metric (%)"),
        x2="hi:Q",
    )
    bars = base.mark_bar(size=18).encode(
        x=alt.X("rel_pct:Q"),
        color=alt.Color("outcome:N",
                        scale=alt.Scale(domain=["Win", "Regression"], range=[BLURPLE, BAD]),
                        legend=alt.Legend(title=None, orient="top")),
    )
    labels = base.mark_text(align="left", dx=6, fontSize=12, color=INK).encode(
        x=alt.X("hi:Q"),
        text=alt.Text("rel_pct:Q", format="+.1f"),
    )
    return (zero + ci + bars + labels).properties(height=170)


def funnel_chart(funnel: pd.DataFrame) -> alt.Chart:
    """Grouped funnel: users per stage, control vs treatment."""
    d = funnel.copy()
    d["Arm"] = d["variant"].str.capitalize()
    order = ["Install", "Activate", "Engage", "Retain"]
    base = alt.Chart(d).encode(
        x=alt.X("stage:N", sort=order, title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        y=alt.Y("users:Q", title="Users", axis=alt.Axis(format="~s")),
        color=alt.Color("Arm:N", scale=alt.Scale(domain=["Control", "Treatment"],
                                                  range=[GRAY, BLURPLE]),
                        legend=alt.Legend(title=None, orient="top")),
        xOffset=alt.XOffset("Arm:N"),
    )
    bars = base.mark_bar()
    labels = base.mark_text(dy=-6, fontSize=11, color=INK).encode(
        text=alt.Text("overall_conv:Q", format=".0%")
    )
    return (bars + labels).properties(height=300)


# ============================================================================
# App
# ============================================================================
df = load_data()
primary, guardrails, segments, funnel, rec = compute_all(df)

st.title("🧪 Feature-Launch Experiment Scorecard")
st.caption(
    f"Experiment: **{EXPERIMENT_NAME}**  ·  "
    f"{len(df):,} users  ·  50/50 A/B assignment  ·  "
    "**100% synthetic data — portfolio demo**"
)

st.sidebar.header("Views")
view = st.sidebar.radio(
    "Go to",
    ["Scorecard", "Retention funnel", "Segment breakdown", "Recommendation"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**  \nSynthetic A/B test for a hypothetical Discord feature. "
    "Stats in `analytics.py` (two-proportion z-test for rates, Welch's t-test "
    "for latency; 95% CIs).  \n\nAuthor: **Leo Chan** — "
    "[github.com/GoldenCrate](https://github.com/GoldenCrate)"
)

# --- View 1: Scorecard ------------------------------------------------------
if view == "Scorecard":
    st.subheader("Primary metric wins, but a guardrail cracks")
    st.markdown(
        "The feature lifts the primary metric with statistical significance — "
        "yet the crash-rate guardrail moves the wrong way. Read both before shipping."
    )

    c_str, t_str = fmt_metric_value(primary)
    k1, k2, k3 = st.columns(3)
    k1.metric(f"{primary.name} — Treatment", t_str,
              fmt_lift(primary), delta_color="normal")
    k2.metric("Control", c_str)
    k3.metric("Statistical significance",
              "Significant" if primary.significant else "Not significant",
              f"p = {primary.p_value:.4f}", delta_color="off")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Primary: " + primary.name + "**")
        st.altair_chart(arm_bar_chart(primary), width='stretch')
        st.caption(
            f"95% CI on lift: {fmt_ci(primary)}. The interval excludes 0, "
            "so the win is unlikely to be noise."
        )
    with right:
        st.markdown("**Guardrails**")
        grows = []
        for g in guardrails:
            c_s, t_s = fmt_metric_value(g)
            status = "OK"
            if g.significant and not g.is_improvement:
                status = "⚠ Regressed"
            elif g.significant and g.is_improvement:
                status = "✓ Improved"
            elif not g.significant:
                status = "Flat (n.s.)"
            grows.append({
                "Guardrail": g.name,
                "Control": c_s,
                "Treatment": t_s,
                "Lift": fmt_lift(g),
                "p-value": f"{g.p_value:.4f}",
                "Status": status,
            })
        gdf = pd.DataFrame(grows)
        st.dataframe(gdf, hide_index=True, width='stretch')
        st.caption(
            "Direction matters: for crash rate and latency, lower is better. "
            "A significant *increase* there is a regression, not a win."
        )

    st.markdown("---")
    st.markdown(f"#### Verdict preview: :{'red' if rec.verdict=='Hold' else 'orange' if rec.verdict=='Iterate' else 'green'}[{rec.verdict}]")
    st.write(rec.headline + "  See the **Recommendation** view for the full rationale.")

# --- View 2: Funnel ---------------------------------------------------------
elif view == "Retention funnel":
    st.subheader("Treatment moves more users down the funnel at every stage")
    st.markdown(
        "Install → Activate → Engage → Retain. Labels show conversion from the "
        "top of the funnel (Install = 100%)."
    )
    st.altair_chart(funnel_chart(funnel), width='stretch')

    st.markdown("**Stage-over-stage conversion**")
    wide = funnel.pivot(index="stage", columns="variant", values="step_conv")
    wide = wide.reindex(["Install", "Activate", "Engage", "Retain"])
    wide = wide.rename(columns={"control": "Control", "treatment": "Treatment"})
    wide["Delta (pts)"] = (wide["Treatment"] - wide["Control"]) * 100
    show = wide.copy()
    show["Control"] = (show["Control"] * 100).map("{:.1f}%".format)
    show["Treatment"] = (show["Treatment"] * 100).map("{:.1f}%".format)
    show["Delta (pts)"] = show["Delta (pts)"].map("{:+.1f}".format)
    st.dataframe(show, width='stretch')
    st.caption(
        "The biggest lift is at Activate (users opening the new feature) and it "
        "carries through Engage. Retain is roughly flat step-over-step, so the "
        "funnel win is driven by getting more people in, not by stickier behavior."
    )

# --- View 3: Segments -------------------------------------------------------
elif view == "Segment breakdown":
    st.subheader("Desktop and web drive the win — mobile quietly regresses")
    st.markdown(
        "Aggregate lift can hide an interaction effect. Splitting the **primary "
        "metric** by platform shows the new threading UI helps on large screens "
        "but *hurts* on mobile, where it's cramped."
    )
    st.altair_chart(segment_lift_chart(segments, primary.name), width='stretch')

    tbl = segments.copy()
    tbl["Platform"] = tbl["platform"].str.capitalize()
    tbl["Control"] = (tbl["control_value"] * 100).map("{:.2f}%".format)
    tbl["Treatment"] = (tbl["treatment_value"] * 100).map("{:.2f}%".format)
    tbl["Rel. lift"] = (tbl["rel_lift"] * 100).map("{:+.1f}%".format)
    tbl["p-value"] = tbl["p_value"].map("{:.4f}".format)
    tbl["Significant?"] = tbl["significant"].map({True: "Yes", False: "No"})
    tbl["Users"] = tbl["n"].map("{:,}".format)
    st.dataframe(
        tbl[["Platform", "Control", "Treatment", "Rel. lift", "p-value",
             "Significant?", "Users"]],
        hide_index=True, width='stretch',
    )
    st.caption(
        "Mobile is the largest segment, so even a modest negative there is a "
        "real risk — and it's the platform most exposed to the crash-rate "
        "regression. This is the single most important slide for the ship call."
    )

# --- View 4: Recommendation -------------------------------------------------
else:
    st.subheader("Recommendation")
    color = verdict_color(rec.verdict)
    st.markdown(
        f"""
        <div style="border-left:8px solid {color}; background:{color}14;
                    padding:18px 22px; border-radius:6px; margin-bottom:8px;">
          <div style="font-size:30px; font-weight:800; color:{color};
                      line-height:1;">{rec.verdict}</div>
          <div style="font-size:16px; color:{INK}; margin-top:8px;">{rec.headline}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Why:**")
    for r in rec.reasons:
        st.markdown(f"- {r}")

    st.markdown("---")
    st.markdown(
        "**How this verdict is made** (in `analytics.py`): the engine weighs the "
        "primary metric against the guardrails, and separates *statistical* "
        "significance (is the move real?) from *practical* significance (is it "
        "big enough to matter?). "
        f"A guardrail must move at least "
        f"{int(A.GUARDRAIL_PRACTICAL_REL_THRESHOLD*100)}% the wrong way *and* be "
        "significant to count as breached; the primary must clear "
        f"{A.PRIMARY_PRACTICAL_REL_THRESHOLD*100:.0f}% relative lift to count as a "
        "practical win."
    )
    st.info(
        "Plain-English takeaway: the feature works — more people send messages "
        "and move through the funnel — but it ships a crash-rate regression and "
        "doesn't help (maybe hurts) mobile users. Hold the full rollout, fix "
        "crashes and the mobile layout, then re-run. This is **synthetic demo "
        "data** built to make that trade-off concrete."
    )
