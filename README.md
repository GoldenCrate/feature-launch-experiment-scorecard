# Feature-Launch Experiment Scorecard

An interactive **A/B-test readout** for a feature launch, built the way a
product data scientist would hand results to a PM: a primary metric with
guardrails, a retention funnel, a per-platform segment breakdown, and a plain
-English **ship / iterate / hold** recommendation that weighs statistical
against practical significance.

The demo scenario is a hypothetical Discord client feature, **"Threaded
Replies v2"**, evaluated across desktop, mobile and web.

> ⚠️ **All data in this project is 100% synthetic**, generated with a fixed
> random seed (`data_gen.py`). No real Discord data — or any real user data —
> is used. It exists purely to demonstrate the analysis and the UI.

---

## Why this project

It mirrors the day-to-day of a **Data Scientist on a client platform team**:
designing and reading experiments, defining guardrails, catching interaction
effects that an aggregate number hides, and turning that into a decision a PM
can act on. The scenario is deliberately *nuanced* — the feature wins on the
headline metric but trips a guardrail and regresses on one platform — because
the interesting part of experimentation is the judgement call, not a green
arrow.

## The views

| View | What it answers |
|------|-----------------|
| **Scorecard** | Did the primary metric (7-day message-send rate) move? Are the guardrails (crash rate, p95 latency, D7 retention) still safe? Each with lift, p-value and a 95% CI. |
| **Retention funnel** | Install → Activate → Engage → Retain, per arm, with stage-over-stage conversion. Where in the funnel does the feature help? |
| **Segment breakdown** | Primary-metric lift split by platform, surfacing the interaction effect (desktop/web win, mobile regresses). |
| **Recommendation** | A plain-English Ship / Iterate / Hold verdict with the full reasoning, separating *statistical* significance from *practical* significance. |

## The analysis (`analytics.py`)

All statistics live in a framework-free, unit-testable module:

- **Rate metrics** (send rate, crash rate, retention): **two-proportion
  z-test** (pooled variance for the p-value) with a **Wald 95% CI** on the
  absolute difference.
- **Continuous metric** (p95 latency): **Welch's t-test** with a 95% CI on the
  difference in means.
- Each guardrail declares a direction (`higher_better` / `lower_better`) so a
  significant move is correctly read as an improvement or a regression.
- The recommendation engine encodes a conservative review policy: *hold* if the
  primary regresses or a guardrail is significantly **and** practically breached;
  *ship* only on a clear, practically meaningful primary win with guardrails
  intact and no regressing segment; otherwise *iterate*.

## Design

Charts follow Cole Nussbaumer Knaflic's *Storytelling with Data*: takeaway
titles that state the finding, decluttered axes, direct value labels instead of
legends-plus-gridlines, and restrained color (blurple = the treatment we're
testing, gray = control context, red reserved for regressions).

---

## Run it

```bash
# 1. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. generate the synthetic dataset (writes data/experiment.csv)
python data_gen.py

# 4. launch the app
streamlit run app.py
```

The app also regenerates the dataset automatically on first run if
`data/experiment.csv` is missing, so step 3 is optional.

## Tests

The analytics module is covered by unit tests (no Streamlit required):

```bash
python -m pytest -q
```

## Project layout

```
app.py             Streamlit multi-view UI
analytics.py       Framework-free stats + recommendation engine (unit-testable)
data_gen.py        Synthetic data generator (fixed seed)
test_analytics.py  Unit tests for the stats + recommendation logic
requirements.txt   Dependencies
README.md          This file
```

---

**Author:** Leo Chan — [github.com/GoldenCrate](https://github.com/GoldenCrate)
