"""
data_gen.py — Synthetic experiment data generator.

Generates a SYNTHETIC A/B test dataset for a hypothetical Discord feature
launch ("Threaded Replies v2"). All data is fabricated with a fixed random
seed for reproducibility. No real user data is used anywhere in this project.

The data is intentionally designed to contain a realistic nuance:
    * The feature WINS on the primary metric overall (message-send rate up).
    * A guardrail REGRESSES (client crash rate rises in treatment).
    * One platform (mobile) REGRESSES on the primary metric even though
      desktop and web drive the aggregate win  -> Simpson-style interaction.

This forces a genuine "ship / iterate / hold" judgement rather than a
rubber-stamp, which is the point of the scorecard.

Run directly to (re)generate data/experiment.csv:
    python data_gen.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "experiment.csv")

# Experiment metadata (synthetic).
EXPERIMENT_NAME = "Threaded Replies v2"
N_USERS = 48_000  # total users enrolled across both arms

# Platform mix (share of users on each platform).
PLATFORMS = ["desktop", "mobile", "web"]
PLATFORM_WEIGHTS = [0.42, 0.45, 0.13]

# --- Ground-truth effect design (what the synthetic "world" looks like) -----
# Primary metric: 7-day message-send rate (did the user send a message?).
# Guardrails: client crash rate (lower is better), p95 client latency in ms
# (lower is better), and D7 retention rate (higher is better).
#
# Baseline (control) rates by platform, and the treatment effect applied.
# Note the mobile primary effect is NEGATIVE: the new threading UI is cramped
# on small screens, hurting sends there while helping elsewhere.
CONTROL_SEND_RATE = {"desktop": 0.620, "mobile": 0.560, "web": 0.590}
TREAT_SEND_LIFT = {"desktop": +0.045, "mobile": -0.020, "web": +0.038}

CONTROL_CRASH_RATE = {"desktop": 0.018, "mobile": 0.031, "web": 0.012}
# Treatment raises crash rate everywhere (the guardrail regression).
TREAT_CRASH_LIFT = {"desktop": +0.006, "mobile": +0.011, "web": +0.004}

CONTROL_RETENTION = {"desktop": 0.515, "mobile": 0.470, "web": 0.480}
TREAT_RETENTION_LIFT = {"desktop": +0.012, "mobile": +0.004, "web": +0.010}

# Continuous guardrail: p95 client latency (ms). Treatment adds a little.
CONTROL_LATENCY_MEAN = {"desktop": 180.0, "mobile": 240.0, "web": 210.0}
CONTROL_LATENCY_SD = {"desktop": 40.0, "mobile": 55.0, "web": 48.0}
TREAT_LATENCY_ADD = {"desktop": 6.0, "mobile": 9.0, "web": 5.0}


def _bernoulli(rng: np.random.Generator, p: np.ndarray) -> np.ndarray:
    """Vectorised Bernoulli draw given a per-row probability array."""
    return (rng.random(len(p)) < p).astype(int)


def generate(seed: int = SEED, n_users: int = N_USERS) -> pd.DataFrame:
    """Build and return the synthetic experiment dataframe (one row per user)."""
    rng = np.random.default_rng(seed)

    platform = rng.choice(PLATFORMS, size=n_users, p=PLATFORM_WEIGHTS)
    # 50/50 assignment with slight random imbalance, as happens in practice.
    variant = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])

    send_p = np.empty(n_users)
    crash_p = np.empty(n_users)
    retain_p = np.empty(n_users)
    latency_mean = np.empty(n_users)
    latency_sd = np.empty(n_users)

    is_treat = variant == "treatment"
    for plat in PLATFORMS:
        mask = platform == plat
        send_p[mask] = CONTROL_SEND_RATE[plat]
        send_p[mask & is_treat] += TREAT_SEND_LIFT[plat]

        crash_p[mask] = CONTROL_CRASH_RATE[plat]
        crash_p[mask & is_treat] += TREAT_CRASH_LIFT[plat]

        retain_p[mask] = CONTROL_RETENTION[plat]
        retain_p[mask & is_treat] += TREAT_RETENTION_LIFT[plat]

        latency_mean[mask] = CONTROL_LATENCY_MEAN[plat]
        latency_mean[mask & is_treat] += TREAT_LATENCY_ADD[plat]
        latency_sd[mask] = CONTROL_LATENCY_SD[plat]

    sent_message = _bernoulli(rng, np.clip(send_p, 0, 1))
    crashed = _bernoulli(rng, np.clip(crash_p, 0, 1))
    retained_d7 = _bernoulli(rng, np.clip(retain_p, 0, 1))
    latency_ms = np.clip(rng.normal(latency_mean, latency_sd), 20, None).round(1)

    # Funnel stages. Everyone installs. Activation (opened feature) depends on
    # variant/platform; engage requires activate; retain requires engage.
    # These are modelled as nested conditional Bernoullis so the funnel is
    # internally consistent (you cannot engage without activating).
    activate_p = np.where(is_treat, 0.74, 0.70)
    # The new UI gets slightly more people to activate overall...
    activated = _bernoulli(rng, activate_p)
    engage_p = np.where(is_treat, 0.63, 0.60)
    engaged = activated * _bernoulli(rng, engage_p)
    # Tie the funnel's final "retain" stage to the D7 retention draw so the
    # two views tell a consistent story.
    retained_stage = engaged * retained_d7

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "experiment": EXPERIMENT_NAME,
            "variant": variant,
            "platform": platform,
            # primary metric (binary)
            "sent_message": sent_message,
            # guardrails
            "crashed": crashed,
            "latency_ms": latency_ms,
            "retained_d7": retained_d7,
            # funnel stages (binary, nested)
            "installed": 1,
            "activated": activated,
            "engaged": engaged,
            "retained_stage": retained_stage,
        }
    )
    return df


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df):,} synthetic rows -> {OUTPUT_CSV}")
    print(df.groupby("variant")[["sent_message", "crashed", "retained_d7"]].mean())


if __name__ == "__main__":
    main()
