"""Synthetic open-ended employee comments, each tied to an engagement score.

Comments are drawn from five latent themes. Some themes (growth, recognition)
co-occur with higher engagement; others (workload, unclear direction) with lower.
That structure is what the topic model should recover and the analysis should link
back to the engagement outcome.
"""
import numpy as np
import pandas as pd

import config

RNG = np.random.default_rng(config.RANDOM_STATE)

# theme -> (engagement pull, phrase bank)
THEMES = {
    "growth": (+1.2, [
        "lots of room to grow and learn new skills here",
        "my manager invests in my development and career path",
        "great opportunities to take on new challenges and stretch",
    ]),
    "recognition": (+0.9, [
        "i feel my work is recognized and valued by leadership",
        "good feedback and people appreciate the effort i put in",
        "wins get celebrated and contributions are acknowledged",
    ]),
    "workload": (-1.1, [
        "the workload is unsustainable and overtime never stops",
        "too much on my plate, constant deadlines, burning out",
        "understaffed teams mean we are always stretched too thin",
    ]),
    "direction": (-0.8, [
        "priorities shift constantly and the strategy is unclear",
        "leadership communication is confusing and direction changes weekly",
        "no clear vision, we keep changing course with no explanation",
    ]),
    "pay": (-0.4, [
        "compensation is below market for this kind of role",
        "pay has not kept up and raises are minimal",
        "salary is a concern compared to similar companies",
    ]),
}


def generate(n: int = 400) -> pd.DataFrame:
    rows = []
    theme_names = list(THEMES)
    for _ in range(n):
        theme = RNG.choice(theme_names)
        pull, bank = THEMES[theme]
        text = RNG.choice(bank)
        engagement = float(np.clip(RNG.normal(3 + pull, 0.7), 1, 5))
        rows.append({"comment": text, "engagement": round(engagement, 2), "latent_theme": theme})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    config.DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA, index=False)
    print(f"Wrote {len(df)} comments to {config.DATA}")
