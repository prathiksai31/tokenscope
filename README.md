# TokenScope

Choosing and budgeting models for AI coding agents. CS 571 semester project (UMass Amherst, Fall 2026).

TokenScope is an interactive dashboard with three linked views built on real benchmark data:

- **V1 Frontier:** accuracy vs. cost for 69 Aider polyglot benchmark runs, with the Pareto frontier.
- **V2 Configurations, retries and edit formats:** three tabs.
- **V3 Monthly cost estimate:** cost against a budget, plus a token breakdown.

Brush a region or click runs in V1 (each click adds or removes a run), and V2 and V3 follow your selection.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

| File | What it is |
| --- | --- |
| `data/polyglot_leaderboard.yml` | Aider polyglot leaderboard, pinned at commit 5dc9490 (Dec 2024 – Oct 2025) |
| `data/runs.csv` | Cleaned run table with derived fields, built by `python src/build_dataset.py` |
| `data/pricing_anthropic_2026-11-05.csv` | Anthropic list prices, retrieved Nov 5, 2026 from platform.claude.com/docs/en/about-claude/pricing |

## Limitations

- 18 runs report no cost. They are shown as "cost unknown" and never priced.
- Only 23 runs report token counts.
- Monthly figures are estimates. They assume each task uses as many tokens as one benchmark exercise.

## Shareable states

Some settings can be set in the URL, for example `?brush=0.3,20,70,100`, `?pick=gpt-5 (low)|gpt-5 (medium)`, `&tasks=12000&budget=500`, `?maxsec=60&minpass=50`.
