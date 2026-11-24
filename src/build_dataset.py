"""Build the cleaned run-level table for the TokenScope project.

Input : data/polyglot_leaderboard.yml  (Aider polyglot leaderboard, pinned commit)
Output: data/runs.csv                  (one row per benchmark run, with derived fields)
"""
import re, sys, pathlib
import yaml, pandas as pd, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "polyglot_leaderboard.yml"
OUT = ROOT / "data" / "runs.csv"

PROVIDERS = [  # (regex on model name, provider) -- first match wins
    (r"claude", "Anthropic"),
    (r"\bgpt|\bo1\b|\bo3|\bo4|openai|gpt-oss|chatgpt", "OpenAI"),
    (r"gemini|gemma", "Google"),
    (r"deepseek", "DeepSeek"),
    (r"qwen|qwq", "Alibaba (Qwen)"),
    (r"grok", "xAI"),
    (r"llama", "Meta"),
    (r"kimi", "Moonshot"),
    (r"mistral|codestral", "Mistral"),
    (r"command", "Cohere"),
    (r"yi-", "01.AI"),
    (r"openhands", "All Hands AI"),
]

def provider(name: str) -> str:
    n = name.lower()
    if "+" in n:            # architect/editor pairs, e.g. "o3 (high) + gpt-4.1"
        n = n.split("+")[0]
    for pat, p in PROVIDERS:
        if re.search(pat, n):
            return p
    return "Other"

def main():
    runs = pd.DataFrame(yaml.safe_load(SRC.read_text()))
    runs["date"] = pd.to_datetime(runs["date"])
    runs["provider"] = runs["model"].map(provider)
    # Quasar/Optimus Alpha were anonymous "stealth" preview models; provider not stated in the data.
    runs["stealth_model"] = runs["model"].str.contains("Alpha", case=False)
    runs.loc[runs["stealth_model"], "provider"] = "Unattributed preview"
    # 0 or missing cost = free preview / local model / not recorded -> treat as unknown
    runs["cost_known"] = runs["total_cost"].fillna(0) > 0
    runs.loc[~runs["cost_known"], "total_cost"] = np.nan
    solved = runs["pass_rate_2"] / 100 * runs["test_cases"]
    runs["tasks_solved"] = solved.round().astype(int)
    runs["cost_per_task"] = runs["total_cost"] / runs["test_cases"]
    runs["cost_per_solved_task"] = runs["total_cost"] / solved
    runs["retry_gain_pp"] = runs["pass_rate_2"] - runs["pass_rate_1"]
    runs["has_tokens"] = runs["prompt_tokens"].notna()
    runs["tokens_per_task"] = (runs["prompt_tokens"] + runs["completion_tokens"]) / runs["test_cases"]
    # Pareto frontier: no other priced run is both cheaper (or equal) and more accurate
    p = runs[runs.cost_known].sort_values(["total_cost", "pass_rate_2"], ascending=[True, False])
    best, front = -1.0, set()
    for idx, r in p.iterrows():
        if r.pass_rate_2 > best:
            front.add(idx); best = r.pass_rate_2
    runs["pareto"] = runs.index.isin(front)
    runs.to_csv(OUT, index=False)
    print(f"wrote {OUT} rows={len(runs)} priced={runs.cost_known.sum()} with_tokens={runs.has_tokens.sum()} pareto={runs.pareto.sum()}")
    if (runs.provider == "Other").any():
        print("UNMAPPED:", runs.loc[runs.provider == "Other", "model"].tolist(), file=sys.stderr)

if __name__ == "__main__":
    main()
