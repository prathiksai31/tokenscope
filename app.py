"""TokenScope: choosing and budgeting models for AI coding agents.

Run locally:  pip install -r requirements.txt  &&  streamlit run app.py
Data: data/runs.csv (built by src/build_dataset.py from the pinned Aider polyglot snapshot)
      data/pricing_anthropic_2026-11-05.csv (Anthropic list prices, retrieved Nov 5, 2026)
"""
import pathlib

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
ACCENT, GREY, DARK = "#2a78d6", "#b9b7b0", "#3b3a37"
OVER, UNDER = "#d03b3b", "#2a78d6"
PROVIDER_COLORS = {"OpenAI": "#2a78d6", "Anthropic": "#eb6834", "Google": "#1baf7a", "Other": "#b9b7b0"}
EXERCISES = 225

st.set_page_config(page_title="TokenScope", layout="wide")
alt.data_transformers.disable_max_rows()


# ---------------------------------------------------------------- data
@st.cache_data
def load_runs() -> pd.DataFrame:
    r = pd.read_csv(ROOT / "data" / "runs.csv", parse_dates=["date"])
    r["family"] = r["model"].str.replace(r"\s*\(.*\)\s*$", "", regex=True)
    r["provider_group"] = r["provider"].where(r["provider"].isin(["OpenAI", "Anthropic", "Google"]), "Other")
    r["cost_per_exercise"] = r["total_cost"] / EXERCISES
    r["seconds_per_case"] = r["seconds_per_case"].round(1)
    r["label"] = r["model"]
    return r


@st.cache_data
def load_prices() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "pricing_anthropic_2026-11-05.csv")


runs = load_runs()
prices = load_prices()
qp = st.query_params

# ---------------------------------------------------------------- header
st.markdown("### TokenScope · choosing and budgeting models for AI coding agents")
st.caption(
    "69 benchmark runs from the Aider polyglot leaderboard (225 exercises each, Dec 2024 – Oct 2025). "
    "Brush or click runs in the frontier; the other views and the budget estimate follow your selection."
)
with st.expander("About the data and its limits"):
    st.markdown(
        "- **Observed:** % solved, cost, seconds per exercise, tokens and edit-format errors come from the "
        "benchmark file (pinned at commit 5dc9490).\n"
        "- **Reference:** Anthropic list prices retrieved Nov 5, 2026, used only when you choose to re-price.\n"
        "- **Hypothetical:** tasks per month, budget and cache-hit share are your inputs.\n"
        "- 18 runs have no cost ($0 or missing) and are shown as *cost unknown*, never imputed. "
        "Only 23 runs report token counts. The snapshot has no 2026 models."
    )

# ---------------------------------------------------------------- filter bar
f1, f2, f3, f4 = st.columns([2.2, 1.6, 1.3, 1.3])
providers = sorted(runs["provider"].unique())
sel_prov = f1.multiselect("Provider", providers, default=providers, placeholder="All providers")
formats = sorted(runs["edit_format"].unique())
sel_fmt = f2.multiselect("Edit format", formats, default=formats)
max_sec = f3.slider("Max seconds per exercise", 10, 720, int(qp.get("maxsec", 720)), step=10)
min_pass = f4.slider("Min % solved", 0, 100, int(qp.get("minpass", 0)), step=5)

view = runs[
    runs["provider"].isin(sel_prov)
    & runs["edit_format"].isin(sel_fmt)
    & (runs["seconds_per_case"].fillna(0) <= max_sec)
    & (runs["pass_rate_2"] >= min_pass)
].copy()
priced = view[view["cost_known"]].copy()
# frontier recomputed on what is visible: a run is on it if no visible run is cheaper (or equal) and more accurate
_best, _front = -1.0, set()
for _i, _r in priced.sort_values(["total_cost", "pass_rate_2"], ascending=[True, False]).iterrows():
    if _r.pass_rate_2 > _best:
        _front.add(_i); _best = _r.pass_rate_2
priced["pareto"] = priced.index.isin(_front)
view["pareto"] = view.index.isin(_front)
st.caption(f"Showing **{len(view)} of {len(runs)} runs** · {len(priced)} with a known cost · "
           f"{int(view['pareto'].sum())} on the cost–accuracy frontier")

# ---------------------------------------------------------------- V1 frontier
brush_init = None
if "brush" in qp:  # e.g. ?brush=0.3,20,75,100 (cost lo, cost hi, % lo, % hi) for a shareable state
    a, b, c, d = (float(x) for x in qp["brush"].split(","))
    brush_init = {"x": [a, b], "y": [c, d]}
brush = alt.selection_interval(name="brush", encodings=["x", "y"], value=brush_init)
pick_init = None
if "pick" in qp:  # e.g. ?pick=gpt-5 (low)|gpt-5 (medium)
    pick_init = [{"dirname": d} for d in runs.loc[runs["model"].isin(qp["pick"].split("|")), "dirname"]]
pick = alt.selection_point(name="pick", fields=["dirname"], on="click", toggle="true", value=pick_init)

color_mode = st.radio("Colour runs by", ["Frontier", "Provider"], horizontal=True, label_visibility="collapsed",
                      index=1 if qp.get("color") == "provider" else 0)
if color_mode == "Frontier":
    color = alt.condition("datum.pareto", alt.value(ACCENT), alt.value(GREY))
else:
    color = alt.Color("provider_group:N", title="Provider",
                      scale=alt.Scale(domain=list(PROVIDER_COLORS), range=list(PROVIDER_COLORS.values())),
                      legend=alt.Legend(orient="bottom-right"))

tooltip = [
    alt.Tooltip("model:N", title="Model"), alt.Tooltip("provider:N", title="Provider"),
    alt.Tooltip("pass_rate_2:Q", title="% solved (2 tries)", format=".1f"),
    alt.Tooltip("pass_rate_1:Q", title="% solved (1 try)", format=".1f"),
    alt.Tooltip("total_cost:Q", title="Run cost ($)", format=",.2f"),
    alt.Tooltip("cost_per_solved_task:Q", title="$ per solved exercise", format=",.3f"),
    alt.Tooltip("seconds_per_case:Q", title="Seconds per exercise"),
    alt.Tooltip("edit_format:N", title="Edit format"),
    alt.Tooltip("date:T", title="Run date", format="%b %d, %Y"),
]
X = alt.X("total_cost:Q", scale=alt.Scale(type="log", domain=[0.1, 300]), title="Total cost of the 225-exercise run ($, log scale)",
          axis=alt.Axis(values=[0.1, 0.3, 1, 3, 10, 30, 100], format="$,.2~f"))
Y = alt.Y("pass_rate_2:Q", scale=alt.Scale(domain=[0, 100]), title="Exercises solved within 2 attempts (%)")

frontier_line = alt.Chart(priced).transform_filter("datum.pareto").mark_line(interpolate="step-after", color=ACCENT, strokeWidth=1.6).encode(x=X, y=Y, order="total_cost:Q")
pts = alt.Chart(priced).mark_circle(size=90, stroke="white", strokeWidth=0.8).encode(
    x=X, y=Y, color=color, tooltip=tooltip,
    opacity=alt.condition(brush & pick, alt.value(0.95), alt.value(0.22)),
    size=alt.condition(pick, alt.value(150), alt.value(90)),
).add_params(brush, pick)
# frontier labels: shorten model ids, then place each label greedily where it hits the fewest points/labels
LABEL_SPOTS = [("left", 8, 4), ("left", 8, -9), ("right", -8, -7), ("right", -8, 8), ("center", 0, -13), ("center", 0, 15)]
def place_labels(df, mask, text, x="total_cost", y="pass_rate_2", xdom=(0.1, 300), ydom=(0, 100), W=720, H=430):
    """Greedy label placement: for each labelled point pick the spot that hits the fewest points/labels."""
    def px(c, v):
        return ((np.log10(c) - np.log10(xdom[0])) / (np.log10(xdom[1]) - np.log10(xdom[0])) * W,
                (1 - (v - ydom[0]) / (ydom[1] - ydom[0])) * H)
    pts_px = [px(c, v) for c, v in zip(df[x], df[y])]
    boxes, out = [], pd.Series(-1, index=df.index)
    for idx, r in df[mask].sort_values(x).iterrows():
        cx, cy = px(r[x], r[y])
        w = 5.4 * len(r[text]) + 2
        best = None
        for k, (al, dx, dy) in enumerate(LABEL_SPOTS):
            x0 = cx + dx if al == "left" else (cx + dx - w if al == "right" else cx - w / 2)
            b = (x0, cy + dy - 6, x0 + w, cy + dy + 6)
            hit = sum(b[0] - 5 < qx < b[2] + 5 and b[1] - 5 < qy < b[3] + 5 for qx, qy in pts_px if (qx, qy) != (cx, cy))
            hit += 10 * sum(not (b[2] < o[0] or o[2] < b[0] or b[3] < o[1] or o[3] < b[1]) for o in boxes)
            hit += 10 * (b[0] < 0 or b[2] > W or b[1] < 0 or b[3] > H)
            if best is None or hit < best[0]:
                best = (hit, k, b)
        boxes.append(best[2])
        out[idx] = best[1]
    return out


def label_layers(data, text, x, y):
    return alt.layer(*[
        alt.Chart(data).transform_filter(f"datum.lpos == {k}").mark_text(align=al, dx=dx, dy=dy, fontSize=10, color="#52514e").encode(x=x, y=y, text=f"{text}:N")
        for k, (al, dx, dy) in enumerate(LABEL_SPOTS)
    ])


DATE_RE = r"-20\d{2}-?\d{2}-?\d{2}"
priced["short"] = priced["model"].str.replace(DATE_RE, "", regex=True)
priced["lpos"] = place_labels(priced, priced["pareto"], "short")
labels = label_layers(priced, "short", X, Y)
v1 = (frontier_line + pts + labels).properties(width=720, height=430, title="V1 · Accuracy vs. cost (blue step line = frontier: no run is both cheaper and more accurate)")

unknown = view[~view["cost_known"]].assign(jit=lambda d: np.random.default_rng(1).uniform(-1, 1, len(d)))
strip = alt.Chart(unknown).mark_point(size=60, color="#898781", filled=False).encode(
    x=alt.X("jit:Q", axis=None, scale=alt.Scale(domain=[-2, 2])), y=alt.Y("pass_rate_2:Q", scale=alt.Scale(domain=[0, 100]), axis=None),
    tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("pass_rate_2:Q", title="% solved", format=".1f")],
).properties(width=60, height=430, title=alt.TitleParams(["Cost", f"unknown (n={len(unknown)})"], fontSize=12))

c1, c2 = st.columns([2.8, 1.2])
with c1:
    event = st.altair_chart(alt.hconcat(v1, strip).resolve_scale(color="independent"), use_container_width=True,
                            on_select="rerun", selection_mode=["brush", "pick"], key="v1")

# ---------------------------------------------------------------- shared selection
sel = event.selection if event else {}
picked = [p.get("dirname") for p in (sel.get("pick") or []) if p.get("dirname")]
b = sel.get("brush") or ({} if brush_init is None else {"total_cost": brush_init["x"], "pass_rate_2": brush_init["y"]})
if picked:
    shortlist = priced[priced["dirname"].isin(picked)]
    how = "clicked"
elif b.get("total_cost"):
    lo, hi = b["total_cost"]; plo, phi = b.get("pass_rate_2", [0, 100])
    shortlist = priced[priced["total_cost"].between(lo, hi) & priced["pass_rate_2"].between(plo, phi)]
    how = "brushed"
else:
    shortlist = priced[priced["pareto"]]
    how = "default: frontier runs"
if "pick" in qp and not sel.get("pick") and not sel.get("brush"):  # URL preset, until the user selects something
    shortlist = priced[priced["model"].isin(qp["pick"].split("|"))]
    how = "clicked"
shortlist = shortlist.sort_values("pass_rate_2", ascending=False)
sel_dirs = set(shortlist["dirname"])
view["selected"] = view["dirname"].isin(sel_dirs)

with c2:
    st.markdown(f"**Selection** ({how}, {len(shortlist)} runs)")
    st.dataframe(
        shortlist[["short", "pass_rate_2", "total_cost", "seconds_per_case"]].rename(
            columns={"short": "Model", "pass_rate_2": "% solved", "total_cost": "Run $", "seconds_per_case": "s/ex"}),
        hide_index=True, height=min(405, 35 * (len(shortlist) + 1) + 3), use_container_width=True,
        column_config={"Model": st.column_config.TextColumn(width=190),
                       "% solved": st.column_config.NumberColumn(format="%.1f", width=60),
                       "Run $": st.column_config.NumberColumn(format="$%.2f", width=64),
                       "s/ex": st.column_config.NumberColumn(format="%.0f", width=44)},
    )

# ---------------------------------------------------------------- V2 configs, retries, formats
left, right = st.columns([1.15, 1])
sel_color = alt.condition("datum.selected", alt.value(ACCENT), alt.value(GREY))
with left:
    st.markdown("**V2 · Configurations, retries and edit formats**")
    t1, t2, t3 = st.tabs(["Reasoning settings", "Retries", "Edit formats"])
    with t1:
        fam = priced.copy()
        multi = fam["family"].value_counts()
        fam = fam[fam["family"].isin(multi[multi > 1].index)]
        fam["selected"] = fam["family"].isin(shortlist["family"])
        fam["flabel"] = fam["family"].str.replace(DATE_RE, "", regex=True)
        _top = fam.groupby("family")["total_cost"].transform("max") == fam["total_cost"]
        fam["lpos"] = place_labels(fam, _top, "flabel", xdom=(0.1, 300), ydom=(20, 100), W=600, H=280)
        fx = alt.X("total_cost:Q", scale=alt.Scale(type="log", domain=[0.1, 300]), title="Run cost ($, log)", axis=alt.Axis(format="$,.2~f"))
        fy = alt.Y("pass_rate_2:Q", title="% solved", scale=alt.Scale(domain=[20, 100]))
        lines = alt.Chart(fam).mark_line(point=True, strokeWidth=1.6).encode(
            x=fx, y=fy, detail="family:N", order="total_cost:Q", color=sel_color,
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("pass_rate_2:Q", format=".1f", title="% solved"),
                     alt.Tooltip("total_cost:Q", format="$,.2f", title="Run cost")])
        ends = label_layers(fam, "flabel", fx, fy)
        st.altair_chart((lines + ends).properties(height=330, title="Same model, more reasoning: what does each step cost?"), use_container_width=True)
        st.caption(f"{fam['family'].nunique()} model families with more than one priced configuration. Families in your selection are blue.")
    with t2:
        rr = view.sort_values("retry_gain_pp", ascending=True)
        order = rr["model"].tolist()
        yy = alt.Y("model:N", sort=order[::-1], title=None, axis=alt.Axis(labelLimit=220, labelFontSize=10, labelOverlap=False))
        rule = alt.Chart(rr).mark_rule(strokeWidth=2).encode(y=yy, x=alt.X("pass_rate_1:Q", title="% solved", scale=alt.Scale(domain=[0, 100])), x2="pass_rate_2:Q", color=sel_color)
        p1 = alt.Chart(rr).mark_point(filled=False, size=40).encode(y=yy, x="pass_rate_1:Q", color=sel_color, tooltip=["model", alt.Tooltip("pass_rate_1", title="1 try"), alt.Tooltip("pass_rate_2", title="2 tries"), alt.Tooltip("retry_gain_pp", title="gain (pp)")])
        p2 = alt.Chart(rr).mark_point(filled=True, size=40).encode(y=yy, x="pass_rate_2:Q", color=sel_color)
        with st.container(height=330):
            st.altair_chart((rule + p1 + p2).properties(height=16 * len(rr), title="Attempt 1 (hollow) → attempt 2 (filled), sorted by gain"), use_container_width=True)
    with t3:
        fm = alt.Chart(view).mark_tick(thickness=2, size=22).encode(
            x=alt.X("percent_cases_well_formed:Q", title="Well-formed edits (%)", scale=alt.Scale(domain=[60, 100])),
            y=alt.Y("edit_format:N", title=None, sort="-x"), color=sel_color,
            tooltip=["model", "edit_format", alt.Tooltip("percent_cases_well_formed", title="% well-formed")])
        med = alt.Chart(view).mark_point(shape="diamond", size=110, filled=True, color=DARK).encode(
            x="median(percent_cases_well_formed):Q", y=alt.Y("edit_format:N", sort="-x"))
        st.altair_chart((fm + med).properties(height=300, title="Edit-format reliability per run (diamond = median)"), use_container_width=True)

# ---------------------------------------------------------------- V3 budget
with right:
    st.markdown("**V3 · Monthly cost estimate**")
    i1, i2 = st.columns(2)
    tasks = i1.number_input("Tasks per month (your input)", 100, 200000, int(qp.get("tasks", 2000)), step=100)
    budget = i2.number_input("Monthly budget, $ (your input)", 0, 1_000_000, int(qp.get("budget", 500)), step=50)
    basis = st.radio("Cost basis", ["Observed benchmark cost", "Re-price with today's Anthropic list prices"], horizontal=True,
                     index=1 if qp.get("basis") == "reprice" else 0)
    calc = shortlist.copy()
    if basis.startswith("Re-price"):
        _pl = prices["model"].tolist()
        preset = st.selectbox("Price preset (reference)", _pl, index=_pl.index(qp["preset"]) if qp.get("preset") in _pl else 1)
        p = prices.set_index("model").loc[preset]
        h = st.slider("Cache-hit share of input tokens (hypothetical)", 0.0, 0.9, 0.0, 0.05)
        st.caption("Assumes warm cache reads at the cache-read price; cache-write charges are excluded.")
        calc = calc[calc["has_tokens"]].copy()
        tin, tout = calc["prompt_tokens"] / EXERCISES, calc["completion_tokens"] / EXERCISES
        calc["cost_per_exercise"] = (tin * ((1 - h) * p.input_usd_per_mtok + h * p.cache_read_usd_per_mtok) + tout * p.output_usd_per_mtok) / 1e6
        if len(calc) < len(shortlist):
            st.caption(f"{len(shortlist) - len(calc)} selected runs have no token counts and are left out of the re-priced estimate.")
    calc["monthly"] = calc["cost_per_exercise"] * tasks
    calc["solved"] = (calc["pass_rate_2"] / 100 * tasks).round()
    calc["status"] = np.where(calc["monthly"] > budget, "Over budget", "Within budget")
    calc["lbl"] = calc.apply(lambda r: f"${r.monthly:,.0f} · ≈{r.solved:,.0f} solved", axis=1)
    if len(calc):
        yb = alt.Y("model:N", sort=calc.sort_values("monthly", ascending=False)["model"].tolist(), title=None, axis=alt.Axis(labelLimit=200), scale=alt.Scale(paddingInner=0.35))
        bars = alt.Chart(calc).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("monthly:Q", title="Estimated monthly cost ($)", scale=alt.Scale(domainMax=float(max(budget * 1.15, calc["monthly"].max() * 1.35)))), y=yb,
            color=alt.Color("status:N", scale=alt.Scale(domain=["Within budget", "Over budget"], range=[UNDER, OVER]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["model", alt.Tooltip("monthly:Q", format="$,.0f", title="Est. monthly $"), alt.Tooltip("solved:Q", title="≈ solved / month"), alt.Tooltip("pass_rate_2:Q", title="% solved", format=".1f")])
        txt = bars.mark_text(align="left", dx=4, fontSize=10, color="#52514e").encode(text="lbl:N", color=alt.value("#52514e"))
        rule = alt.Chart(pd.DataFrame({"b": [budget]})).mark_rule(strokeDash=[5, 4], color=DARK).encode(x="b:Q")
        rule_t = alt.Chart(pd.DataFrame({"b": [budget], "t": [f"budget ${budget:,}"]})).mark_text(align="left", dx=4, dy=-4, fontSize=10, color=DARK).encode(x="b:Q", y=alt.value(-6), text="t:N")
        bh = max(120, 30 * len(calc))
        st.altair_chart((bars + txt + rule + rule_t).properties(height=bh), use_container_width=True, height=bh + 90)
    else:
        st.info("Select runs with a known cost in V1 to see an estimate.")
    st.caption(f"Estimated monthly cost, assuming each task uses as many tokens as one benchmark exercise "
               f"({'observed cost per exercise' if basis.startswith('Observed') else 'real token counts × reference prices'} × {tasks:,} tasks).")
    tok = shortlist[shortlist["has_tokens"]].copy()
    if len(tok):
        tl = tok.melt(id_vars=["model"], value_vars=["prompt_tokens", "completion_tokens"], var_name="kind", value_name="tokens")
        tl["kind"] = tl["kind"].map({"prompt_tokens": "Input", "completion_tokens": "Output"})
        tl["k_per_ex"] = tl["tokens"] / EXERCISES / 1000
        st.altair_chart(alt.Chart(tl).mark_bar().encode(
            x=alt.X("k_per_ex:Q", title="Thousand tokens per exercise", stack="zero"), y=alt.Y("model:N", title=None, axis=alt.Axis(labelLimit=200), scale=alt.Scale(paddingInner=0.35)),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["Input", "Output"], range=[ACCENT, "#eb6834"]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["model", "kind", alt.Tooltip("k_per_ex:Q", format=".1f", title="K tokens / exercise")]).properties(height=max(60, 30 * len(tok)), title=f"Where the tokens go (n={len(tok)} selected runs with token counts)"),
            use_container_width=True, height=max(60, 30 * len(tok)) + 190)
