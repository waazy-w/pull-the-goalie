"""Pull the Goalie Sooner - Streamlit app.

    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pullgoalie.model import D_MAX, Rates, solve, threshold_curve  # noqa: E402

DATA = ROOT / "data" / "processed"
OPTIMAL = "#2a78d6"  # series 1: what the model says
ACTUAL = "#eb6834"  # series 2: what teams did
MUTED = "#8a8985"
GRID = "rgba(128,128,128,0.18)"

st.set_page_config(page_title="Pull the Goalie Sooner", page_icon="🥅", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load():
    summary = json.loads((DATA / "summary.json").read_text())
    tables = {
        name: pd.read_csv(DATA / f"{name}.csv")
        for name in ("audit", "leaderboard", "league_trend", "calibration", "team_strength", "bootstrap", "games", "rates", "curves")
    }
    tables["replay"] = pd.read_parquet(DATA / "replay.parquet")
    return summary, tables


SUMMARY, T = load()


def rates_obj() -> Rates:
    r = SUMMARY["rates"]
    conv = lambda d: {int(k): v for k, v in d.items()}  # noqa: E731
    return Rates(conv(r["trail_in_for"]), conv(r["trail_in_against"]), conv(r["trail_out_for"]), conv(r["trail_out_against"]), conv(r["tied_for"]))


OPP_PULL = {1: SUMMARY["league_median_pull_seconds"]["1"], 2: SUMMARY["league_median_pull_seconds"]["2"], 3: 255.0, 4: 305.0}
T_STAR = {int(k): v for k, v in SUMMARY["optimal_pull_seconds"].items()}
CI = {int(k): v for k, v in SUMMARY["optimal_pull_ci95"].items()}
MED = {int(k): v for k, v in SUMMARY["league_median_pull_seconds"].items()}


def mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def season_label(s: int) -> str:
    s = str(int(s))
    return f"{s[:4]}-{s[6:]}"


def style(fig: go.Figure, height: int = 360, title: str = "", **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, y=0.97, yanchor="top", yref="container", x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=90, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(size=13),
        **kw,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def solve_cached(off: float, dfn: float, ot_win: float, terminal: str = "points", policy: tuple | None = None, k_max: int = 120):
    s = solve(rates_obj(), k_max=k_max, ot_win=ot_win, our_off=off, our_def=dfn, opp_pull=OPP_PULL,
              our_policy=dict(policy) if policy else None, terminal=terminal)
    return s


@st.cache_data(show_spinner=False)
def live_curve(deficit: int, t_eval: int, off: float, dfn: float, step: int):
    return threshold_curve(rates_obj(), deficit, t_eval, OPP_PULL, off, dfn, step)


# ----------------------------------------------------------------------------- header
seasons = SUMMARY["seasons"]
st.title("🥅 Pull the Goalie Sooner")
st.markdown(
    f"**Every NHL coach knows they should empty the net earlier. This shows exactly how much waiting costs.**  \n"
    f"A dynamic-programming model of late-game hockey, fit to **{SUMMARY['games']:,} NHL games** "
    f"({season_label(seasons[0])} to {season_label(seasons[-1])}, current season through the latest final), "
    f"audits **{SUMMARY['decisions']:,}** real third-period decisions."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Optimal pull, down 1", mmss(T_STAR[1]), help="Even strength, league-average teams.")
c1.caption(f"95% CI {mmss(CI[1][0])} to {mmss(CI[1][1])} · NHL teams wait until **{mmss(MED[1])}**")
c2.metric("Optimal pull, down 2", mmss(T_STAR[2]))
c2.caption(f"95% CI {mmss(CI[2][0])} to {mmss(CI[2][1])} · NHL teams wait until **{mmss(MED[2])}**")
c3.metric("Decisions made late", f"{SUMMARY['share_late']:.0%}")
c3.caption(f"of {SUMMARY['decisions']:,} late-deficit stints")
c4.metric("Standings points lost / season", f"{SUMMARY['points_lost_per_season_league']:.1f}")
c4.caption("league-wide, from waiting too long")

tab_verdict, tab_calc, tab_audit, tab_replay, tab_method = st.tabs(
    ["📉 The verdict", "🧮 Bench calculator", "🏒 Team audit", "🎬 Game replay", "🔬 Method & validation"]
)

# ----------------------------------------------------------------------------- verdict
with tab_verdict:
    st.subheader("Waiting is the most expensive habit in hockey's last five minutes")
    left, right = st.columns(2)
    for col, d in ((left, 1), (right, 2)):
        t_eval = int(T["curves"].loc[T["curves"]["deficit"] == d, "t_eval"].iloc[0])
        cur = T["curves"][T["curves"]["deficit"] == d].reset_index(drop=True)
        best = cur.iloc[(cur["pull_at"] - T_STAR[d]).abs().argmin()]
        typical = cur.iloc[(cur["pull_at"] - MED[d]).abs().argmin()]
        fig = go.Figure()
        fig.add_scatter(x=cur["pull_at"], y=cur["comeback"] * 100, mode="lines", line=dict(color=OPTIMAL, width=2),
                        name="Chance to tie or win", hovertemplate="pull at %{customdata}: %{y:.1f}%<extra></extra>",
                        customdata=[mmss(x) for x in cur["pull_at"]], showlegend=False)
        fig.add_scatter(x=[best["pull_at"]], y=[best["comeback"] * 100], mode="markers+text", marker=dict(color=OPTIMAL, size=11),
                        text=[f"model: {mmss(T_STAR[d])}"], textposition="top center", showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=[typical["pull_at"]], y=[typical["comeback"] * 100], mode="markers+text", marker=dict(color=ACTUAL, size=11),
                        text=[f"NHL median: {mmss(MED[d])}"], textposition="bottom center", showlegend=False, hoverinfo="skip")
        style(fig, title=f"Down {d} with {mmss(t_eval)} left: when do you pull?")
        fig.update_xaxes(autorange="reversed", title="pull the goalie with this much time left",
                         tickvals=list(range(0, t_eval + 1, 60)), ticktext=[mmss(x) for x in range(0, t_eval + 1, 60)])
        lo, hi = cur["comeback"].min() * 100, cur["comeback"].max() * 100
        fig.update_yaxes(title="chance to reach OT or win (%)", range=[lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.25], ticksuffix="%")
        col.plotly_chart(fig, width="stretch")
        col.markdown(
            f"Pulling at **{mmss(T_STAR[d])}** instead of the league's typical **{mmss(MED[d])}** "
            f"lifts the comeback chance from **{typical['comeback']:.1%}** to **{best['comeback']:.1%}** "
            f"(+{best['points'] - typical['points']:.3f} expected standings points per game)."
        )

    st.markdown("#### Nobody is adjusting")
    tr = T["league_trend"]
    fig = go.Figure()
    for d, dash in ((1, "solid"), (2, "dot")):
        x = tr[tr["deficit"] == d]
        lab = [season_label(s) for s in x["season"]]
        fig.add_scatter(x=lab, y=x["median_actual"], mode="lines+markers", name=f"Actual median, down {d}",
                        line=dict(color=ACTUAL, width=2, dash=dash), marker=dict(size=9),
                        customdata=[mmss(v) for v in x["median_actual"]], hovertemplate="%{customdata}")
        fig.add_scatter(x=lab, y=x["optimal"], mode="lines", name=f"Optimal, down {d}",
                        line=dict(color=OPTIMAL, width=2, dash=dash), customdata=[mmss(v) for v in x["optimal"]], hovertemplate="%{customdata}")
    style(fig, height=340, title="Median pull time vs. optimal, by season")
    fig.update_yaxes(title="time left when goalie pulled", tickvals=list(range(0, 541, 60)), ticktext=[mmss(v) for v in range(0, 541, 60)], rangemode="tozero")
    st.plotly_chart(fig, width="stretch")
    st.caption("The 2026-27 point covers only the games played so far this season. Down-2 medians include stints that began late, when the right move was to pull immediately.")

# ----------------------------------------------------------------------------- calculator
with tab_calc:
    st.subheader("Bench calculator: should we pull right now?")
    ts = T["team_strength"]
    a, b, c = st.columns([1, 1, 1])
    with a:
        deficit = st.radio("We're down by", [1, 2, 3], horizontal=True, format_func=lambda d: f"{d} goal" + ("s" if d > 1 else ""))
        mins = st.slider("Time left in the 3rd (minutes)", 0.0, 12.0, 4.0, 0.25)
        t_left = int(round(mins * 60))
        st.caption(f"= {mmss(t_left)} remaining")
    with b:
        mp_choice = st.radio("Manpower", ["Even strength", "We're on the power play", "We're shorthanded"])
        pp_left = 0
        if mp_choice != "Even strength":
            pp_left = st.slider("Penalty time remaining (s)", 5, 120, 90, 5)
        k = pp_left if mp_choice.startswith("We're on") else -pp_left
    with c:
        team_opt = ["League-average team"] + [f"{r.team} {season_label(r.season)}" for r in ts.sort_values(["season", "team"], ascending=[False, True]).itertuples()]
        pick = st.selectbox("Team strength preset (shrunk 5v5 rates)", team_opt)
        if pick == "League-average team":
            off, dfn = 1.0, 1.0
        else:
            tm, sl = pick.split(" ")
            row = ts[(ts["team"] == tm) & (ts["season"].map(season_label) == sl)].iloc[0]
            off, dfn = float(row["off_factor"]), float(row["def_factor"])
        off = st.slider("Our scoring vs. league", 0.6, 1.4, round(off, 2), 0.01)
        dfn = st.slider("Our goals allowed vs. league", 0.6, 1.4, round(dfn, 2), 0.01)
        ot_win = st.slider("Our chance to win OT/shootout", 0.3, 0.7, 0.5, 0.01)

    sol = solve_cached(round(off, 2), round(dfn, 2), round(ot_win, 2))
    di, ki = D_MAX - deficit, k + sol.k_max
    t_left = max(t_left, 1)
    q_in, q_out = float(sol.Q_in[t_left, di, ki]), float(sol.Q_out[t_left, di, ki])
    t_pull = sol.pull_time(deficit, k)
    pull_now = q_out > q_in
    gain = abs(q_out - q_in)

    if pull_now:
        st.success(f"### ✅ PULL THE GOALIE NOW\nThe model would have pulled from **{mmss(min(t_pull, 1200))}** left in this situation. "
                   f"Every second you wait from here costs expected standings points.")
    else:
        st.info(f"### ⏳ Keep the goalie in, for now\nThe model starts pulling at **{mmss(min(t_pull, 1200))}** left in this situation. Pulling now would cost about **{gain * 100:.2f}** standings points per 100 times a coach faces this spot.")
    if mp_choice != "Even strength":
        st.caption("Power-play and shorthanded scoring rates come from far fewer minutes than even strength (see Method), so treat those recommendations as directional.")

    # expected points if we wait until T (pull at T), from now, even strength only
    if mp_choice == "Even strength" and t_left > 0:
        cur = live_curve(deficit, t_left, round(off, 2), round(dfn, 2), max(5, t_left // 60))
        fig = go.Figure()
        fig.add_scatter(x=cur["pull_at"], y=cur["comeback"] * 100, mode="lines", line=dict(color=OPTIMAL, width=2),
                        name="Chance to tie or win", customdata=[mmss(x) for x in cur["pull_at"]],
                        hovertemplate="pull at %{customdata}: %{y:.1f}%<extra></extra>", showlegend=False)
        fig.add_vline(x=MED.get(deficit, 255), line=dict(color=ACTUAL, width=2, dash="dot"))
        fig.add_annotation(x=MED.get(deficit, 255), y=1, yref="paper", text="NHL median", showarrow=False, font=dict(color=ACTUAL), xanchor="left")
        style(fig, height=320, title=f"From here ({mmss(t_left)}, down {deficit}): comeback chance by when you pull")
        fig.update_xaxes(autorange="reversed", title="pull with this much time left",
                         tickvals=list(range(0, t_left + 1, 60)), ticktext=[mmss(x) for x in range(0, t_left + 1, 60)])
        fig.update_yaxes(title="chance to reach OT or win", ticksuffix="%")
        st.plotly_chart(fig, width="stretch")

    with st.expander("Rule-of-thumb card for the bench"):
        rows = []
        for d in (1, 2):
            rows.append({"Situation": f"Down {d}, even strength", "Pull at": mmss(sol.pull_time(d, 0))})
            rows.append({"Situation": f"Down {d}, our PP (2:00 left on it)", "Pull at": mmss(min(sol.pull_time(d, 120), 1200))})
            rows.append({"Situation": f"Down {d}, we're shorthanded (2:00)", "Pull at": mmss(sol.pull_time(d, -120))})
        st.table(pd.DataFrame(rows))

# ----------------------------------------------------------------------------- audit
with tab_audit:
    st.subheader("Which teams leave the most points on the table?")
    lb = T["leaderboard"]
    s_opts = sorted(lb["season"].unique(), reverse=True)
    season = st.selectbox("Season", s_opts, index=1 if len(s_opts) > 1 else 0, format_func=season_label)
    x = lb[lb["season"] == season].sort_values("points_lost_per82", ascending=True)
    fig = go.Figure(go.Bar(
        x=x["points_lost_per82"], y=x["team"], orientation="h", marker=dict(color=ACTUAL, cornerradius=4),
        customdata=np.stack([x["decisions"], x["median_pull_down1"].map(mmss), x["median_pull_down2"].map(lambda v: mmss(v) if pd.notna(v) else "-")], axis=1),
        hovertemplate="<b>%{y}</b><br>%{x:.2f} pts lost per 82 GP<br>%{customdata[0]} decisions<br>median pull down 1: %{customdata[1]}<br>median pull down 2: %{customdata[2]}<extra></extra>",
    ))
    style(fig, height=760, title=f"Expected standings points lost to late pulls, {season_label(season)} (per 82 games)")
    fig.update_layout(hovermode="closest", bargap=0.25)
    fig.update_xaxes(title="standings points lost", gridcolor=GRID, showgrid=True)
    st.plotly_chart(fig, width="stretch")

    tsx = T["team_strength"][T["team_strength"]["season"] == season][["team", "opt_down1", "opt_down2", "off_factor", "def_factor"]]
    tbl = x.sort_values("points_lost_per82", ascending=False).merge(tsx, on="team", how="left")
    show = pd.DataFrame({
        "Team": tbl["team"],
        "Late-deficit decisions": tbl["decisions"],
        "Pts lost / 82": tbl["points_lost_per82"].round(2),
        "Median pull, down 1": tbl["median_pull_down1"].map(lambda v: mmss(v) if pd.notna(v) else "-"),
        "Their optimal, down 1": tbl["opt_down1"].map(lambda v: mmss(v) if pd.notna(v) else "-"),
        "Median pull, down 2": tbl["median_pull_down2"].map(lambda v: mmss(v) if pd.notna(v) else "-"),
        "Their optimal, down 2": tbl["opt_down2"].map(lambda v: mmss(v) if pd.notna(v) else "-"),
        "Avg. seconds late": tbl["avg_seconds_late"].round(0).astype(int),
    })
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("“Their optimal” adjusts the model for each team's shrunk 5v5 scoring and goals-allowed rates: strong attacks gain more from an extra skater and should pull earlier.")

# ----------------------------------------------------------------------------- replay
with tab_replay:
    st.subheader("Replay a decision")
    au = T["audit"]
    a1, a2 = st.columns(2)
    r_seasons = sorted(au["season"].unique(), reverse=True)
    r_season = a1.selectbox("Season ", r_seasons, index=1 if len(r_seasons) > 1 else 0, format_func=season_label, key="rs")
    teams = sorted(au.loc[au["season"] == r_season, "team"].unique())
    r_team = a2.selectbox("Team", teams, index=teams.index("NYR") if "NYR" in teams else 0)
    eps = au[(au["season"] == r_season) & (au["team"] == r_team)].sort_values("points_lost", ascending=False)
    labels = [
        f"{r.date} vs {r.opp}: down {r.deficit}, pulled {mmss(r.actual_time) if r.status != 'never pulled' else 'never'} (optimal {mmss(r.decision_time)}), -{r.points_lost:.3f} pts"
        + (" [imputed]" if r.status.startswith("censored") else "")
        for r in eps.itertuples()
    ]
    if not labels:
        st.info("No late-deficit decisions for this team yet.")
    else:
        idx = st.selectbox("Decision (most costly first)", range(len(labels)), format_func=lambda i: labels[i])
        ep = eps.iloc[idx]
        rp = T["replay"]
        g = rp[rp["game_id"] == ep["game_id"]].copy()
        side, opp = ep["side"], ("away" if ep["side"] == "home" else "home")
        g["diff"] = g[f"{side}_score"] - g[f"{opp}_score"]
        g = g[g["t0"] <= 720]

        fig = go.Figure()
        # score difference as a step line
        xs, ys = [], []
        for r in g.itertuples():
            xs += [r.t0, r.t1]
            ys += [r.diff, r.diff]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=MUTED, width=2, shape="hv"), name=f"{ep['team']} score margin", hoverinfo="skip", showlegend=False)
        pulled = g[g[f"{side}_pull"]]
        if len(pulled):
            for r in pulled.itertuples():
                fig.add_vrect(x0=r.t0, x1=r.t1, fillcolor=ACTUAL, opacity=0.25, line_width=0, layer="below")
        goals = g[g["goal_by"].notna()]
        for r in goals.itertuples():
            mine = r.goal_by == side
            fig.add_scatter(x=[r.t1], y=[r.diff + (1 if mine else -1)], mode="markers", marker=dict(size=12, symbol="circle" if mine else "x", color=OPTIMAL if mine else MUTED),
                            showlegend=False, hovertemplate=f"{'Goal ' + ep['team'] if mine else 'Goal ' + ep['opp']} at {mmss(r.t1)}<extra></extra>")
        fig.add_vline(x=ep["decision_time"], line=dict(color=OPTIMAL, width=2))
        fig.add_annotation(x=ep["decision_time"], y=1.02, yref="paper", text=f"model: pull at {mmss(ep['decision_time'])}", showarrow=False, font=dict(color=OPTIMAL), xanchor="right")
        if ep["status"] != "never pulled" and not str(ep["status"]).startswith("censored"):
            fig.add_vline(x=ep["actual_time"], line=dict(color=ACTUAL, width=2, dash="dot"))
            fig.add_annotation(x=ep["actual_time"], y=0.92, yref="paper", text=f"actual: {mmss(ep['actual_time'])}", showarrow=False, font=dict(color=ACTUAL), xanchor="left")
        style(fig, height=380, title=f"{ep['date']}: {ep['team']} vs {ep['opp']} - last 12:00 of the 3rd (shaded = net empty)")
        fig.update_xaxes(autorange="reversed", tickvals=list(range(0, 721, 60)), ticktext=[mmss(v) for v in range(0, 721, 60)], title="time left")
        fig.update_yaxes(title="goal margin", dtick=1)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch")

        sol = solve_cached(1.0, 1.0, 0.5, "comeback", None, 0)
        sol_pol = solve_cached(1.0, 1.0, 0.5, "comeback", ((int(ep["deficit"]), int(round(ep["actual_time"]))),), 0)
        p_opt = sol.value(int(ep["decision_time"]), -int(ep["deficit"]))
        p_act = sol_pol.value(int(ep["decision_time"]), -int(ep["deficit"]))
        m1, m2, m3 = st.columns(3)
        m1.metric("Comeback chance, optimal timing", f"{p_opt:.1%}")
        m2.metric("Comeback chance, their timing", f"{p_act:.1%}", f"{(p_act - p_opt) * 100:+.1f} pts", delta_color="normal")
        m3.metric("Expected standings points lost", f"{ep['points_lost']:.3f}")
        st.caption("Chances are evaluated at the moment the model would have pulled, for a league-average team. "
                   "[imputed] = the score changed before this team pulled, so we used its typical pull time.")

# ----------------------------------------------------------------------------- method
with tab_method:
    st.subheader("How it works")
    st.markdown(
        """
1. **Data.** Every regular-season play from the public NHL API. Its `situationCode` says whether each goalie is in net on every play,
   so we recover the exact seconds a net was empty. Empty nets during **delayed penalties** (the feed's `delayed-penalty` event, or a
   short empty-net run ended by an opponent penalty) are excluded: only *strategic* pulls count.
2. **Scoring rates.** In the last 10:00 of the third, goals for and against the trailing team are Poisson processes whose rate depends on
   goalie in/out and manpower (even, power play, shorthanded). Sparse cells are shrunk toward even-strength rates (30-minute gamma prior).
3. **Decision model.** A finite-horizon Markov decision process on a 1-second grid over *(time left, goal margin, power-play clock)*.
   The reward is **standings points** (win 2, overtime loss 1), not win probability, because reaching overtime is worth a point.
   When we lead, the opponent pulls at the league's observed median time.
4. **Uncertainty.** 200 bootstrap resamples of games give 95% intervals on the optimal pull time.
5. **Audit.** Each third-period stint spent trailing by 1 or 2 that reaches the optimal pull time is a decision. Cost = expected points under
   the optimal policy minus expected points if you pull when the team actually did, from the same model.
6. **Validation.** Rates fit on 2023-24 and 2024-25 only, then tested on later seasons' real pulls (below).
"""
    )
    st.markdown("#### Late-game scoring rates (goals per 60 minutes, trailing team)")
    rt = T["rates"].copy()
    rt.columns = ["Situation", "Goalie in: GF/60", "Goalie in: GA/60", "Pulled: GF/60", "Pulled: GA/60", "Minutes (in)", "Minutes (pulled)"]
    st.dataframe(rt.round(2), hide_index=True, width="stretch")

    m1, m2 = st.columns(2)
    with m1:
        bs = T["bootstrap"]
        fig = go.Figure()
        fig.add_histogram(x=bs["down1"], name="down 1", marker=dict(color=OPTIMAL), opacity=0.85, nbinsx=25)
        fig.add_histogram(x=bs["down2"], name="down 2", marker=dict(color="#6250d6"), opacity=0.85, nbinsx=25)
        style(fig, height=320, title="Bootstrap: optimal pull time (200 resamples)")
        fig.update_layout(barmode="overlay", hovermode="closest", bargap=0.05)
        fig.update_xaxes(title="seconds left", autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    with m2:
        cal = T["calibration"]
        fig = go.Figure()
        fig.add_scatter(x=[0, 0.3], y=[0, 0.3], mode="lines", line=dict(color=MUTED, width=1, dash="dot"), name="perfect", hoverinfo="skip")
        fig.add_scatter(x=cal["predicted"], y=cal["actual"], mode="markers+lines", marker=dict(size=10, color=OPTIMAL), line=dict(color=OPTIMAL, width=2),
                        name="model", customdata=cal["n"], hovertemplate="predicted %{x:.1%}<br>actual %{y:.1%}<br>n=%{customdata}<extra></extra>")
        style(fig, height=320, title="Out-of-sample calibration at the moment of each pull")
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="predicted chance to tie or win", tickformat=".0%")
        fig.update_yaxes(title="actual", tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    calm = SUMMARY["calibration"]
    st.caption(f"Brier score {calm['brier']:.4f} vs {calm['brier_baseline']:.4f} for a constant-rate baseline, on {calm['n']} held-out pulls.")

    st.markdown(
        """
#### Limitations (what a skeptical analytics department should know)
* **Zone and possession are not modeled.** Coaches tend to pull with the puck in the offensive zone, and the model under-predicts comebacks in the
  highest-probability bin (actual 25% vs predicted 15%). Pulled scoring rates are averaged over all empty-net time.
* **Score effects are pooled** across deficits of 1 to 3 goals; the deficit-specific rates in the data are similar.
* **Team strength** uses full-season 5v5 rates, not the specific players on the ice; the audit itself uses a league-average team.
* Penalties drawn *after* the decision are not modeled, and power-play/shorthanded rates rest on hundreds, not thousands, of minutes.
"""
    )
    st.caption("Data: NHL public API (api-web.nhle.com). Open source: model, data pipeline, and this app are in the GitHub repository.")
