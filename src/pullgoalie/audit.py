"""Audit every late-game deficit: when did the team pull vs. when should it?

A *stint* is a stretch of the third period during which a team trails by
exactly d goals. A stint is a decision point if it reaches the optimal pull
time T*(d). The decision time is min(T*, stint start): if you fall behind by
two with 9:00 left and T*(2) is 13:00, the right move is to pull right away.

Cost of a late pull = V*(decision time) - V(pull at the actual time), where
both values come from the same dynamic program (expected standings points
for a league-average team). If a goal changed the score before the team
pulled, we never saw when they *would* have pulled; we impute the team's
typical pull time for that deficit, capped at the moment the goal came.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import D_MAX, Rates, Solution, solve

POLICY_GRID = 5  # seconds between evaluated pull-time thresholds


def add_team_ids(tv: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games.set_index("game_id")
    tv = tv.copy()
    home = tv["side"].eq("home")
    tv["team"] = np.where(home, tv["game_id"].map(g["home"]), tv["game_id"].map(g["away"]))
    tv["opp"] = np.where(home, tv["game_id"].map(g["away"]), tv["game_id"].map(g["home"]))
    return tv


def find_pulls(tv: pd.DataFrame) -> pd.DataFrame:
    """First strategic pull of each (game, team, deficit stint) in the 3rd period."""
    # Intervals are already in play order within each game; keep that order.
    v = tv[(tv["period"] == 3)].sort_values(["game_id", "side"], kind="stable")
    rows = []
    for (gid, side), x in v.groupby(["game_id", "side"], sort=False):
        x = x[x["dur"] >= 0]
        diff = x["diff"].to_numpy()
        t0 = x["t0"].to_numpy()
        t1 = x["t1"].to_numpy()
        pulled = x["pulled"].to_numpy()
        mp = x["mp"].to_numpy()
        n = len(x)
        i = 0
        while i < n:
            d = diff[i]
            j = i
            while j + 1 < n and diff[j + 1] == d:
                j += 1
            if d < 0:
                start, end = t0[i], t1[j]
                pidx = np.flatnonzero(pulled[i : j + 1])
                pull_t = None
                pull_mp = None
                if len(pidx):
                    k = i + pidx[0]
                    pull_t = (t0[k] + t1[k]) / 2.0
                    pull_mp = int(mp[k])
                ended_by_goal = j + 1 < n or (x["gf"].to_numpy()[j] or x["ga"].to_numpy()[j])
                rows.append(
                    {
                        "game_id": gid,
                        "side": side,
                        "team": x["team"].iat[0],
                        "opp": x["opp"].iat[0],
                        "season": x["season"].iat[0],
                        "deficit": int(-d),
                        "stint_start": float(start),
                        "stint_end": float(end),
                        "pull_time": pull_t,
                        "pull_mp": pull_mp,
                        "ended_by_goal": bool(ended_by_goal and end > 0),
                    }
                )
            i = j + 1
    return pd.DataFrame(rows)


def policy_values(rates: Rates, deficit: int, opp_pull: dict, horizon: int = 1200) -> dict[int, np.ndarray]:
    """V(t, -deficit) under "pull at T" for every T on a grid (even strength)."""
    out = {}
    for T in range(0, horizon + 1, POLICY_GRID):
        s = solve(rates, horizon=horizon, k_max=0, opp_pull=opp_pull, our_policy={deficit: T})
        out[T] = s.V[:, D_MAX - deficit, 0]
    return out


def _lookup(pv: dict[int, np.ndarray], T: float, t: float) -> float:
    Tg = int(round(T / POLICY_GRID) * POLICY_GRID)
    Tg = min(max(Tg, 0), max(pv))
    return float(pv[Tg][int(round(t))])


def audit(stints: pd.DataFrame, opt: Solution, pvs: dict, deficits=(1, 2)) -> pd.DataFrame:
    s = stints[stints["deficit"].isin(deficits)].copy()
    s["opt_time"] = s["deficit"].map({d: opt.pull_time(d) for d in deficits})
    # Decision points: the stint is still alive at the optimal pull time.
    s = s[s["stint_end"] < s["opt_time"]].copy()
    s["decision_time"] = np.minimum(s["opt_time"], s["stint_start"])

    pulled = s["pull_time"].notna()
    # Typical observed pull time per team-season-deficit, for censored stints.
    observed = s[pulled | ~s["ended_by_goal"]].assign(pt=lambda x: x["pull_time"].fillna(0))
    team_med = observed.groupby(["season", "team", "deficit"])["pt"].median()
    league_med = observed.groupby(["season", "deficit"])["pt"].median()

    status, actual, cost = [], [], []
    for r in s.itertuples():
        V_star = opt.value(int(r.decision_time), -r.deficit)
        if r.pull_time is not None and not np.isnan(r.pull_time):
            T = r.pull_time
            st = "on time" if T >= r.decision_time - 5 else "late"
        elif not r.ended_by_goal:
            T, st = 0.0, "never pulled"
        else:
            T = team_med.get((r.season, r.team, r.deficit), league_med.get((r.season, r.deficit), 0.0))
            T = min(T, r.stint_end)
            st = "censored (imputed)"
        if T > r.decision_time:  # pulled before the optimal moment
            t_eval = int(round(T))
            c = opt.value(t_eval, -r.deficit) - float(opt.Q_out[t_eval, D_MAX - r.deficit, opt.k_max])
            st = "early" if T > r.decision_time + 5 else st
        else:
            c = V_star - _lookup(pvs[r.deficit], T, r.decision_time)
        status.append(st)
        actual.append(T)
        cost.append(max(c, 0.0))
    s["status"] = status
    s["actual_time"] = actual
    s["seconds_late"] = np.maximum(s["decision_time"] - s["actual_time"], 0)
    s["points_lost"] = cost
    return s


def team_leaderboard(a: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    gp = pd.concat([games[["season", "home"]].rename(columns={"home": "team"}), games[["season", "away"]].rename(columns={"away": "team"})])
    gp = gp.groupby(["season", "team"]).size().rename("gp")
    obs = a[a["status"] != "censored (imputed)"]
    lb = a.groupby(["season", "team"]).agg(
        decisions=("points_lost", "size"),
        points_lost=("points_lost", "sum"),
        avg_seconds_late=("seconds_late", "mean"),
    )
    med = obs.groupby(["season", "team", "deficit"])["actual_time"].median().unstack()
    lb["median_pull_down1"] = med.get(1)
    lb["median_pull_down2"] = med.get(2)
    lb = lb.join(gp)
    lb["points_lost_per82"] = lb["points_lost"] / lb["gp"] * 82
    return lb.reset_index().sort_values(["season", "points_lost_per82"], ascending=[True, False])
