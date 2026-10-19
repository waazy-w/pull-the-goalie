"""Run the whole analysis and write the small files the app reads.

    python scripts/fetch_data.py      # once (or to refresh the current season)
    python scripts/build_outputs.py

Outputs (data/processed/):
    summary.json         headline numbers, rates, optimal pull times + 95% CIs
    rates.csv            late-game scoring rates by state
    bootstrap.csv        bootstrap draws of the optimal pull times
    audit.csv            every late-deficit decision with points lost
    leaderboard.csv      team-season totals
    league_trend.csv     actual vs optimal pull time by season
    calibration.csv      out-of-sample predicted vs actual comeback rates
    team_strength.csv    shrunk 5v5 offense/defense factors + team-specific T*
    replay.csv.gz        third-period intervals for games in the audit
    games.csv            game results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pullgoalie.audit import add_team_ids, audit, find_pulls, policy_values, team_leaderboard  # noqa: E402
from pullgoalie.fetch import load_season  # noqa: E402
from pullgoalie.model import late_counts, rates_from_counts, solve, threshold_curve  # noqa: E402
from pullgoalie.parse import build_intervals, game_table, team_view  # noqa: E402

OUT = ROOT / "data" / "processed"
TRAIN_SEASONS = (20232024, 20242025)  # for the out-of-sample calibration
N_BOOT = 200
RNG = np.random.default_rng(2026)
VERDICT_CURVES = ((1, 420), (2, 720))  # (deficit, seconds left when the curve starts)


def observed_pull_medians(stints: pd.DataFrame) -> dict:
    """League median pull time by deficit (stints that saw a pull), used as
    the opponent's behaviour when we are the team leading."""
    p = stints[stints["pull_time"].notna()]
    med = p.groupby("deficit")["pull_time"].median()
    return {int(d): float(med.get(d, med.max())) for d in (1, 2, 3, 4)}


def team_strength(tv: pd.DataFrame, prior_minutes: float = 600.0) -> pd.DataFrame:
    """Shrunk 5v5 GF/60 and GA/60 multipliers per team-season (gamma-Poisson)."""
    ev = tv[(tv["sk"] == 5) & (tv["opp_sk"] == 5) & ~tv["net_empty"] & ~tv["opp_net_empty"]]
    g = ev.groupby(["season", "team"]).agg(minutes=("dur", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
    g["minutes"] /= 60
    lg = g.groupby("season")[["minutes", "gf"]].sum()
    league_rate = (lg["gf"] / lg["minutes"]).rename("league_rate")
    g = g.join(league_rate, on="season")
    g["off_factor"] = ((g["gf"] + g["league_rate"] * prior_minutes) / (g["minutes"] + prior_minutes)) / g["league_rate"]
    g["def_factor"] = ((g["ga"] + g["league_rate"] * prior_minutes) / (g["minutes"] + prior_minutes)) / g["league_rate"]
    g["gf60_5v5"] = 60 * g["gf"] / g["minutes"]
    g["ga60_5v5"] = 60 * g["ga"] / g["minutes"]
    return g.reset_index()


def calibration(tv_test: pd.DataFrame, stints_test: pd.DataFrame, rates, opp_pull) -> pd.DataFrame:
    """At the moment each team actually pulled (even strength), predict the
    chance of reaching overtime or winning; compare with what happened."""
    sol = solve(rates, k_max=0, opp_pull=opp_pull, our_policy={1: 1200, 2: 1200, 3: 1200, 4: 1200}, terminal="comeback")
    p = stints_test[stints_test["pull_time"].notna() & stints_test["deficit"].isin([1, 2]) & (stints_test["pull_mp"] == 0)].copy()
    p["pred"] = [sol.value(int(round(t)), -d, 0) for t, d in zip(p["pull_time"], p["deficit"])]

    # Did the team end regulation tied or ahead?
    end = tv_test[tv_test["period"] == 3].groupby(["game_id", "side"]).tail(1)
    end = end.assign(final_diff=end["diff"] + end["gf"].astype(int) - end["ga"].astype(int))
    p = p.merge(end[["game_id", "side", "final_diff"]], on=["game_id", "side"], how="left")
    p["came_back"] = (p["final_diff"] >= 0).astype(int)
    p["bin"] = pd.qcut(p["pred"], 8, duplicates="drop")
    cal = p.groupby("bin", observed=True).agg(predicted=("pred", "mean"), actual=("came_back", "mean"), n=("pred", "size"))
    cal = cal.reset_index(drop=True)
    brier = float(((p["pred"] - p["came_back"]) ** 2).mean())
    base = float(((p["came_back"].mean() - p["came_back"]) ** 2).mean())
    cal.attrs.update(brier=brier, brier_baseline=base, n=len(p))
    return cal


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    games_raw = []
    for f in sorted((ROOT / "data" / "raw").glob("*.jsonl.gz")):
        games_raw += load_season(f)
    games = game_table(games_raw)
    print(f"{len(games)} regular-season games")

    iv = build_intervals(games_raw)
    tv = add_team_ids(team_view(iv), games)
    stints = find_pulls(tv)
    counts = late_counts(tv)

    # ---- model on all seasons ------------------------------------------------
    rates = rates_from_counts(counts)
    opp_pull = observed_pull_medians(stints)
    opt = solve(rates, opp_pull=opp_pull)
    t_star = {d: opt.pull_time(d) for d in (1, 2)}
    print("optimal pull (EV):", t_star, "league medians:", opp_pull)

    # ---- bootstrap over games --------------------------------------------------
    gids = counts["game_id"].unique()
    boot = []
    for b in range(N_BOOT):
        draw = pd.Series(RNG.choice(gids, size=len(gids), replace=True)).value_counts()
        rb = rates_from_counts(counts, weights=draw)
        sb = solve(rb, k_max=0, opp_pull=opp_pull)
        boot.append({"draw": b, "down1": sb.pull_time(1), "down2": sb.pull_time(2)})
    boot = pd.DataFrame(boot)
    boot.to_csv(OUT / "bootstrap.csv", index=False)
    ci = {d: [float(boot[f"down{d}"].quantile(q)) for q in (0.025, 0.975)] for d in (1, 2)}
    print("95% CI:", ci)

    # ---- audit -------------------------------------------------------------------
    pvs = {d: policy_values(rates, d, opp_pull) for d in (1, 2)}
    a = audit(stints, opt, pvs)
    a = a.merge(games[["game_id", "date"]], on="game_id", how="left")
    a.to_csv(OUT / "audit.csv", index=False)
    lb = team_leaderboard(a, games)
    lb.to_csv(OUT / "leaderboard.csv", index=False)

    obs = a[a["status"] != "censored (imputed)"]
    trend = obs.groupby(["season", "deficit"]).agg(
        median_actual=("actual_time", "median"),
        pct_late=("status", lambda s: float((s.isin(["late", "never pulled"])).mean())),
        decisions=("status", "size"),
    ).reset_index()
    trend["optimal"] = trend["deficit"].map(t_star)
    trend.to_csv(OUT / "league_trend.csv", index=False)

    # ---- team strength -> team-specific optimal times -------------------------
    ts = team_strength(tv)
    tst = []
    for r in ts.itertuples():
        s = solve(rates, k_max=0, opp_pull=opp_pull, our_off=r.off_factor, our_def=r.def_factor)
        tst.append((s.pull_time(1), s.pull_time(2)))
    ts["opt_down1"], ts["opt_down2"] = zip(*tst)
    ts.to_csv(OUT / "team_strength.csv", index=False)

    # ---- out-of-sample calibration ---------------------------------------------
    train_ids = games.loc[games["season"].isin(TRAIN_SEASONS), "game_id"]
    rates_train = rates_from_counts(counts[counts["game_id"].isin(train_ids)])
    test_mask = ~tv["season"].isin(TRAIN_SEASONS)
    cal = calibration(tv[test_mask], stints[~stints["season"].isin(TRAIN_SEASONS)], rates_train, opp_pull)
    cal.to_csv(OUT / "calibration.csv", index=False)
    print("calibration:", cal.attrs)

    # ---- replay data (third period only, audited games) -------------------------
    keep = iv["game_id"].isin(a["game_id"].unique()) & (iv["period"] == 3)
    iv.loc[keep].drop(columns=["home_delayed", "away_delayed"]).to_csv(OUT / "replay.csv.gz", index=False)
    games.to_csv(OUT / "games.csv", index=False)
    # Headline curves (down 1 from 7:00, down 2 from 12:00) so the app opens instantly.
    curves = pd.concat([
        threshold_curve(rates, d, t_eval, opp_pull).assign(deficit=d, t_eval=t_eval) for d, t_eval in VERDICT_CURVES
    ])
    curves.to_csv(OUT / "curves.csv", index=False)
    rates.as_frame().to_csv(OUT / "rates.csv", index=False)

    # ---- headline summary --------------------------------------------------------
    cur = games["season"].max()
    complete = lb[lb["season"] != cur]
    summary = {
        "games": int(len(games)),
        "seasons": sorted(int(s) for s in games["season"].unique()),
        "current_season": int(cur),
        "optimal_pull_seconds": {str(d): t_star[d] for d in (1, 2)},
        "optimal_pull_ci95": {str(d): ci[d] for d in (1, 2)},
        "optimal_pull_pp": {str(d): opt.pull_time(d, 120) for d in (1, 2)},
        "optimal_pull_sh": {str(d): opt.pull_time(d, -120) for d in (1, 2)},
        "league_median_pull_seconds": {str(d): opp_pull[d] for d in (1, 2)},
        "decisions": int(len(a)),
        "share_late": float(a["status"].isin(["late", "never pulled", "censored (imputed)"]).mean()),
        "points_lost_per_season_league": float(a[a["season"] != cur].groupby("season")["points_lost"].sum().mean()),
        "points_lost_per_team_season": float(complete["points_lost"].mean()),
        "worst_team_season": complete.sort_values("points_lost", ascending=False).head(1)[["season", "team", "points_lost"]].to_dict("records")[0],
        "calibration": cal.attrs,
        "rates": {k: {str(m): float(v) for m, v in getattr(rates, k).items()} for k in ("trail_in_for", "trail_in_against", "trail_out_for", "trail_out_against", "tied_for")},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("optimal_pull_seconds", "optimal_pull_ci95", "league_median_pull_seconds", "share_late", "points_lost_per_season_league", "points_lost_per_team_season", "worst_team_season")}, indent=1, default=float))


if __name__ == "__main__":
    main()
