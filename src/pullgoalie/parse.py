"""Turn flattened play-by-play into timed game-state intervals.

An interval runs from one play to the next inside a regulation period. It takes
the on-ice situation of the play that closes it (the NHL feed only reveals a
pulled goalie at the first play *after* the pull), and the score from before
that play (so a goal is attributed to the state it was scored in).

The API's ``situationCode`` is four digits:
    [away goalie in net][away skaters][home skaters][home goalie in net]
e.g. "0651" = away net empty, 6 away skaters vs 5 home skaters.

Two reasons a net is empty:
* a *strategic* pull (trailing late, extra attacker), which is what we study;
* a *delayed penalty* (the referee's arm is up; play stops as soon as the
  offending team touches the puck). Those are flagged and excluded.
"""

from __future__ import annotations

import pandas as pd

REG_PERIODS = (1, 2, 3)
PERIOD_SECONDS = 1200


def _parse_situation(code: str):
    if not code or len(code) != 4 or not code.isdigit():
        return None
    return int(code[0]), int(code[1]), int(code[2]), int(code[3])


def game_intervals(game: dict) -> pd.DataFrame:
    """One row per inter-play interval in regulation for a single game."""
    home_id, away_id = game["home_id"], game["away_id"]
    plays = sorted(game["plays"], key=lambda p: p["sort"])

    rows = []
    home_score = away_score = 0
    # delayed[team_id] is True while that team is on an extra attacker because
    # the *other* team has a delayed penalty coming.
    delayed = {home_id: False, away_id: False}
    prev = None
    for p in plays:
        if p["period"] not in REG_PERIODS or p["period_type"] != "REG":
            prev = None
            continue
        sit = _parse_situation(p["situation"])
        if prev is not None and prev["period"] == p["period"] and sit is not None:
            t0, t1 = prev["t_remaining"], p["t_remaining"]
            dur = t0 - t1
            if dur >= 0:
                ag, ask, hsk, hg = sit
                goal_by = None
                if p["type"] == "goal":
                    goal_by = "home" if p["team_id"] == home_id else "away"
                rows.append(
                    {
                        "game_id": game["game_id"],
                        "season": game["season"],
                        "period": p["period"],
                        "t0": t0,  # seconds left in the period at the start
                        "t1": t1,
                        "dur": dur,
                        "home_score": home_score,
                        "away_score": away_score,
                        "away_g": ag,
                        "away_sk": ask,
                        "home_sk": hsk,
                        "home_g": hg,
                        "goal_by": goal_by,
                        "home_delayed": delayed[home_id],
                        "away_delayed": delayed[away_id],
                        "end_type": p["type"],
                    }
                )

        # Update running state after the play.
        t = p["type"]
        if t == "goal":
            if p.get("home_score") is not None:
                home_score, away_score = p["home_score"], p["away_score"]
            elif p["team_id"] == home_id:
                home_score += 1
            else:
                away_score += 1
            delayed = {home_id: False, away_id: False}
        elif t == "delayed-penalty" and p["team_id"] in delayed:
            other = away_id if p["team_id"] == home_id else home_id
            delayed[other] = True
        elif t == "penalty" and p["team_id"] in delayed:
            other = away_id if p["team_id"] == home_id else home_id
            delayed[other] = False
        elif t in ("stoppage", "period-end"):
            delayed = {home_id: False, away_id: False}
        prev = p
    return pd.DataFrame(rows)


def _flag_strategic(df: pd.DataFrame) -> pd.DataFrame:
    """Mark which empty-net intervals are strategic pulls.

    Besides the explicit delayed-penalty flag, older feeds sometimes miss the
    ``delayed-penalty`` event, so a short empty-net run (<= 60 s) that ends in
    a penalty against the opponent is also treated as a delayed penalty.
    """
    df = df.copy()
    for side, opp in (("home", "away"), ("away", "home")):
        out = df[f"{side}_g"].eq(0)
        # Run ids: a new run starts whenever "out" switches on, or a new game/period.
        new_block = (out != out.shift()) | (df["game_id"] != df["game_id"].shift()) | (
            df["period"] != df["period"].shift()
        )
        run = new_block.cumsum()
        g = df[out].groupby(run[out])
        run_dur = g["dur"].sum()
        run_delayed_flag = g[f"{side}_delayed"].any()
        # Heuristic: run ended by a penalty play while short.
        last_idx = pd.Series(df.index[out], index=df.index[out]).groupby(run[out]).last()
        next_idx = last_idx + 1
        valid = next_idx.isin(df.index)
        ends_in_pen = pd.Series(False, index=last_idx.index)
        ends_in_pen[valid] = df.loc[next_idx[valid], "end_type"].eq("penalty").to_numpy() | df.loc[
            last_idx[valid], "end_type"
        ].eq("penalty").to_numpy()
        delayed_run = run_delayed_flag | (ends_in_pen & (run_dur <= 60))
        df[f"{side}_pull"] = False
        strategic_runs = delayed_run[~delayed_run].index
        df.loc[out & run.isin(strategic_runs), f"{side}_pull"] = True
    return df


def build_intervals(games: list[dict]) -> pd.DataFrame:
    parts = [game_intervals(g) for g in games if g["game_type"] == 2]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return _flag_strategic(df)


def game_table(games: list[dict]) -> pd.DataFrame:
    """One row per game with final result (needed for standings points)."""
    rows = []
    for g in games:
        if g["game_type"] != 2:
            continue
        rows.append(
            {
                "game_id": g["game_id"],
                "season": g["season"],
                "date": g["date"],
                "home_id": g["home_id"],
                "home": g["home"],
                "away_id": g["away_id"],
                "away": g["away"],
                "home_final": g["home_final"],
                "away_final": g["away_final"],
                "last_period_type": g["last_period_type"],
            }
        )
    return pd.DataFrame(rows)


def team_view(iv: pd.DataFrame) -> pd.DataFrame:
    """Re-express intervals from each team's point of view (2 rows per interval).

    Columns: diff (team score - opponent), pulled (team's net strategically
    empty), opp_pulled, mp (manpower edge excluding the extra attacker:
    +1 power play, -1 shorthanded), gf / ga (goal scored at the end).
    """
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        v = pd.DataFrame(
            {
                "game_id": iv["game_id"],
                "season": iv["season"],
                "period": iv["period"],
                "t0": iv["t0"],
                "t1": iv["t1"],
                "dur": iv["dur"],
                "side": side,
                "diff": iv[f"{side}_score"] - iv[f"{opp}_score"],
                "net_empty": iv[f"{side}_g"].eq(0),
                "pulled": iv[f"{side}_pull"],
                "opp_net_empty": iv[f"{opp}_g"].eq(0),
                "opp_pulled": iv[f"{opp}_pull"],
                "sk": iv[f"{side}_sk"] - iv[f"{side}_g"].eq(0).astype(int),
                "opp_sk": iv[f"{opp}_sk"] - iv[f"{opp}_g"].eq(0).astype(int),
                "gf": iv["goal_by"].eq(side),
                "ga": iv["goal_by"].eq(opp),
            }
        )
        v["mp"] = (v["sk"] - v["opp_sk"]).clip(-1, 1)
        out.append(v)
    return pd.concat(out, ignore_index=True)
