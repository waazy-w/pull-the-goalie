import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pullgoalie.audit import add_team_ids, find_pulls  # noqa: E402
from pullgoalie.model import D_MAX, Rates, solve  # noqa: E402
from pullgoalie.parse import build_intervals, game_table, team_view  # noqa: E402

HOME, AWAY = 1, 2


def play(period, elapsed, sit, typ, team=None, sort=0, **kw):
    return {"period": period, "period_type": "REG", "t_elapsed": elapsed, "t_remaining": 1200 - elapsed,
            "situation": sit, "type": typ, "sort": sort, "team_id": team, "home_score": kw.get("hs"),
            "away_score": kw.get("as_"), "pen_type": None, "pen_minutes": None}


def game(plays):
    for i, p in enumerate(plays):
        p["sort"] = i
    return {"game_id": 2025020001, "season": 20252026, "game_type": 2, "date": "2025-10-07", "state": "OFF",
            "home_id": HOME, "home": "HOM", "away_id": AWAY, "away": "AWY", "home_final": 1, "away_final": 0,
            "last_period_type": "REG", "plays": plays}


def sample_game():
    return game([
        play(3, 0, "1551", "period-start"),
        play(3, 600, "1551", "goal", HOME, hs=1, as_=0),
        # delayed penalty on HOME: away net empties briefly, must NOT count
        play(3, 700, "1551", "delayed-penalty", HOME),
        play(3, 710, "0651", "shot-on-goal", AWAY),
        play(3, 712, "0651", "penalty", HOME),
        play(3, 712, "1541", "faceoff", AWAY),
        play(3, 832, "1551", "faceoff", AWAY),
        # strategic pull: away down 1 with ~2:30 left, stays out to the end
        play(3, 1050, "1551", "hit", AWAY),
        play(3, 1060, "0651", "shot-on-goal", AWAY),
        play(3, 1200, "0651", "period-end"),
    ])


def test_intervals_cover_period_and_flag_only_strategic_pull():
    g = sample_game()
    iv = build_intervals([g])
    assert iv["dur"].sum() == 1200
    # the delayed-penalty empty net is not a strategic pull
    assert not iv.loc[(iv["t1"] >= 488) & (iv["t1"] <= 490), "away_pull"].any()
    assert iv.loc[iv["t1"] == 0, "away_pull"].all()


def test_find_pull_time_is_midpoint_of_first_empty_interval():
    g = sample_game()
    tv = add_team_ids(team_view(build_intervals([g])), game_table([g]))
    st = find_pulls(tv)
    away = st[(st["team"] == "AWY") & (st["deficit"] == 1)].iloc[0]
    assert away["pull_time"] == (150 + 140) / 2


def flat_rates(in_f=2.3, in_a=2.2, out_f=5.6, out_a=18.0):
    same = lambda v: {-1: v, 0: v, 1: v}  # noqa: E731
    return Rates(same(in_f), same(in_a), same(out_f), same(out_a), same(2.0))


def test_dp_terminal_values_and_monotonicity():
    s = solve(flat_rates(), k_max=0)
    assert s.value(0, 1) == 2.0 and s.value(0, 0) == 1.5 and s.value(0, -1) == 0.0
    v = [s.value(t, -1) for t in (30, 120, 300, 600)]
    assert all(np.diff(v) > 0)  # more time left -> better chance when trailing


def test_pull_earlier_when_down_two_and_never_if_pulling_is_useless():
    s = solve(flat_rates(), k_max=0)
    assert 0 < s.pull_time(1) < s.pull_time(2)
    useless = solve(flat_rates(out_f=2.3, out_a=18.0), k_max=0)
    assert useless.pull_time(1) == 0


def test_better_offense_pulls_earlier():
    base = solve(flat_rates(), k_max=0).pull_time(1)
    strong = solve(flat_rates(), k_max=0, our_off=1.3).pull_time(1)
    assert strong > base


def test_fixed_policy_never_beats_optimal():
    r = flat_rates()
    opt = solve(r, k_max=0)
    for T in (0, 60, 120, 400):
        pol = solve(r, k_max=0, our_policy={1: T})
        assert pol.value(300, -1) <= opt.value(300, -1) + 1e-12
    assert np.isclose(solve(r, k_max=0, our_policy={1: opt.pull_time(1)}).value(600, -1), opt.value(600, -1), atol=1e-6)
    assert D_MAX == 4
