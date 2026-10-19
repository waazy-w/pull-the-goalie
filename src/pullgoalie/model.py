"""Scoring rates and the pull-the-goalie dynamic program.

Model
-----
Late in the third period goals arrive as Poisson processes whose rates depend
on the on-ice state: whether the trailing team has pulled its goalie, and the
manpower situation (even strength / power play / shorthanded).

We solve a finite-horizon Markov decision process on a one-second grid:

    state  = (seconds left t, score diff d from our view, power-play clock k)
    action = keep goalie in / pull (only meaningful while trailing)
    reward = NHL standings points at the end of regulation:
             win 2, tie -> overtime worth 1 + P(win OT/SO), loss 0

The value V(t, d, k) is the expected standings points from that state, and
the optimal policy tells you the earliest moment pulling beats waiting. When
we lead, the opponent pulls according to the league's observed habits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WINDOW = 600  # seconds of the 3rd period used to estimate late-game rates
PRIOR_MINUTES = 30.0  # pseudo-minutes for shrinking sparse cells
D_MAX = 4  # score differences beyond +/-4 are absorbed
PERIOD = 1200


@dataclass
class Rates:
    """Goals per 60 minutes. Keys are manpower edge mp in {-1, 0, 1}."""

    trail_in_for: dict
    trail_in_against: dict
    trail_out_for: dict
    trail_out_against: dict
    tied_for: dict  # rate for one team when tied (by its own mp)
    minutes: dict = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        rows = []
        for mp, label in ((0, "Even strength"), (1, "Power play"), (-1, "Shorthanded")):
            rows.append(
                {
                    "situation": label,
                    "goalie_in_GF60": self.trail_in_for[mp],
                    "goalie_in_GA60": self.trail_in_against[mp],
                    "pulled_GF60": self.trail_out_for[mp],
                    "pulled_GA60": self.trail_out_against[mp],
                    "minutes_in": self.minutes.get(("in", mp), np.nan),
                    "minutes_pulled": self.minutes.get(("out", mp), np.nan),
                }
            )
        return pd.DataFrame(rows)


def _late_window(tv: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Third-period intervals clipped to the last `window` seconds."""
    v = tv[(tv["period"] == 3) & (tv["t1"] < window)].copy()
    v["dur"] = np.minimum(v["t0"], window) - v["t1"]
    # Drop intervals with an empty net that is not a strategic pull (delayed
    # penalties) and odd strengths (3-on-3, penalty shots).
    odd = (v["net_empty"] & ~v["pulled"]) | (v["opp_net_empty"] & ~v["opp_pulled"])
    odd |= (v["sk"] < 3) | (v["opp_sk"] < 3)
    return v[~odd & (v["dur"] >= 0)]


def late_counts(tv: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Per-game exposure and goals by state; the unit we resample for bootstraps."""
    v = _late_window(tv, window)
    trail = (v["diff"] < 0) & (v["diff"] >= -3) & ~v["opp_pulled"]
    tied = (v["diff"] == 0) & ~v["pulled"] & ~v["opp_pulled"]
    v = v[trail | tied].copy()
    v["kind"] = np.where(v["diff"] == 0, "tie", np.where(v["pulled"], "out", "in"))
    g = v.groupby(["game_id", "kind", "mp"]).agg(minutes=("dur", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
    g["minutes"] = g["minutes"] / 60
    return g.reset_index()


def estimate_rates(tv: pd.DataFrame, window: int = WINDOW) -> Rates:
    return rates_from_counts(late_counts(tv, window))


def rates_from_counts(counts: pd.DataFrame, weights: pd.Series | None = None) -> Rates:
    """Rates from per-game counts; `weights` (by game_id) supports bootstrapping."""
    c = counts
    if weights is not None:
        w = c["game_id"].map(weights).fillna(0).to_numpy()
        c = c.assign(minutes=c["minutes"] * w, gf=c["gf"] * w, ga=c["ga"] * w)
    tot = c.groupby(["kind", "mp"])[["minutes", "gf", "ga"]].sum()

    def agg(kind):
        return tot.loc[kind].reindex([-1, 0, 1]).fillna(0) if kind in tot.index.get_level_values(0) else (
            pd.DataFrame(0.0, index=[-1, 0, 1], columns=["minutes", "gf", "ga"])
        )

    t_in, t_out, t_tie = agg("in"), agg("out"), agg("tie")

    def rate(goals, minutes, prior_rate):
        return 60 * (goals + prior_rate / 60 * PRIOR_MINUTES) / (minutes + PRIOR_MINUTES)

    def raw(goals, minutes):
        return 60 * goals / max(minutes, 1e-9)

    # Even-strength rates have plenty of data; use them to build priors for
    # the sparse power-play / shorthanded cells.
    ev_in_f, ev_in_a = raw(t_in.loc[0, "gf"], t_in.loc[0, "minutes"]), raw(t_in.loc[0, "ga"], t_in.loc[0, "minutes"])
    ev_out_f, ev_out_a = raw(t_out.loc[0, "gf"], t_out.loc[0, "minutes"]), raw(t_out.loc[0, "ga"], t_out.loc[0, "minutes"])

    r = {k: {} for k in ("in_f", "in_a", "out_f", "out_a", "tie")}
    for mp in (-1, 0, 1):
        r["in_f"][mp] = rate(t_in.loc[mp, "gf"], t_in.loc[mp, "minutes"], ev_in_f)
        r["in_a"][mp] = rate(t_in.loc[mp, "ga"], t_in.loc[mp, "minutes"], ev_in_a)
        r["tie"][mp] = rate(t_tie.loc[mp, "gf"], t_tie.loc[mp, "minutes"], ev_in_f)
    for mp in (-1, 0, 1):
        # Pulled cells: prior scales the even-strength pulled rate by how much
        # this manpower state changes goalie-in scoring.
        f_scale = r["in_f"][mp] / r["in_f"][0]
        a_scale = r["in_a"][mp] / r["in_a"][0]
        r["out_f"][mp] = rate(t_out.loc[mp, "gf"], t_out.loc[mp, "minutes"], ev_out_f * f_scale)
        r["out_a"][mp] = rate(t_out.loc[mp, "ga"], t_out.loc[mp, "minutes"], ev_out_a * a_scale)

    minutes = {("in", mp): t_in.loc[mp, "minutes"] for mp in (-1, 0, 1)}
    minutes.update({("out", mp): t_out.loc[mp, "minutes"] for mp in (-1, 0, 1)})
    return Rates(r["in_f"], r["in_a"], r["out_f"], r["out_a"], r["tie"], minutes)


def threshold_curve(rates: "Rates", deficit: int, t_eval: int, opp_pull: dict, our_off: float = 1.0,
                    our_def: float = 1.0, step: int = 10) -> pd.DataFrame:
    """From (t_eval left, down `deficit`, even strength): expected points and
    comeback chance if you pull with T seconds left, for T in [0, t_eval]."""
    rows = []
    for T in range(0, t_eval + 1, step):
        kw = dict(horizon=t_eval, k_max=0, our_off=our_off, our_def=our_def, opp_pull=opp_pull, our_policy={deficit: T})
        rows.append({
            "pull_at": T,
            "points": solve(rates, **kw).value(t_eval, -deficit),
            "comeback": solve(rates, terminal="comeback", **kw).value(t_eval, -deficit),
        })
    return pd.DataFrame(rows)


@dataclass
class Solution:
    V: np.ndarray  # [t, d_index, k_index] expected points
    pull: np.ndarray  # bool, optimal (or given) action while trailing
    Q_in: np.ndarray
    Q_out: np.ndarray
    k_max: int

    def value(self, t: int, d: int, k: int = 0) -> float:
        d = int(np.clip(d, -D_MAX, D_MAX))
        return float(self.V[t, d + D_MAX, k + self.k_max])

    def pull_time(self, deficit: int, k: int = 0) -> int:
        """Seconds left at which pulling becomes optimal (and stays optimal)."""
        col = self.pull[:, D_MAX - deficit, k + self.k_max]
        idx = np.flatnonzero(col[1:])
        return int(idx.max() + 1) if len(idx) else 0


def solve(
    rates: Rates,
    horizon: int = PERIOD,
    k_max: int = 120,
    ot_win: float = 0.5,
    our_off: float = 1.0,
    our_def: float = 1.0,
    opp_pull: dict | None = None,
    our_policy: dict | None = None,
    terminal: str = "points",
) -> Solution:
    """Backward induction.

    our_off / our_def multiply our scoring / conceding rates (team strength).
    opp_pull: {deficit: seconds} the opponent pulls at when trailing.
    our_policy: {deficit: seconds} to *evaluate* a fixed pull time instead of
        optimizing (deficits missing from the dict are optimized).
    terminal: "points" (standings points) or "comeback" (P(reach OT or win)).
    """
    opp_pull = opp_pull or {1: 75, 2: 150, 3: 200, 4: 200}
    nd = 2 * D_MAX + 1
    ks = np.arange(-k_max, k_max + 1)
    nk = len(ks)
    mp = np.sign(ks)  # our manpower edge

    def arr(table, sign=1):
        return np.array([table[int(m)] for m in sign * mp]) / 3600.0

    # Per-second goal probabilities as vectors over k.
    in_f, in_a = arr(rates.trail_in_for) * our_off, arr(rates.trail_in_against) * our_def
    out_f, out_a = arr(rates.trail_out_for) * our_off, arr(rates.trail_out_against) * our_def
    tie_f, tie_a = arr(rates.tied_for) * our_off, arr(rates.tied_for, -1) * our_def
    # When leading: opponent is the trailing team with manpower -mp.
    lead_opp_in_f = arr(rates.trail_in_against, -1) * our_off  # our goals
    lead_opp_in_a = arr(rates.trail_in_for, -1) * our_def  # their goals
    lead_opp_out_f = arr(rates.trail_out_against, -1) * our_off
    lead_opp_out_a = arr(rates.trail_out_for, -1) * our_def

    V = np.zeros((horizon + 1, nd, nk))
    pull = np.zeros((horizon + 1, nd, nk), dtype=bool)
    Q_in = np.full((horizon + 1, nd, nk), np.nan)
    Q_out = np.full((horizon + 1, nd, nk), np.nan)

    ds = np.arange(-D_MAX, D_MAX + 1)
    if terminal == "points":
        V[0] = np.where(ds > 0, 2.0, np.where(ds == 0, 1.0 + ot_win, 0.0))[:, None]
    else:
        V[0] = (ds >= 0).astype(float)[:, None]

    # Index helpers for the power-play clock after one second.
    k_next = np.clip(ks - np.sign(ks), -k_max, k_max) + k_max  # ticks toward 0
    k_zero = np.full(nk, k_max)  # PP ends (our PP + we score, or theirs + they score)
    k_after_for = np.where(ks > 0, k_zero, k_next)  # we score: our PP ends
    k_after_against = np.where(ks < 0, k_zero, k_next)  # they score: their PP ends

    for t in range(1, horizon + 1):
        Vn = V[t - 1]
        for di, d in enumerate(ds):
            up = min(di + 1, nd - 1)
            dn = max(di - 1, 0)
            stay = Vn[di, k_next]
            v_for = Vn[up, k_after_for]
            v_against = Vn[dn, k_after_against]

            def q(pf, pa):
                return pf * v_for + pa * v_against + (1 - pf - pa) * stay

            if d < 0:
                qi, qo = q(in_f, in_a), q(out_f, out_a)
                Q_in[t, di], Q_out[t, di] = qi, qo
                if our_policy is not None and -d in our_policy:
                    act = np.full(nk, t <= our_policy[-d])
                else:
                    act = qo > qi
                pull[t, di] = act
                V[t, di] = np.where(act, qo, qi)
            elif d == 0:
                V[t, di] = q(tie_f, tie_a)
            else:
                if t <= opp_pull.get(min(d, 4), 0):
                    V[t, di] = q(lead_opp_out_f, lead_opp_out_a)
                else:
                    V[t, di] = q(lead_opp_in_f, lead_opp_in_a)
    return Solution(V, pull, Q_in, Q_out, k_max)
