"""Download NHL regular-season play-by-play from the public NHL API.

Each game is flattened to a compact table of events (one row per play) and
cached as gzipped JSON lines under data/raw/, so re-running only fetches games
that are missing or were not final yet.
"""

from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
FINAL_STATES = {"OFF", "FINAL"}
MAX_GAMES_PER_SEASON = 1400  # 32 teams x 82 games / 2 = 1312, with headroom


def _mmss_to_seconds(text: str) -> int:
    minutes, seconds = text.split(":")
    return int(minutes) * 60 + int(seconds)


def flatten_game(raw: dict) -> dict:
    """Keep only the fields the model needs."""
    plays = []
    for p in raw.get("plays", []):
        det = p.get("details", {})
        pd_ = p["periodDescriptor"]
        plays.append(
            {
                "period": pd_["number"],
                "period_type": pd_.get("periodType", "REG"),
                "t_elapsed": _mmss_to_seconds(p["timeInPeriod"]),
                "t_remaining": _mmss_to_seconds(p["timeRemaining"]),
                "situation": p.get("situationCode", ""),
                "type": p["typeDescKey"],
                "sort": p.get("sortOrder", 0),
                "team_id": det.get("eventOwnerTeamId"),
                "home_score": det.get("homeScore"),
                "away_score": det.get("awayScore"),
                "pen_type": det.get("typeCode") if p["typeDescKey"] == "penalty" else None,
                "pen_minutes": det.get("duration") if p["typeDescKey"] == "penalty" else None,
            }
        )
    return {
        "game_id": raw["id"],
        "season": raw["season"],
        "game_type": raw["gameType"],
        "date": raw["gameDate"],
        "state": raw["gameState"],
        "home_id": raw["homeTeam"]["id"],
        "home": raw["homeTeam"]["abbrev"],
        "away_id": raw["awayTeam"]["id"],
        "away": raw["awayTeam"]["abbrev"],
        "home_final": raw["homeTeam"].get("score"),
        "away_final": raw["awayTeam"].get("score"),
        "last_period_type": (raw.get("gameOutcome") or {}).get("lastPeriodType"),
        "plays": plays,
    }


def _fetch_one(session: requests.Session, game_id: int) -> dict | None:
    for attempt in range(4):
        try:
            r = session.get(API.format(game_id=game_id), timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return flatten_game(r.json())
        except (requests.RequestException, ValueError):
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {game_id}")


def fetch_season(season_start_year: int, raw_dir: Path, workers: int = 8) -> Path:
    """Fetch every final regular-season game of a season into one .jsonl.gz file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{season_start_year}{season_start_year + 1}.jsonl.gz"

    cached: dict[int, dict] = {}
    if out.exists():
        with gzip.open(out, "rt", encoding="utf-8") as f:
            for line in f:
                g = json.loads(line)
                if g["state"] in FINAL_STATES:
                    cached[g["game_id"]] = g

    ids = [int(f"{season_start_year}02{n:04d}") for n in range(1, MAX_GAMES_PER_SEASON + 1)]
    todo = [i for i in ids if i not in cached]
    session = requests.Session()
    session.headers["User-Agent"] = "pull-the-goalie research (open source)"

    misses_in_a_row = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit in chunks so we can stop once the season's id range runs out.
        for start in range(0, len(todo), workers * 4):
            chunk = todo[start : start + workers * 4]
            futures = {pool.submit(_fetch_one, session, gid): gid for gid in chunk}
            for fut in as_completed(futures):
                g = fut.result()
                if g is None:
                    misses_in_a_row += 1
                    continue
                misses_in_a_row = 0
                if g["state"] in FINAL_STATES:
                    cached[g["game_id"]] = g
            if misses_in_a_row >= len(chunk):
                break
            # Current season: games not played yet still exist but are not final.
            if all(futures[f] not in cached for f in futures):
                break

    with gzip.open(out, "wt", encoding="utf-8") as f:
        for gid in sorted(cached):
            f.write(json.dumps(cached[gid]) + "\n")
    print(f"{season_start_year}-{season_start_year + 1}: {len(cached)} final games -> {out.name}")
    return out


def load_season(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
