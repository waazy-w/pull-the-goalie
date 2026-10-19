"""Download play-by-play for the analysed seasons.

    python scripts/fetch_data.py            # 2023-24 .. 2026-27
    python scripts/fetch_data.py 2026       # just the current season (refresh)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pullgoalie.fetch import fetch_season  # noqa: E402

SEASONS = [2023, 2024, 2025, 2026]

if __name__ == "__main__":
    seasons = [int(a) for a in sys.argv[1:]] or SEASONS
    for s in seasons:
        fetch_season(s, ROOT / "data" / "raw")
