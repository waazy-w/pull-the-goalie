"""Assemble the static GitHub Pages site (the Streamlit app running in the
browser through stlite / Pyodide).

    python scripts/build_site.py _site
    python -m http.server -d _site 8000   # preview locally
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APP_FILES = [
    "app.py",
    "src/pullgoalie/__init__.py",
    "src/pullgoalie/model.py",
]
DATA_SUFFIXES = (".csv", ".json", ".csv.gz")

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "_site").resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    data = sorted(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "data" / "processed").iterdir()
        if p.name.endswith(DATA_SUFFIXES) and not p.name.startswith("_")
    )
    files = APP_FILES + data
    for f in files:
        dest = out / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    shutil.copy2(ROOT / "site" / "index.html", out / "index.html")
    (out / "files.json").write_text(json.dumps(files, indent=1))
    (out / ".nojekyll").write_text("")
    print(f"site: {len(files)} files")
