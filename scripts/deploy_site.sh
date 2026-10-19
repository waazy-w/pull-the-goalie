#!/usr/bin/env bash
# Build the in-browser site and publish it to the gh-pages branch (GitHub Pages).
#   bash scripts/deploy_site.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
[ -x .venv/Scripts/python ] && PY=.venv/Scripts/python
[ -x .venv/bin/python ] && PY=.venv/bin/python
$PY scripts/build_site.py _site
rm -rf _site/.git
git -C _site init -q -b gh-pages
git -C _site add -A
git -C _site commit -q -m "Deploy site from $(git rev-parse --short HEAD)"
git -C _site push -q -f "$(git remote get-url origin)" gh-pages
echo "Deployed. Pages: https://$(git remote get-url origin | sed -E 's#https://github.com/([^/]+)/([^/.]+).*#\1.github.io/\2#')/"
