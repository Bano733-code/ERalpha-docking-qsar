#!/usr/bin/env bash
# Runs the complete pipeline. Use  --csv  if ChEMBL data was downloaded manually.
set -e
cd "$(dirname "$0")/.."
export PATH="$PWD/bin:$PATH"
SRC="--source api"
[ "$1" = "--csv" ] && SRC="--source csv"
python scripts/01_prepare_protein.py
python scripts/02_redock.py
python scripts/03_chembl_data.py $SRC
python scripts/04_qsar.py
python scripts/05_screen_dock.py --n 100 --workers 4
python scripts/06_interactions.py
python scripts/07_make_report.py
