#!/usr/bin/env bash
# One-time setup: Python packages, AutoDock Vina binary, and the 3ERT structure.
set -e
cd "$(dirname "$0")/.."
pip install -q -r requirements.txt
mkdir -p bin data/raw
if [ ! -x bin/vina ]; then
  wget -q -O bin/vina https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64
  chmod +x bin/vina
fi
if [ ! -f data/raw/3ERT.pdb ]; then
  wget -q -O data/raw/3ERT.pdb https://files.rcsb.org/download/3ERT.pdb
fi
echo "Setup finished."
