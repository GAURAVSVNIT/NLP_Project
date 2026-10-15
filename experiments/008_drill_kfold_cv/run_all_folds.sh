#!/bin/sh
# Runs each (config, fold) in its own Python process, then pools the scores.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
for cfg in mixdv; do
  for k in 0 1 2 3 4; do
    $PY -X utf8 experiments/008_drill_kfold_cv/run_drill_kfold_cv.py "$cfg" "$k" || echo "FAILED $cfg fold $k rc=$?"
  done
done
$PY -X utf8 experiments/008_drill_kfold_cv/run_drill_kfold_cv.py
