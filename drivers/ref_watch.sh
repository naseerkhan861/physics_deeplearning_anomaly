#!/bin/bash
# Scores newly completed stronger-recipe fits and, when all 36 family
# records exist, derives the frozen contrasts. Polls every 10 minutes.
cd /home/edison/agents_project/pgdn_project
V=/home/edison/agents_project/scienctific/.venv/bin/python
export OMP_NUM_THREADS=4
while true; do
  $V ref_family_score.py >> results/logs/ref_watch.log 2>&1
  $V ref_protocol_analyze.py >> results/logs/ref_watch.log 2>&1
  if [ -f results/raw/ref_analysis.json ] && \
     grep -q '"status": "complete"' results/raw/ref_analysis.json; then \
     echo DONE >> results/logs/ref_watch.log; break; fi
  sleep 600
done
