#!/bin/bash
# P0-1 stronger-recipe mechanism replication (round-18 review).
# Frozen reference-style recipe (validated in the audit): lr 1e-4,
# MultiStepLR, 700 epochs, external natural-image anomaly source,
# gray-128, base width 16, CUDA, matched augmentation stream per seed.
# Four arms with derivative matching OFF: base, in, cpl, incpl
# (the reconstruction network and its ordinary loss remain in every
# arm); screw/capsule/bottle x seeds 42/0/1 = 36 fits.
# Deviations from original DRAEM listed in ref_recipe_protocol.json.
set -u
cd /home/edison/agents_project/pgdn_project
V=/home/edison/agents_project/scienctific/.venv/bin/python
MVTEC=/home/edison/VS_Projects/defectsGenerations/mvtec_dataset
export PYTHONUNBUFFERED=1

run() { local RT=$1; shift; echo "=== $RT START $(date +%s)" >> results/walltime.log; \
  "$@" > results/logs/${RT}.log 2>&1; rc=$?; \
  echo "=== $RT END rc=$rc $(date +%s)" >> results/walltime.log; \
  if [ $rc -ne 0 ]; then echo "FAILED: $RT"; fi; }

FLAGS_base="--no-physics"
FLAGS_in="--phys-channel on  --phys-rec-loss off --phys-couple-loss off"
FLAGS_cpl="--phys-channel off --phys-rec-loss off --phys-couple-loss on"
FLAGS_incpl="--phys-channel on  --phys-rec-loss off --phys-couple-loss on"

for SEED in 42 0 1; do
  for CAT in screw capsule bottle; do
    for ARM in base in cpl incpl; do
      TAG=ref_${CAT}_${ARM}_s${SEED}
      [ -f "results/raw/${TAG}_eval.json" ] && { echo "skip $TAG"; continue; }
      run ${TAG}_train $V train.py --dataset mvtec --data-root $MVTEC/$CAT \
        --run-name $TAG --epochs 700 --seed $SEED --lambda-phys 0.5 \
        $(eval echo \$FLAGS_${ARM}) \
        --lr 1e-4 --scheduler multistep --anomaly-source external \
        --source-dir data/src_screw --device cuda
      run ${TAG}_eval $V eval.py --dataset mvtec --data-root $MVTEC/$CAT \
        --checkpoint runs/$TAG/last.pt --seed $SEED \
        --out results/raw/${TAG}_eval.json --device cuda
    done
  done
done
echo "REF_RECIPE_FITS_DONE"
