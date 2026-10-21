#!/bin/bash
# Review-driven experiments (CPU-safe): (a) the 2x2x2 factorial ablation of
# the three PGDN interventions (input channel / derivative reconstruction
# loss / coupling loss); (b) derivative-prior channel controls (grad, lap,
# var, shuffle); (c) official-DRAEM-protocol-style arms (lr 1e-4,
# MultiStepLR, 700 epochs, external anomaly source) auditing the weak-twin
# results.  Usage: bash factorial_driver.sh <shard 1-8>
set -u
cd /home/edison/agents_project/pgdn_project
PY=/home/edison/agents_project/scienctific/.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
MVTEC=/home/edison/VS_Projects/defectsGenerations/mvtec_dataset
V3=/home/edison/agents_project/pgdn_v3/data
SHARD=${1:?usage: factorial_driver.sh <1-8>}

run() { local RT=$1; shift; echo "=== $RT START $(date +%s)" >> results/walltime.log; \
  "$@" > results/logs/${RT}.log 2>&1; rc=$?; \
  echo "=== $RT END rc=$rc $(date +%s)" >> results/walltime.log; \
  if [ $rc -ne 0 ]; then echo "FAILED: $RT"; tail -3 results/logs/${RT}.log; fi; }

cfg2flags() {
  case $1 in
  base)   echo "--no-physics" ;;
  in)     echo "--phys-channel on  --phys-rec-loss off --phys-couple-loss off" ;;
  rec)    echo "--phys-channel off --phys-rec-loss on  --phys-couple-loss off" ;;
  cpl)    echo "--phys-channel off --phys-rec-loss off --phys-couple-loss on" ;;
  inrec)  echo "--phys-channel on  --phys-rec-loss on  --phys-couple-loss off" ;;
  incpl)  echo "--phys-channel on  --phys-rec-loss off --phys-couple-loss on" ;;
  recpl)  echo "--phys-channel off --phys-rec-loss on  --phys-couple-loss on" ;;
  full)   echo "--phys-channel on  --phys-rec-loss on  --phys-couple-loss on" ;;
  esac
}

bench() { # TAG ROOT DS EPOCHS CONFIG
  local TAG=$1 ROOT=$2 DS=$3 EP=$4 CFG=$5
  local FLAGS=$(cfg2flags $CFG)
  run fact_${TAG}_${CFG}_train $PY train.py --dataset $DS --data-root $ROOT \
    --run-name fact_${TAG}_${CFG} --epochs $EP --seed 42 --lambda-phys 0.5 $FLAGS --device cpu
  run fact_${TAG}_${CFG}_eval $PY eval.py --dataset $DS --data-root $ROOT \
    --checkpoint runs/fact_${TAG}_${CFG}/last.pt --seed 42 \
    --out results/raw/fact_${TAG}_${CFG}_eval.json --device cpu
  echo "FACTORIAL SHARD $SHARD: $TAG $CFG DONE $(date +%H:%M:%S)" >> results/walltime.log
}

# categories per shard: one shard per factorial config
CATS="screw bottle capsule cashew"
case $SHARD in
1) CFGS="base" ;;
2) CFGS="in" ;;
3) CFGS="rec" ;;
4) CFGS="cpl" ;;
5) CFGS="inrec" ;;
6) CFGS="incpl" ;;
7) CFGS="recpl" ;;
8) CFGS="full" ;;
esac

for CFG in $CFGS; do
  for C in $CATS; do
    bench $C $MVTEC/$C mvtec 8 $CFG
  done
  bench construct data/construct construct 6 $CFG
  echo "FACTORIAL SHARD $SHARD: config $CFG COMPLETE" >> results/walltime.log
done

# channel-variant controls on shard 8 (after full config): channel content
# swaps with both losses on, isolating the information in the channel
if [ "$SHARD" = "8" ]; then
  for VAR in grad lap var shuffle; do
    for T in screw $MVTEC/screw mvtec construct data/construct construct; do
      :;
    done
    for PAIR in "screw $MVTEC/screw mvtec 8" "construct data/construct construct 6"; do
      set -- $PAIR
      run ctrl_$1_${VAR}_train $PY train.py --dataset $3 --data-root $2 \
        --run-name ctrl_$1_${VAR} --epochs $4 --seed 42 --lambda-phys 0.5 \
        --channel-variant $VAR --device cpu
      run ctrl_$1_${VAR}_eval $PY eval.py --dataset $3 --data-root $2 \
        --checkpoint runs/ctrl_$1_${VAR}/last.pt --seed 42 \
        --out results/raw/ctrl_$1_${VAR}_eval.json --device cpu
    done
  done
fi
echo "FACTORIAL_SHARD_${SHARD}_DONE"
