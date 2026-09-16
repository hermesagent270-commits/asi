#!/usr/bin/env bash
set -u

root="$(git rev-parse --show-toplevel)"
cd "$root" || exit 1

arm="$1"
seed="$2"
campaign="outputs/ipmnist_screening/intentional_updates_r1"
out="$campaign/shards/${arm}_seed${seed}.json"
log="$campaign/logs/${arm}_seed${seed}.log"

mkdir -p "$campaign/shards" "$campaign/logs"
if [ -e "$out" ]; then
  echo "refusing occupied shard destination: $out" >&2
  exit 1
fi

OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$arm" \
  --seed "$seed" \
  --n-tasks 60 \
  --task-length 5000 \
  --noise-mode step \
  --out "$out" \
  --progress-every 20 >"$log" 2>&1
