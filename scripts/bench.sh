#!/usr/bin/env bash
# Reproduces results/*.json on the current machine. Close other apps first and
# stay on AC power; see docs/benchmarking.md.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p aether-cli
A=target/release/aether
$A bench -w particles-local --count 1000,10000,100000 --steps 200 --warmup 5 --repeats 20 \
  --workgroup-sweep 64,128,256 -o results/m2-particles-local.json --csv results/m2-particles-local.csv --plot-dir docs/img
$A bench -w heat --grid 256,512,1024 --steps 200 --warmup 5 --repeats 20 \
  --workgroup-sweep 64,128,256 -o results/m2-heat.json --csv results/m2-heat.csv --plot-dir docs/img
$A bench -w nbody-allpairs --count 256,1024,4096 --steps 20 --warmup 5 --repeats 20 \
  --workgroup-sweep 64,128,256 -o results/m2-nbody.json --csv results/m2-nbody.csv --plot-dir docs/img

# Optimisation experiment: one compute pass per step vs all steps in one pass.
for mode in per-step batched; do
  $A bench -w heat --grid 256,512,1024 --backend gpu --steps 200 --warmup 5 --repeats 20 \
    --pass-mode $mode -o results/experiments/m2-heat-$mode.json
  $A bench -w particles-local --count 1000,10000,100000 --backend gpu --steps 200 --warmup 5 --repeats 20 \
    --pass-mode $mode -o results/experiments/m2-particles-local-$mode.json
done
