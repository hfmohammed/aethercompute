#!/usr/bin/env bash
# One-command demo: build, validate all workloads, run a short benchmark, open the viewer.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p aether-cli
A=target/release/aether
$A info
for w in heat particles-local nbody-allpairs; do
  size=$([ "$w" = heat ] && echo 256 || ([ "$w" = nbody-allpairs ] && echo 1024 || echo 10000))
  $A validate -w "$w" --size "$size" --steps 50
done
$A bench -w heat --grid 256,512 --steps 100 --warmup 3 --repeats 10
echo "Opening the viewer (Esc to quit) ..."
$A viewer -w particles-local --count 100000
