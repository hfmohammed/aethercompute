#!/usr/bin/env bash
# Official benchmark runner: three runs with fixed, different order seeds,
# aggregated with `aether compare`, published into the repository only after
# every run succeeded.
#
#   ./scripts/bench.sh            official run (clean tree + AC power required)
#   ./scripts/bench.sh --check    only run the preflight checks
#   ./scripts/bench.sh --smoke    tiny sizes, no AC/clean requirement, never
#                                 publishes: exercises the pipeline end to end
#
# Every run is written to a fresh staging directory OUTSIDE the repository, so
# the working tree stays clean for the whole sweep and runs cannot overwrite
# each other. Staging results are never deleted, including after a failure.
#
#   <staging>/seed-101/{particles,heat,nbody}.{json,csv}
#   <staging>/seed-101/{heat,particles}-{per-step,batched}.json   (pass-mode experiment)
#   <staging>/seed-202/..., seed-303/...
#   <staging>/aggregate/*.{md,json,txt}
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=official
case "${1:-}" in
  "") ;;
  --check) MODE=check ;;
  --smoke) MODE=smoke ;;
  *) echo "usage: $0 [--check|--smoke]" >&2; exit 64 ;;
esac

SEEDS=(101 202 303)
A=target/release/aether

die() { echo "error: $*" >&2; exit 1; }

# Re-checked before every run: the laptop can be unplugged, or a file edited,
# halfway through a sweep that takes a while.
preflight() {
  [[ $MODE == smoke ]] && return 0
  [[ -z "$(git status --porcelain --untracked-files=no)" ]] \
    || die "working tree is dirty; commit or stash first so the report records a clean commit"
  command -v pmset >/dev/null \
    || die "official runs need macOS (pmset) to verify AC power"
  pmset -g batt | head -1 | grep -q "AC Power" \
    || die "not on AC power: $(pmset -g batt | head -1)"
}

preflight
if [[ $MODE == check ]]; then
  echo "preflight ok: clean tree, AC power"
  exit 0
fi

if [[ $MODE == smoke ]]; then
  PSZ=1000,2000; HSZ=64,128; NSZ=64,128
  STEPS_P=20; STEPS_H=20; STEPS_N=5; WARM=1; REP=3; WG=64
else
  PSZ=1000,10000,100000; HSZ=256,512,1024; NSZ=256,1024,4096
  STEPS_P=200; STEPS_H=200; STEPS_N=20; WARM=5; REP=20; WG=64,128,256
fi

cargo build --release --locked -p aether-cli
SHA=$(git rev-parse --short HEAD)
STAGE="${TMPDIR:-/tmp}/aether-bench-$MODE-$SHA-$(date +%Y%m%d-%H%M%S)"
[[ ! -e $STAGE ]] || die "staging directory $STAGE already exists"
mkdir -p "$STAGE/aggregate"
echo "staging: $STAGE (kept even if a run fails)"

COMMON=(--bench-mode reset --require-gpu --require-gpu-timestamps)

for seed in "${SEEDS[@]}"; do
  d="$STAGE/seed-$seed"
  mkdir -p "$d"
  for w in particles heat nbody; do
    preflight
    echo "== seed $seed: $w"
    case $w in
      particles) ARGS=(-w particles-local --count "$PSZ" --steps "$STEPS_P") ;;
      heat)      ARGS=(-w heat --grid "$HSZ" --steps "$STEPS_H") ;;
      nbody)     ARGS=(-w nbody-allpairs --count "$NSZ" --steps "$STEPS_N") ;;
    esac
    $A bench "${ARGS[@]}" --warmup "$WARM" --repeats "$REP" --workgroup-sweep "$WG" \
      "${COMMON[@]}" --order-seed "$seed" -o "$d/$w.json" --csv "$d/$w.csv"
  done

  # Optimisation experiment: one compute pass per step vs one pass per batch.
  for mode in per-step batched; do
    preflight
    echo "== seed $seed: pass-mode $mode"
    $A bench -w heat --grid "$HSZ" --steps "$STEPS_H" --backend gpu --pass-mode "$mode" \
      --warmup "$WARM" --repeats "$REP" "${COMMON[@]}" --order-seed "$seed" \
      -o "$d/heat-$mode.json"
    $A bench -w particles-local --count "$PSZ" --steps "$STEPS_P" --backend gpu --pass-mode "$mode" \
      --warmup "$WARM" --repeats "$REP" "${COMMON[@]}" --order-seed "$seed" \
      -o "$d/particles-$mode.json"
  done
done

# --- aggregate: runs (seeds) are the replicates -----------------------------
NAMES="particles heat nbody heat-per-step heat-batched particles-per-step particles-batched"
for name in $NAMES; do
  $A compare "$STAGE"/seed-*/"$name".json \
    --markdown "$STAGE/aggregate/$name.md" --json "$STAGE/aggregate/$name.json" \
    | tee "$STAGE/aggregate/$name.txt"
done
# Batched vs per-step pass: candidate = batched, baseline = per-step (ratio < 1 is faster).
for w in heat particles; do
  rc=0
  $A compare "$STAGE"/seed-*/"$w-batched.json" --baseline "$STAGE"/seed-*/"$w-per-step.json" --ignore-pass-mode \
    --markdown "$STAGE/aggregate/$w-batching-vs-per-step.md" \
    --json "$STAGE/aggregate/$w-batching-vs-per-step.json" \
    | tee "$STAGE/aggregate/$w-batching-vs-per-step.txt" || rc=$?
  [[ $rc -eq 0 ]] || echo "note: batched vs per-step comparison exited $rc (see $w-batching-vs-per-step.txt)"
done

if [[ $MODE == smoke ]]; then
  echo "smoke run complete; nothing published. Results are in $STAGE"
  exit 0
fi

# --- publish: only reached when every run above succeeded -------------------
preflight
if compgen -G "results/m2-*" >/dev/null || [[ -d results/experiments ]]; then
  mkdir -p results/legacy-continuous-v1
  for f in results/m2-*; do [[ -e $f ]] && mv -n "$f" results/legacy-continuous-v1/; done
  [[ -d results/experiments ]] && mv -n results/experiments results/legacy-continuous-v1/experiments
fi
mkdir -p results/runs results/aggregate
cp -R "$STAGE"/seed-* results/runs/
cp "$STAGE"/aggregate/* results/aggregate/
for w in particles heat nbody; do
  $A plot "results/runs/seed-${SEEDS[0]}/$w.json" -o docs/img
done
cat <<EOF

published into results/ (raw runs in results/runs/, aggregates in results/aggregate/,
previous continuous-mode results archived in results/legacy-continuous-v1/).
Charts in docs/img come from the seed-${SEEDS[0]} run; the tables use all three.
Next: copy the tables from results/aggregate/*.md into README.md, then commit.
EOF
