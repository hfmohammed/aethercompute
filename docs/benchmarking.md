# Benchmark methodology

## What is measured

For each (workload, size, backend[, workgroup size]) the harness records:

| Phase       | Definition |
|-------------|------------|
| cold start  | `setup_ms`: workload construction, including buffer upload and (GPU) shader/pipeline compilation. `first_run_ms`: the first `steps` steps after construction. |
| warm-up     | `--warmup` unmeasured runs of `--steps` steps (default 3+; the provided scripts use 5). |
| compute     | Wall time of `steps` simulation steps, from enqueue until the device reports idle. GPU: one submit of N compute passes + wait. CPU: the loop. Excludes upload and readback. |
| readback    | Wall time to copy the final state to host memory (GPU: copy to staging, map, memcpy; CPU: a buffer clone). |
| end-to-end  | compute + readback, per sample. State stays resident between samples, so there is no per-sample upload; upload is part of cold start. |

Each measured repeat produces one sample of each. All raw samples are stored in
the JSON; the summary gives n, min, median, mean, p95, p99, max, stddev, the
95% confidence half-width of the mean (Student t), a **bootstrap 95% interval
of the median** (2000 resamples, fixed seed) and `order_drift`. Medians are used
for throughput and speed-up. These are *wall-clock* times measured from the
host; no GPU timestamp queries are used yet, so "GPU compute" includes
submission and synchronisation overhead.

Timing samples are not independent (thermal throttling, boost clocks), so the
Student-t interval understates uncertainty; quote the bootstrap median interval.
`order_drift` is the median of the second half of the samples (in run order)
over the median of the first half; the runner prints a warning when it is more
than 10% from 1.0.

## Sample state: `--bench-mode`

| Mode | Behaviour |
|------|-----------|
| `reset` (default) | The workload is re-initialised (untimed) before every sample, so each sample performs identical work from the identical state. After the last sample the state is compared with the scalar reference again (`final_validation`), so what was *timed* is what was validated. |
| `continuous` | One simulation keeps evolving through warm-up and all samples. Models a sustained run, but later samples start from different states (heat converges, particles disperse). The final state is checked for non-finite values (`final_nonfinite`). |

## Run order and thermal bias

Within each problem size, backends and GPU workgroup sizes run in a **seeded
random order** (`--order-seed`, default 1; `--no-shuffle` for fixed order), so
no backend systematically runs on a hot or cold machine. The executed order is
stored in `run_order` and each record's `run_index`; the report itself stays in
canonical (size, backend, workgroup) order. `--cooldown-ms N` idles before
every run. The JSON also stores the exact command line (`command`).

## Config and memory validation

Floating-point inputs (`dt`, `softening`, `force_scale`, `damping`, `alpha`,
`diffusion_number`) must be finite, `alpha` must be positive, and
`damping*dt` must stay below 1 (otherwise the drag factor flips sign). A
benchmark also needs `steps >= 1`. Each size's estimated footprint (state
buffers plus readback copies) must fit within 60% of physical RAM.

## Correctness gate

Every record is validated before warm-up: the state after the first `steps`
steps is compared with a scalar-CPU reference computed from the same seed (and,
in `reset` mode, again after the final sample). GPU
rows use the tolerances in `numerics.md`; Rayon rows must be bit-identical. A
fast but incorrect backend is a failed result: `aether bench` exits with code 2
and the row is marked `FAIL`.

## Speed-up reporting

Speed-up is `median(CPU time) / median(GPU time)`, reported against **both** the
scalar and the Rayon baseline, for compute only and end-to-end (including
readback). Rayon uses all logical cores of the machine (8 on the M2). When
several GPU workgroup sizes are swept, charts show the best workgroup size per
problem size and the CSV/JSON keep all of them.

Throughput is `updates_per_step * steps / median_compute_seconds`, in millions
of particle-updates or cell-updates per second. N-body and local-particle
throughputs are not comparable with each other (O(N^2) vs O(N) work per update).

## Reproducing

```bash
./scripts/bench.sh        # writes results/*.json, *.csv and docs/img/*.svg
```

or individually:

```bash
cargo build --release -p aether-cli
target/release/aether bench -w heat --grid 512 --steps 200 --warmup 5 --repeats 20 \
    --workgroup-sweep 64,128,256 -o results/m2-heat.json --csv results/m2-heat.csv --plot-dir docs/img
```

Each JSON records: schema version, git commit (and whether the tree was dirty),
timestamp, OS/chip/RAM/power source, rustc version, build profile, adapter name,
backend, limits, the full `ExperimentConfig` (workload, size, steps, seed,
workgroup size, dt, ...), the timing-scope text above, raw samples, cold-start
times and the validation error metrics. `aether plot report.json` regenerates
the SVG charts from a saved report.

## Controlling the environment

* Use a release build (`cargo build --release`); the report records the build profile.
* Stay on AC power (recorded in `machine.power_source`) and close other apps; let the
  machine cool between sweeps. Thermal throttling is not directly observable from
  user space, so compare the p95/max against the median in the JSON to spot it.
* Do not run `cargo` jobs while benchmarking.
* The `--steps` per sample can be raised for very fast kernels so each sample is
  long relative to timer and submit overhead; for the O(N^2) mode it is lowered.

## What these numbers do not show

* No GPU timestamp queries: GPU-only time is inferred from wall clock around
  submit + wait, so it includes submission overhead (small at 200 steps per submit).
* Results are for one machine (see `results/` for the exact hardware). Nothing
  here predicts other GPUs.
* CI runs CPU-only checks; GPU differential tests are executed locally on Apple
  Silicon (they skip when no adapter is present).

## Profiling

App-level timings above need no extra tooling. For a GPU frame/dispatch capture
use Xcode: run `target/release/aether bench ...` (or the viewer) under
Instruments' Metal System Trace, or set `MTL_CAPTURE_ENABLED=1` and capture a
Metal workload from Xcode. This requires the full Xcode app; Command Line Tools
are enough to build and run everything else.
