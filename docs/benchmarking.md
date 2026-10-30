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
host. For GPU records the harness additionally reports **device execution
time** from timestamp queries (below); the two series are never merged.

Timing samples are not independent (thermal throttling, boost clocks), so the
Student-t interval understates uncertainty; quote the bootstrap median interval.
`order_drift` is the median of the second half of the samples (in run order)
over the median of the first half; the runner prints a warning when it is more
than 10% from 1.0.

## GPU timestamps

When the adapter supports `TIMESTAMP_QUERY` (it is the only optional feature
requested, and only if present; Apple GPUs need nothing more for pass-boundary
timestamps) every `step(n)` batch is bracketed by two timestamps: the begin
timestamp is written by the first compute pass and the end timestamp by the last
one, so the interval covers the whole batch in both `batched` and `per-step`
pass modes. The ticks are resolved into a small buffer inside the same command
buffer; that buffer is mapped and read only after the host compute timer has
stopped, so reading it does not inflate `compute_ms`.

Per record, kept as separate series next to the host wall-clock ones:

| field | meaning |
|-------|---------|
| `samples.device_compute_ms`, `device_compute` | GPU execution time per sample and its summary (median, bootstrap CI, drift) |
| `samples.host_device_gap_ms`, `host_device_gap` | `host compute - device compute`, sample by sample. A **host/device timing gap**: it holds command encoding, submission, queueing and synchronisation latency, not just submission overhead |
| `median_device_step_ms` | median device time / `steps` |
| `device_timing` | `available`, `reason`, `period_ns` (ns per tick), `feature` |

CPU backends and adapters without the feature have no device series; wall-clock
results are still complete and `device_timing.available` is `false` with the
reason. `--require-gpu-timestamps` turns that absence into a failure (exit 2).
The CSV carries `device_median_ms`, its bootstrap interval,
`host_device_gap_median_ms`, `device_timestamps` and `timestamp_period_ns`.
Timestamps measure GPU execution only; the speed-up tables still use host
wall-clock compute and end-to-end times.

## Sample state: `--bench-mode`

| Mode | Behaviour |
|------|-----------|
| `reset` (default) | The workload is re-initialised (untimed) before every sample, so each sample performs identical work from the identical state. After the last sample the state is compared with the scalar reference again (`final_validation`), so what was *timed* is what was validated. |
| `continuous` | One simulation keeps evolving through warm-up and all samples. Models a sustained run, but later samples start from different states (heat converges, particles disperse). The final state is checked for non-finite values (`final_nonfinite`). |

## Run order and thermal bias

Within each problem size, backends and GPU workgroup sizes run in a **seeded
random order**. The seed is random per run unless `--order-seed N` is given (it
is always recorded), so repeated runs place backends in different thermal
positions; for a headline number, run three times and compare. `--no-shuffle`
gives a fixed order. A fixed seed only relocates bias, it does not remove it. The executed order is
stored in `run_order` and each record's `run_index`; the report itself stays in
canonical (size, backend, workgroup) order. `--cooldown-ms N` idles before
every run. The JSON also stores the exact command line (`command`).

## Config and memory validation

Floating-point inputs (`dt`, `softening`, `force_scale`, `damping`, `alpha`,
`diffusion_number`) must be finite, `alpha` must be positive, and
`damping*dt` must stay below 1 (otherwise the drag factor flips sign). A
benchmark also needs `steps >= 1`. Each size's *approximate* footprint (three
copies of the state) must fit within 60% of physical RAM (read from `sysctl` on
macOS, `/proc/meminfo` on Linux; unknown RAM disables the check). It is an
estimate: the scalar reference stays resident during other runs, and temporary
host/GPU copies can exceed it.

## Record status

Each record is `success`, `skipped-unavailable` (no GPU adapter on this
machine) or `failed` (the run errored, e.g. shader compile or device loss).
Only `skipped-unavailable` is optional (`NoAdapter` is the sole error that maps
to it; a device that fails to open on a machine with an adapter is `failed`); `failed` and numerically invalid
records always fail the command (exit code 2) and never contribute to speed-ups
or charts, and neither does a CPU baseline that failed validation.

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

## Aggregating and comparing runs: `aether compare`

One report is one *run*; runs are the replicates. Never pool samples across
runs: samples inside a run are autocorrelated, so the run is the unit.

```bash
# three runs with different order seeds (the default seed is random per run)
for i in 1 2 3; do aether bench ... --order-seed $i -o results/run$i.json; done

# aggregate: median of per-run medians, min/max spread, per-run values
aether compare results/run1.json results/run2.json results/run3.json --markdown table.md

# compare candidate runs with baseline runs; exit code 2 on regression
aether compare cand1.json cand2.json cand3.json \
    --baseline base1.json base2.json base3.json --threshold 0.10 --metric compute
```

* Records are aligned by workload, size, backend, GPU workgroup and pass mode,
  steps, and every numerical parameter (seed, dt, softening, force scale,
  damping, boundary, diffusion number, alpha). Records with no counterpart are
  reported and skipped; if nothing aligns the command fails.
* Only successful, validated records are used. Failed, skipped and invalid
  records are excluded (and counted in a warning).
* `--metric` picks one series: `compute` (host wall-clock), `device` (GPU
  timestamps), `gap`, `readback` or `end-to-end`. They are never merged.
* A **regression** needs the candidate median-of-medians to exceed the baseline
  by more than `--threshold` *and* every candidate run to be slower than every
  baseline run. Beyond the threshold with overlapping runs, or with fewer than
  two runs per side, is **inconclusive** (exit 2 only with `--strict`).
* Reset and continuous reports, and legacy schema-v1 reports (continuous), are
  refused unless `--allow-mode-mismatch` is given.
* Warnings: dirty working trees, non-release builds, battery power, differing
  chip / OS / rustc / adapter / commit between runs, reused order seeds, and a
  config present in only some runs.
* Outputs: terminal table, `--json` (full result including per-run values) and
  `--markdown` (a PR-ready comparison table, or in aggregate mode README-style
  tables of per-run speed-up ratios with min-max ranges).

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
