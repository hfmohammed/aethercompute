//! Experiment harness: workload factory, benchmark runner, machine metadata
//! and JSON/CSV/SVG export.

pub mod export;
pub mod machine;
pub mod plot;
pub mod record;
pub mod runner;

pub use machine::MachineInfo;
pub use record::{BenchRecord, BenchReport, ColdStart, Samples, Speedup, TimingScope};
pub use runner::{create_workload, run_sweep, BenchError, SweepSpec};
