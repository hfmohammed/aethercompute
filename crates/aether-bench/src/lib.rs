//! Experiment harness: workload factory, benchmark runner, machine metadata
//! and JSON/CSV/SVG export.

pub mod compare;
pub mod export;
pub mod machine;
pub mod plot;
pub mod record;
pub mod runner;

pub use machine::MachineInfo;
pub use record::{
    BenchMode, BenchRecord, BenchReport, ColdStart, RecordStatus, Samples, Speedup, TimingScope,
};
pub use runner::{classify_gpu_error, create_workload, run_sweep, BenchError, SweepSpec};
