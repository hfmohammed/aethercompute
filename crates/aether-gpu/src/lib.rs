//! wgpu/WGSL compute backend (Metal on macOS) plus a GPU-side renderer.

pub mod context;
pub mod render;
pub mod sim;

pub use context::{AdapterReport, GpuContext, GpuError, LimitsReport};
pub use render::{render_to_rgba, Renderer, View};
pub use sim::GpuWorkload;
