//! Native interactive viewer (winit + wgpu). The simulation and renderer share
//! one device, so frames are drawn straight from the simulation's GPU buffers
//! with no per-frame readback.

mod app;
pub mod screenshot;

pub use app::run;
