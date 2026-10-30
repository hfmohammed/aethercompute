//! GPU implementation of the [`Workload`] trait.

use std::sync::Arc;

use aether_core::{
    heat, particles, DeviceTimingInfo, ExperimentConfig, PassMode, Snapshot, Workload,
    WorkloadError, WorkloadKind,
};
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::context::{map_read, GpuContext, GpuError};

const PARTICLES_LOCAL_WGSL: &str = include_str!("../../../shaders/particles_local.wgsl");
const NBODY_WGSL: &str = include_str!("../../../shaders/nbody_allpairs.wgsl");
const HEAT_WGSL: &str = include_str!("../../../shaders/heat_step.wgsl");

/// Pass-boundary timestamp plumbing: two ticks (begin, end) around the whole
/// `step(n)` batch, resolved on the GPU into a small buffer that is only read
/// back by `take_device_time_ms`, outside any host timer.
struct StepTimer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// A batch has been timed since the last `take_device_time_ms`.
    pending: bool,
}

const TIMESTAMP_BYTES: u64 = 16;

/// Persistent-buffer GPU simulation: state ping-pongs between two storage
/// buffers, the uniform block is written once, and the staging buffer is only
/// touched by `snapshot`.
pub struct GpuWorkload {
    ctx: Arc<GpuContext>,
    cfg: ExperimentConfig,
    buffers: [wgpu::Buffer; 2],
    uniform: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    pipeline: wgpu::ComputePipeline,
    /// Index of the buffer that holds the current state.
    cur: usize,
    state_bytes: u64,
    groups: (u32, u32),
    timer: Option<StepTimer>,
}

/// Split a 1D workgroup count into (x, y) so neither exceeds `max`.
pub fn fold_dispatch(groups: u64, max: u32) -> Result<(u32, u32), GpuError> {
    let max = max as u64;
    let gx = groups.min(max).max(1);
    let gy = groups.div_ceil(gx).max(1);
    if gy > max {
        return Err(GpuError::Limit(format!(
            "dispatch of {groups} workgroups exceeds device limit ({max} x {max})"
        )));
    }
    Ok((gx as u32, gy as u32))
}

impl GpuWorkload {
    pub fn new(ctx: Arc<GpuContext>, cfg: ExperimentConfig) -> Result<Self, WorkloadError> {
        cfg.validate()?;
        let d = &ctx.device;
        let lim = &ctx.report.limits;

        let wg = cfg.workgroup_size;
        if wg > lim.max_compute_workgroup_size_x || wg > lim.max_compute_invocations_per_workgroup {
            return Err(GpuError::Limit(format!(
                "workgroup size {wg} exceeds adapter limits (x: {}, invocations: {})",
                lim.max_compute_workgroup_size_x, lim.max_compute_invocations_per_workgroup
            ))
            .into());
        }

        let (initial_bytes, uniform_bytes, wgsl, items): (Vec<u8>, Vec<u8>, &str, u64) =
            if cfg.workload.is_particles() {
                let init = particles::initial_particles(&cfg);
                let params = cfg.particle_params();
                let src = if cfg.workload == WorkloadKind::NbodyAllpairs {
                    NBODY_WGSL
                } else {
                    PARTICLES_LOCAL_WGSL
                };
                (
                    cast_slice(&init).to_vec(),
                    bytemuck::bytes_of(&params).to_vec(),
                    src,
                    cfg.count as u64,
                )
            } else {
                let init = heat::initial_grid(&cfg);
                let params = cfg.heat_params()?;
                (
                    cast_slice(&init).to_vec(),
                    bytemuck::bytes_of(&params).to_vec(),
                    HEAT_WGSL,
                    cfg.grid as u64 * cfg.grid as u64,
                )
            };

        let state_bytes = initial_bytes.len() as u64;
        if state_bytes > lim.max_buffer_size || state_bytes > lim.max_storage_buffer_binding_size {
            return Err(GpuError::Limit(format!(
                "state of {state_bytes} bytes exceeds adapter limits (max buffer {}, max storage binding {})",
                lim.max_buffer_size, lim.max_storage_buffer_binding_size
            ))
            .into());
        }
        let groups = fold_dispatch(
            items.div_ceil(wg as u64),
            lim.max_compute_workgroups_per_dimension,
        )?;

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let built = ctx.validated(|| {
            let make = |label: &str| {
                d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &initial_bytes,
                    usage,
                })
            };
            let buffers = [make("state-a"), make("state-b")];
            let uniform = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: &uniform_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let staging = d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: state_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("step-shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
            let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("step-layout"),
                entries: &[
                    storage(0, true),
                    storage(1, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("step-pipeline-layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("step-pipeline"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("WG", wg as f64)],
                    ..Default::default()
                },
                cache: None,
            });
            let bg = |read: &wgpu::Buffer, write: &wgpu::Buffer| {
                d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("step-bind-group"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: read.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: write.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform.as_entire_binding(),
                        },
                    ],
                })
            };
            let bind_groups = [bg(&buffers[0], &buffers[1]), bg(&buffers[1], &buffers[0])];
            (buffers, uniform, staging, bind_groups, pipeline)
        })?;
        let (buffers, uniform, staging, bind_groups, pipeline) = built;

        let timer = ctx.timestamps.then(|| StepTimer {
            queries: d.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("step-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp-resolve"),
                size: TIMESTAMP_BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp-readback"),
                size: TIMESTAMP_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            pending: false,
        });

        Ok(Self {
            ctx,
            cfg,
            buffers,
            uniform,
            staging,
            bind_groups,
            pipeline,
            cur: 0,
            state_bytes,
            groups,
            timer,
        })
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Storage buffer holding the current state (for GPU-side rendering).
    pub fn current_buffer(&self) -> &wgpu::Buffer {
        &self.buffers[self.cur]
    }

    pub fn state_bytes(&self) -> u64 {
        self.state_bytes
    }

    pub fn dispatch_groups(&self) -> (u32, u32) {
        self.groups
    }

    pub fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    fn upload_initial(&mut self) {
        let bytes: Vec<u8> = if self.cfg.workload.is_particles() {
            cast_slice(&particles::initial_particles(&self.cfg)).to_vec()
        } else {
            cast_slice(&heat::initial_grid(&self.cfg)).to_vec()
        };
        self.ctx.queue.write_buffer(&self.buffers[0], 0, &bytes);
        self.cur = 0;
    }
}

impl Workload for GpuWorkload {
    fn describe(&self) -> String {
        format!(
            "gpu ({}, {})",
            self.ctx.report.name, self.ctx.report.backend
        )
    }

    fn config(&self) -> &ExperimentConfig {
        &self.cfg
    }

    fn initialize(&mut self) -> Result<(), WorkloadError> {
        self.upload_initial();
        self.ctx.wait_idle()?;
        Ok(())
    }

    fn step(&mut self, steps: u32) -> Result<(), WorkloadError> {
        if steps == 0 {
            return Ok(());
        }
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("steps"),
            });
        let (pipeline, groups) = (&self.pipeline, self.groups);
        let mut cur = self.cur;
        let mut record = |pass: &mut wgpu::ComputePass<'_>| {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_groups[cur], &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
            cur ^= 1;
        };
        // Begin timestamp on the first pass, end timestamp on the last, so the
        // interval covers the complete batch in either pass mode.
        let timer = self.timer.as_ref();
        let writes = |begin: bool, end: bool| {
            // Middle passes of a per-step batch write nothing.
            timer
                .filter(|_| begin || end)
                .map(|t| wgpu::ComputePassTimestampWrites {
                    query_set: &t.queries,
                    beginning_of_pass_write_index: begin.then_some(0),
                    end_of_pass_write_index: end.then_some(1),
                })
        };
        match self.cfg.pass_mode {
            PassMode::PerStep => {
                for i in 0..steps {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("step"),
                        timestamp_writes: writes(i == 0, i + 1 == steps),
                    });
                    record(&mut pass);
                }
            }
            PassMode::Batched => {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("steps"),
                    timestamp_writes: writes(true, true),
                });
                for _ in 0..steps {
                    record(&mut pass);
                }
            }
        }
        if let Some(t) = &self.timer {
            enc.resolve_query_set(&t.queries, 0..2, &t.resolve, 0);
            enc.copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, TIMESTAMP_BYTES);
        }
        if let Some(t) = &mut self.timer {
            t.pending = true;
        }
        self.cur = cur;
        self.ctx.queue.submit([enc.finish()]);
        Ok(())
    }

    fn device_timing_info(&self) -> DeviceTimingInfo {
        if self.timer.is_some() {
            DeviceTimingInfo {
                available: true,
                reason: "GPU execution time between pass-boundary timestamps around the whole step batch".into(),
                period_ns: Some(self.ctx.queue.get_timestamp_period()),
                feature: Some("TIMESTAMP_QUERY".into()),
            }
        } else {
            DeviceTimingInfo::unavailable("adapter does not support TIMESTAMP_QUERY")
        }
    }

    fn take_device_time_ms(&mut self) -> Result<Option<f64>, WorkloadError> {
        let Some(t) = self.timer.as_mut().filter(|t| t.pending) else {
            return Ok(None);
        };
        t.pending = false;
        let bytes = map_read(&self.ctx, &t.readback, TIMESTAMP_BYTES)?;
        let tick = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
        let period_ns = self.ctx.queue.get_timestamp_period() as f64;
        Ok(tick(1)
            .checked_sub(tick(0))
            .map(|d| d as f64 * period_ns / 1e6))
    }

    fn sync(&mut self) -> Result<(), WorkloadError> {
        self.ctx.wait_idle()?;
        Ok(())
    }

    fn snapshot(&mut self) -> Result<Snapshot, WorkloadError> {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        enc.copy_buffer_to_buffer(
            &self.buffers[self.cur],
            0,
            &self.staging,
            0,
            self.state_bytes,
        );
        self.ctx.queue.submit([enc.finish()]);
        let bytes = map_read(&self.ctx, &self.staging, self.state_bytes)?;
        Ok(if self.cfg.workload.is_particles() {
            Snapshot::Particles(cast_slice::<u8, particles::Particle>(&bytes).to_vec())
        } else {
            Snapshot::Heat(cast_slice::<u8, f32>(&bytes).to_vec())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_small() {
        assert_eq!(fold_dispatch(10, 65535).unwrap(), (10, 1));
        assert_eq!(fold_dispatch(1, 65535).unwrap(), (1, 1));
    }

    #[test]
    fn fold_large() {
        let (x, y) = fold_dispatch(200_000, 65535).unwrap();
        assert_eq!(x, 65535);
        assert!(x as u64 * y as u64 >= 200_000);
        assert!(y <= 65535);
    }

    #[test]
    fn fold_overflow_is_error() {
        assert!(fold_dispatch(65535u64 * 65535 + 1, 65535).is_err());
    }
}
