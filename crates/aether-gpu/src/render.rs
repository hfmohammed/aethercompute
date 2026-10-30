//! GPU-side rendering of simulation state straight from the storage buffers.

use aether_core::{Workload, WorkloadKind};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::context::{GpuContext, GpuError};
use crate::sim::GpuWorkload;

const PARTICLE_RENDER_WGSL: &str = include_str!("../../../shaders/particle_render.wgsl");
const HEAT_RENDER_WGSL: &str = include_str!("../../../shaders/heat_render.wgsl");

/// Camera + colour scaling. 32 bytes, matches the WGSL `View` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct View {
    pub center: [f32; 2],
    pub zoom: f32,
    pub aspect: f32,
    /// Quad half-size in NDC.
    pub point_size: f32,
    /// Speed (particles) or temperature (heat) mapped to the top of the ramp.
    pub vmax: f32,
    pub grid: u32,
    pub _pad: u32,
}

impl View {
    pub fn new(aspect: f32) -> Self {
        Self {
            center: [0.0, 0.0],
            zoom: 0.9,
            aspect,
            point_size: 0.004,
            vmax: 1.0,
            grid: 1,
            _pad: 0,
        }
    }

    /// Sensible defaults for a workload.
    pub fn for_workload(kind: WorkloadKind, grid: u32, count: u32, aspect: f32) -> Self {
        let mut v = Self::new(aspect);
        v.grid = grid;
        if kind == WorkloadKind::NbodyAllpairs {
            v.vmax = 3.0;
            v.zoom = 1.8;
            v.point_size = 0.004;
        } else if kind.is_particles() {
            v.vmax = 1.2;
            v.point_size = match count {
                0..=2_000 => 0.008,
                2_001..=20_000 => 0.004,
                _ => 0.0025,
            };
        } else {
            v.vmax = 1.0;
        }
        v
    }
}

pub struct Renderer {
    particle: wgpu::RenderPipeline,
    heat: wgpu::RenderPipeline,
    particle_layout: wgpu::BindGroupLayout,
    heat_layout: wgpu::BindGroupLayout,
    view_buf: wgpu::Buffer,
    pub format: wgpu::TextureFormat,
}

impl Renderer {
    pub fn new(ctx: &GpuContext, format: wgpu::TextureFormat) -> Result<Self, GpuError> {
        ctx.validated(|| {
            let d = &ctx.device;
            let layout_entries = |vis| {
                [
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: vis,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ]
            };
            let particle_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particle-render-layout"),
                entries: &layout_entries(wgpu::ShaderStages::VERTEX),
            });
            let heat_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("heat-render-layout"),
                entries: &layout_entries(wgpu::ShaderStages::FRAGMENT),
            });

            let additive = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            };
            let make = |label: &str, src: &str, layout: &wgpu::BindGroupLayout, blend| {
                let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                });
                let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(layout)],
                    immediate_size: 0,
                });
                d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };
            let particle = make(
                "particle-render",
                PARTICLE_RENDER_WGSL,
                &particle_layout,
                Some(additive),
            );
            let heat = make("heat-render", HEAT_RENDER_WGSL, &heat_layout, None);
            let view_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("view"),
                contents: bytemuck::bytes_of(&View::new(1.0)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            Self {
                particle,
                heat,
                particle_layout,
                heat_layout,
                view_buf,
                format,
            }
        })
    }

    /// Record a frame into `target`; the simulation's current buffer is read
    /// directly by the shaders, so no CPU readback happens.
    pub fn draw(
        &self,
        ctx: &GpuContext,
        sim: &GpuWorkload,
        view: &View,
        target: &wgpu::TextureView,
    ) {
        ctx.queue
            .write_buffer(&self.view_buf, 0, bytemuck::bytes_of(view));
        let particles = sim.config().workload.is_particles();
        let layout = if particles {
            &self.particle_layout
        } else {
            &self.heat_layout
        };
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render-bind-group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim.current_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.view_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.015,
                            g: 0.017,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if particles {
                pass.set_pipeline(&self.particle);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..6, 0..sim.config().count);
            } else {
                pass.set_pipeline(&self.heat);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        ctx.queue.submit([enc.finish()]);
    }
}

/// Render one frame offscreen and return tightly packed RGBA8 pixels.
pub fn render_to_rgba(
    ctx: &GpuContext,
    sim: &GpuWorkload,
    width: u32,
    height: u32,
    view: &View,
) -> Result<Vec<u8>, GpuError> {
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let renderer = Renderer::new(ctx, format)?;
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let tv = tex.create_view(&Default::default());
    renderer.draw(ctx, sim, view, &tv);

    let bpr = (width * 4).div_ceil(256) * 256;
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen-readback"),
        size: bpr as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen-copy"),
        });
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([enc.finish()]);
    let padded = crate::context::map_read(ctx, &buf, bpr as u64 * height as u64)?;
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height as usize {
        let s = row * bpr as usize;
        out.extend_from_slice(&padded[s..s + (width * 4) as usize]);
    }
    Ok(out)
}
