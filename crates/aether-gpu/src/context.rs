use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter found: {0}")]
    NoAdapter(String),
    #[error("failed to create GPU device: {0}")]
    Device(String),
    #[error("{0}")]
    Limit(String),
    #[error("GPU validation error: {0}")]
    Validation(String),
    #[error("GPU readback failed: {0}")]
    Readback(String),
}

impl From<GpuError> for aether_core::WorkloadError {
    fn from(e: GpuError) -> Self {
        aether_core::WorkloadError::Gpu(e.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LimitsReport {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_texture_dimension_2d: u32,
}

/// What `aether info` prints and what benchmark JSON records (F02).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterReport {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub limits: LimitsReport,
    /// Optional features the adapter supports (none are required).
    pub optional_features: Vec<String>,
}

impl AdapterReport {
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let l = adapter.limits();
        let f = adapter.features();
        let mut optional_features = Vec::new();
        for (name, feat) in [
            ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY),
            ("SHADER_F16", wgpu::Features::SHADER_F16),
            ("SUBGROUP", wgpu::Features::SUBGROUP),
            ("PUSH_CONSTANTS", wgpu::Features::IMMEDIATES),
        ] {
            if f.contains(feat) {
                optional_features.push(name.to_string());
            }
        }
        Self {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            driver: info.driver,
            driver_info: info.driver_info,
            vendor_id: info.vendor,
            device_id: info.device,
            limits: LimitsReport {
                max_buffer_size: l.max_buffer_size,
                max_storage_buffer_binding_size: l.max_storage_buffer_binding_size,
                max_compute_workgroup_size_x: l.max_compute_workgroup_size_x,
                max_compute_invocations_per_workgroup: l.max_compute_invocations_per_workgroup,
                max_compute_workgroups_per_dimension: l.max_compute_workgroups_per_dimension,
                max_compute_workgroup_storage_size: l.max_compute_workgroup_storage_size,
                max_storage_buffers_per_shader_stage: l.max_storage_buffers_per_shader_stage,
                max_texture_dimension_2d: l.max_texture_dimension_2d,
            },
            optional_features,
        }
    }
}

/// Every adapter visible to wgpu on this machine (for `aether info`).
pub fn enumerate_adapters() -> Vec<AdapterReport> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .iter()
        .map(AdapterReport::from_adapter)
        .collect()
}

/// A selected adapter with an open device and queue.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub report: AdapterReport,
    /// `TIMESTAMP_QUERY` was supported and enabled on the device.
    pub timestamps: bool,
}

impl GpuContext {
    /// Headless context on the high-performance adapter.
    pub fn new() -> Result<Arc<Self>, GpuError> {
        Self::with_surface(
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()),
            None,
        )
    }

    /// Context that can present to `surface` (for the viewer).
    pub fn with_surface(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Arc<Self>, GpuError> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
            ..Default::default()
        }))
        .map_err(|e| GpuError::NoAdapter(e.to_string()))?;

        let report = AdapterReport::from_adapter(&adapter);
        tracing::info!(
            adapter = %report.name,
            backend = %report.backend,
            max_buffer = report.limits.max_buffer_size,
            "selected GPU adapter"
        );

        // Request only TIMESTAMP_QUERY, and only when the adapter has it.
        // Pass-boundary timestamps need nothing beyond this feature.
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("aether-device"),
            required_features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| GpuError::Device(e.to_string()))?;

        Ok(Arc::new(Self {
            instance,
            adapter,
            device,
            queue,
            report,
            timestamps,
        }))
    }

    /// Run `f`, converting any wgpu validation error raised inside into `Err`.
    pub fn validated<T>(&self, f: impl FnOnce() -> T) -> Result<T, GpuError> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let out = f();
        match pollster::block_on(scope.pop()) {
            Some(e) => Err(GpuError::Validation(e.to_string())),
            None => Ok(out),
        }
    }

    /// Block until all submitted work has finished.
    pub fn wait_idle(&self) -> Result<(), GpuError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| GpuError::Readback(e.to_string()))
    }

    /// Copy `size` bytes of `src` back to the host (blocking).
    pub fn read_buffer(&self, src: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, GpuError> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        map_read(self, &staging, size)
    }
}

/// Map a staging buffer for reading and copy its contents out.
pub(crate) fn map_read(
    ctx: &GpuContext,
    staging: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, GpuError> {
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..size)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    ctx.wait_idle()?;
    rx.recv()
        .map_err(|e| GpuError::Readback(e.to_string()))?
        .map_err(|e| GpuError::Readback(e.to_string()))?;
    let data = staging
        .slice(..size)
        .get_mapped_range()
        .map_err(|e| GpuError::Readback(e.to_string()))?
        .to_vec();
    staging.unmap();
    Ok(data)
}
