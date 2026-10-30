use std::sync::Arc;
use std::time::{Duration, Instant};

use aether_core::{BackendKind, ExperimentConfig, Workload};
use aether_gpu::{render_to_rgba, GpuContext, GpuWorkload, Renderer, View};
use anyhow::{anyhow, Context, Result};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const HELP: &str = "space pause | +/- speed | R reset | drag pan | scroll zoom | 0 reset view | [ ] point size | S screenshot | Esc quit";

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    ctx: Arc<GpuContext>,
    renderer: Renderer,
    sim: GpuWorkload,
}

struct App {
    cfg: ExperimentConfig,
    gfx: Option<Gfx>,
    error: Option<anyhow::Error>,
    view: View,
    paused: bool,
    /// Simulation steps per rendered frame.
    speed: f32,
    step_accum: f32,
    total_steps: u64,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    frames: u32,
    window_start: Instant,
    fps: f32,
    step_ms: Option<f64>,
    shots: u32,
}

impl App {
    fn new(cfg: ExperimentConfig) -> Self {
        let view = View::for_workload(cfg.workload, cfg.grid, cfg.count, 1.0);
        Self {
            cfg,
            gfx: None,
            error: None,
            view,
            paused: false,
            speed: 1.0,
            step_accum: 0.0,
            total_steps: 0,
            dragging: false,
            last_cursor: None,
            frames: 0,
            window_start: Instant::now(),
            fps: 0.0,
            step_ms: None,
            shots: 0,
        }
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("AetherCompute")
                        .with_inner_size(PhysicalSize::new(1100, 800)),
                )
                .context("creating window")?,
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .context("creating surface")?;
        let ctx = GpuContext::with_surface(instance, Some(&surface))?;

        let caps = surface.get_capabilities(&ctx.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| anyhow!("surface reports no formats"))?;
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: Default::default(),
        };
        surface.configure(&ctx.device, &config);

        let mut cfg = self.cfg.clone();
        cfg.backend = BackendKind::Gpu;
        let sim = GpuWorkload::new(ctx.clone(), cfg)?;
        let renderer = Renderer::new(&ctx, format)?;
        self.view.aspect = config.width as f32 / config.height as f32;
        tracing::info!(adapter = %ctx.report.name, "viewer ready");
        eprintln!("{HELP}");
        self.gfx = Some(Gfx {
            window,
            surface,
            config,
            ctx,
            renderer,
            sim,
        });
        Ok(())
    }

    fn reset_view(&mut self) {
        let aspect = self.view.aspect;
        self.view = View::for_workload(self.cfg.workload, self.cfg.grid, self.cfg.count, aspect);
    }

    fn title(&self) -> String {
        let g = self.gfx.as_ref().unwrap();
        format!(
            "AetherCompute | {} {} {} | {} | {:.0} fps | {:.2} steps/frame{} | step {} {}",
            self.cfg.workload.as_str(),
            self.cfg.workload.complexity(),
            self.cfg.size_label(),
            g.ctx.report.name,
            self.fps,
            self.speed,
            if self.paused { " [paused]" } else { "" },
            self.total_steps,
            self.step_ms
                .map(|m| format!("| {m:.3} ms/step"))
                .unwrap_or_default(),
        )
    }

    fn redraw(&mut self) -> Result<()> {
        let steps = if self.paused {
            0
        } else {
            self.step_accum += self.speed;
            let n = self.step_accum.floor();
            self.step_accum -= n;
            n as u32
        };
        let measure = steps > 0 && self.frames == 0;
        let g = self.gfx.as_mut().unwrap();
        if steps > 0 {
            g.sim.step(steps)?;
            self.total_steps += steps as u64;
        }
        // Once per second, time one isolated, synchronised step for the title.
        if measure {
            g.sim.sync()?;
            let t = Instant::now();
            g.sim.step(1)?;
            g.sim.sync()?;
            self.step_ms = Some(t.elapsed().as_secs_f64() * 1e3);
            self.total_steps += 1;
        }

        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = match g.surface.get_current_texture() {
            Cst::Success(f) | Cst::Suboptimal(f) => f,
            Cst::Outdated | Cst::Lost => {
                g.surface.configure(&g.ctx.device, &g.config);
                g.window.request_redraw();
                return Ok(());
            }
            Cst::Timeout | Cst::Occluded => {
                g.window.request_redraw();
                return Ok(());
            }
            Cst::Validation => return Err(anyhow!("surface validation error")),
        };
        let tv = frame.texture.create_view(&Default::default());
        g.renderer.draw(&g.ctx, &g.sim, &self.view, &tv);
        g.window.pre_present_notify();
        g.ctx.queue.present(frame);

        self.frames += 1;
        let el = self.window_start.elapsed();
        if el >= Duration::from_secs(1) {
            self.fps = self.frames as f32 / el.as_secs_f32();
            self.frames = 0;
            self.window_start = Instant::now();
            let title = self.title();
            self.gfx.as_ref().unwrap().window.set_title(&title);
        }
        self.gfx.as_ref().unwrap().window.request_redraw();
        Ok(())
    }

    fn screenshot(&mut self) -> Result<()> {
        let g = self.gfx.as_mut().unwrap();
        let (w, h) = (g.config.width, g.config.height);
        let mut view = self.view;
        view.aspect = w as f32 / h as f32;
        let px = render_to_rgba(&g.ctx, &g.sim, w, h, &view)?;
        self.shots += 1;
        let path = std::path::PathBuf::from(format!(
            "aether-{}-{}.png",
            self.cfg.workload.as_str(),
            self.shots
        ));
        crate::screenshot::save_png(&path, w, h, &px)?;
        eprintln!("saved {}", path.display());
        Ok(())
    }

    fn key(&mut self, event_loop: &ActiveEventLoop, key: Key) -> Result<()> {
        match key.as_ref() {
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Named(NamedKey::Space) => self.paused = !self.paused,
            Key::Character("+") | Key::Character("=") => self.speed = (self.speed * 2.0).min(64.0),
            Key::Character("-") => self.speed = (self.speed / 2.0).max(0.125),
            Key::Character("r") | Key::Character("R") => {
                let g = self.gfx.as_mut().unwrap();
                g.sim.initialize()?;
                self.total_steps = 0;
                self.step_ms = None;
            }
            Key::Character("0") => self.reset_view(),
            Key::Character("[") => self.view.point_size = (self.view.point_size * 0.8).max(0.0005),
            Key::Character("]") => self.view.point_size = (self.view.point_size * 1.25).min(0.05),
            Key::Character("s") | Key::Character("S") => self.screenshot()?,
            _ => {}
        }
        Ok(())
    }

    fn handle(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) -> Result<()> {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.config.width = size.width.max(1);
                    g.config.height = size.height.max(1);
                    g.surface.configure(&g.ctx.device, &g.config);
                    self.view.aspect = g.config.width as f32 / g.config.height as f32;
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                self.key(event_loop, event.logical_key)?;
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state == ElementState::Pressed;
                if !self.dragging {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    if let (Some((lx, ly)), Some(g)) = (self.last_cursor, self.gfx.as_ref()) {
                        // One screen height spans 2/zoom world units.
                        let k = 2.0 / (self.view.zoom * g.config.height as f32);
                        self.view.center[0] -= (position.x - lx) as f32 * k;
                        self.view.center[1] += (position.y - ly) as f32 * k;
                    }
                }
                self.last_cursor = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.view.zoom = (self.view.zoom * 1.1f32.powf(dy)).clamp(0.05, 200.0);
            }
            WindowEvent::RedrawRequested => self.redraw()?,
            _ => {}
        }
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_none() {
            if let Err(e) = self.init(event_loop) {
                self.error = Some(e);
                event_loop.exit();
                return;
            }
            self.gfx.as_ref().unwrap().window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Err(e) = self.handle(event_loop, event) {
            self.error = Some(e);
            event_loop.exit();
        }
    }
}

/// Open the viewer window and run until it is closed.
pub fn run(cfg: ExperimentConfig) -> Result<()> {
    cfg.validate()?;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(cfg);
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
