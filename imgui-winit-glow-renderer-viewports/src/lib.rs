use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    num::NonZeroU32,
    ptr::null_mut,
    rc::Rc,
    slice,
};

use glow::HasContext;
use glutin::{
    config::ConfigTemplateBuilder,
    context::{
        ContextAttributesBuilder, NotCurrentContext, NotCurrentGlContext, PossiblyCurrentGlContext,
    },
    display::GetGlDisplay,
    prelude::GlDisplay,
    surface::{GlSurface, Surface, SurfaceAttributesBuilder, WindowSurface},
};
use glutin_winit::DisplayBuilder;
use imgui::{BackendFlags, ConfigFlags, Id, Io, Key, MouseButton, ViewportFlags};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use thiserror::Error;
use winit::{
    dpi::{LogicalPosition, LogicalSize},
    event::{ElementState, TouchPhase},
    event_loop::EventLoopWindowTarget,
    keyboard::{Key as WinitKey, KeyLocation, NamedKey},
    platform::{
        modifier_supplement::KeyEventExtModifierSupplement, windows::WindowBuilderExtWindows,
    },
    window::{CursorIcon, Window, WindowBuilder, WindowLevel},
};

const VERTEX_SHADER: &str = include_str!("vertex_shader.glsl");
const FRAGMENT_SHADER: &str = include_str!("fragment_shader.glsl");

#[derive(Debug, Error)]
pub enum RendererError {
    #[error("OpenGL shader creation failed: {0}")]
    GlShaderCreationFailed(String),
    #[error("OpenGL program creation failed: {0}")]
    GlProgramCreationFailed(String),
    #[error("OpenGL texture creation failed: {0}")]
    GlTextureCreationFailed(String),
    #[error("OpenGL buffer creation failed: {0}")]
    GlBufferCreationFailed(String),
    #[error("OpenGL vertex array creation failed: {0}")]
    GlVertexArrayCreationFailed(String),
    #[error("Failed to create viewport window")]
    WindowCreationFailed,
    #[error("Failed to create viewport window context")]
    WindowContextCreationFailed,
    #[error("Failed to create viewport window surface")]
    WindowSurfaceCreationFailed,
    #[error("Failed to make viewport context current")]
    MakeCurrentFailed,
    #[error("Failed to make swap buffers on surface")]
    SwapBuffersFailed,
}

#[derive(Debug)]
enum ViewportEvent {
    Create(Id),
    Destroy(Id),
    SetPos(Id, [f32; 2]),
    SetSize(Id, [f32; 2]),
    SetVisible(Id),
    SetFocus(Id),
    SetTitle(Id, String),
}

type WindowBuilderCallback = dyn Fn(&imgui::Viewport) -> WindowBuilder;

pub struct Renderer {
    gl_objects: GlObjects,
    glutin_config: Option<glutin::config::Config>,
    window_builder: Option<Box<WindowBuilderCallback>>,
    /// The tuple members have to stay in exactly this order
    /// to ensure that surface, context and window are dropped in this order
    extra_windows: HashMap<
        Id,
        (
            GlObjects,
            Surface<WindowSurface>,
            Option<NotCurrentContext>,
            Window,
        ),
    >,
    event_queue: Rc<RefCell<VecDeque<ViewportEvent>>>,
    font_width: u32,
    font_height: u32,
    font_data: Vec<u8>,
    last_cursor: CursorIcon,
}

#[derive(Debug)]
struct GlObjects {
    program: glow::Program,
    font_atlas: glow::Texture,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    ibo: glow::Buffer,
}

impl GlObjects {
    pub fn new(
        font_width: u32,
        font_height: u32,
        font_data: &[u8],
        glow: &glow::Context,
    ) -> Result<Self, RendererError> {
        let program = unsafe {
            let vertex_shader = glow
                .create_shader(glow::VERTEX_SHADER)
                .map_err(RendererError::GlShaderCreationFailed)?;
            glow.shader_source(vertex_shader, VERTEX_SHADER);
            glow.compile_shader(vertex_shader);
            assert!(
                glow.get_shader_compile_status(vertex_shader),
                "Vertex Shader contains error"
            );

            let fragment_shader = glow
                .create_shader(glow::FRAGMENT_SHADER)
                .map_err(RendererError::GlShaderCreationFailed)?;
            glow.shader_source(fragment_shader, FRAGMENT_SHADER);
            glow.compile_shader(fragment_shader);
            assert!(
                glow.get_shader_compile_status(fragment_shader),
                "Fragment Shader contains error"
            );

            let program = glow
                .create_program()
                .map_err(RendererError::GlProgramCreationFailed)?;
            glow.attach_shader(program, vertex_shader);
            glow.attach_shader(program, fragment_shader);
            glow.link_program(program);
            assert!(
                glow.get_program_link_status(program),
                "Program contains error"
            );

            glow.delete_shader(vertex_shader);
            glow.delete_shader(fragment_shader);

            program
        };

        let font_atlas = unsafe {
            let tex = glow
                .create_texture()
                .map_err(RendererError::GlTextureCreationFailed)?;
            glow.bind_texture(glow::TEXTURE_2D, Some(tex));
            glow.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            glow.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            glow.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            glow.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            glow.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                font_width as i32,
                font_height as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                Some(font_data),
            );

            tex
        };

        let vbo = unsafe {
            glow.create_buffer()
                .map_err(RendererError::GlBufferCreationFailed)?
        };
        let ibo = unsafe {
            glow.create_buffer()
                .map_err(RendererError::GlBufferCreationFailed)?
        };

        let vao = unsafe {
            let vao = glow
                .create_vertex_array()
                .map_err(RendererError::GlVertexArrayCreationFailed)?;

            glow.bind_vertex_array(Some(vao));
            glow.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            glow.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            glow.enable_vertex_attrib_array(0);
            glow.enable_vertex_attrib_array(1);
            glow.enable_vertex_attrib_array(2);
            glow.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 20, 0);
            glow.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 20, 8);
            glow.vertex_attrib_pointer_f32(2, 4, glow::UNSIGNED_BYTE, true, 20, 16);
            glow.bind_vertex_array(None);

            vao
        };

        Ok(Self {
            program,
            font_atlas,
            vao,
            vbo,
            ibo,
        })
    }
}

#[derive(Debug)]
struct GlStateBackup {
    viewport: [i32; 4],
    blend_enabled: bool,
    blend_func_src: i32,
    blend_func_dst: i32,
    scissor_enabled: bool,
    scissor: [i32; 4],
    vao: u32,
    vbo: u32,
    ibo: u32,
    active_texture: u32,
    texture: u32,
    program: u32,
}

fn to_native_gl<T>(handle: u32, constructor: fn(NonZeroU32) -> T) -> Option<T> {
    if handle != 0 {
        Some(constructor(NonZeroU32::new(handle).unwrap()))
    } else {
        None
    }
}

impl GlStateBackup {
    fn backup(context: &glow::Context) -> Self {
        unsafe {
            let mut viewport = [0; 4];
            context.get_parameter_i32_slice(glow::VIEWPORT, &mut viewport);

            let blend_enabled = context.is_enabled(glow::BLEND);
            let blend_func_src = context.get_parameter_i32(glow::BLEND_SRC);
            let blend_func_dst = context.get_parameter_i32(glow::BLEND_DST);

            let scissor_enabled = context.is_enabled(glow::SCISSOR_TEST);
            let mut scissor = [0; 4];
            context.get_parameter_i32_slice(glow::SCISSOR_BOX, &mut scissor);

            let vao = context.get_parameter_i32(glow::VERTEX_ARRAY_BINDING) as _;
            let vbo = context.get_parameter_i32(glow::ARRAY_BUFFER_BINDING) as _;
            let ibo = context.get_parameter_i32(glow::ELEMENT_ARRAY_BUFFER_BINDING) as _;

            let active_texture = context.get_parameter_i32(glow::ACTIVE_TEXTURE) as _;
            context.active_texture(0);
            let texture = context.get_parameter_i32(glow::TEXTURE_BINDING_2D) as _;

            let program = context.get_parameter_i32(glow::CURRENT_PROGRAM) as _;

            Self {
                viewport,
                blend_enabled,
                blend_func_src,
                blend_func_dst,
                scissor_enabled,
                scissor,
                vao,
                vbo,
                ibo,
                active_texture,
                texture,
                program,
            }
        }
    }

    fn restore(&self, context: &glow::Context) {
        unsafe {
            context.viewport(
                self.viewport[0],
                self.viewport[1],
                self.viewport[2],
                self.viewport[3],
            );

            Self::enable(context, glow::BLEND, self.blend_enabled);
            context.blend_func(self.blend_func_src as _, self.blend_func_dst as _);

            Self::enable(context, glow::SCISSOR_TEST, self.scissor_enabled);
            context.scissor(
                self.scissor[0],
                self.scissor[1],
                self.scissor[2],
                self.scissor[3],
            );

            context.bind_vertex_array(to_native_gl(self.vao, glow::NativeVertexArray));

            context.bind_buffer(
                glow::ARRAY_BUFFER,
                to_native_gl(self.vbo, glow::NativeBuffer),
            );
            context.bind_buffer(
                glow::ELEMENT_ARRAY_BUFFER,
                to_native_gl(self.ibo, glow::NativeBuffer),
            );

            context.bind_texture(
                glow::TEXTURE_2D,
                to_native_gl(self.texture, glow::NativeTexture),
            );
            context.active_texture(self.active_texture);

            context.use_program(to_native_gl(self.program, glow::NativeProgram));
        }
    }

    fn enable(context: &glow::Context, feature: u32, value: bool) {
        unsafe {
            if value {
                context.enable(feature);
            } else {
                context.disable(feature);
            }
        }
    }
}

impl Renderer {
    pub fn new(
        imgui: &mut imgui::Context,
        main_window: &Window,
        gl_context: &glow::Context,
        window_builder: Option<Box<WindowBuilderCallback>>,
    ) -> Result<Self, RendererError> {
        let dpi_scale = main_window.scale_factor();
        let io = imgui.io_mut();

        // there is no good way to handle viewports on wayland,
        // so we disable them
        match main_window.raw_window_handle() {
            RawWindowHandle::Wayland(_) => {}
            _ => {
                io.backend_flags
                    .insert(BackendFlags::PLATFORM_HAS_VIEWPORTS);
                io.backend_flags
                    .insert(BackendFlags::RENDERER_HAS_VIEWPORTS);
            }
        }

        io.backend_flags.insert(BackendFlags::HAS_MOUSE_CURSORS);
        io.backend_flags.insert(BackendFlags::HAS_SET_MOUSE_POS);

        io.backend_flags
            .insert(BackendFlags::RENDERER_HAS_VTX_OFFSET);

        let window_size = main_window.inner_size().cast::<f32>();
        let logical_size = window_size.to_logical::<f32>(dpi_scale);
        io.display_size = logical_size.into();
        io.display_framebuffer_scale = [dpi_scale as f32, dpi_scale as f32];

        let viewport = imgui.main_viewport_mut();

        let main_pos = main_window
            .inner_position()
            .unwrap_or_default()
            .cast::<f32>();
        let logical_pos = main_pos.to_logical::<f32>(dpi_scale);

        viewport.pos = logical_pos.into();
        viewport.work_pos = logical_pos.into();
        viewport.size = logical_size.into();
        viewport.work_size = logical_size.into();
        viewport.dpi_scale = dpi_scale as f32;
        viewport.platform_user_data = Box::into_raw(Box::new(ViewportData {
            pos: logical_pos.into(),
            size: logical_size.into(),
            focus: true,
            minimized: false,
        }))
        .cast();

        let mut monitors = Vec::new();
        for monitor in main_window.available_monitors() {
            monitors.push(imgui::PlatformMonitor {
                main_pos: [monitor.position().x as f32, monitor.position().y as f32],
                main_size: [monitor.size().width as f32, monitor.size().height as f32],
                work_pos: [monitor.position().x as f32, monitor.position().y as f32],
                work_size: [monitor.size().width as f32, monitor.size().height as f32],
                dpi_scale: monitor.scale_factor() as f32,
            });
        }
        imgui
            .platform_io_mut()
            .monitors
            .replace_from_slice(&monitors);

        imgui.set_platform_name(Some(format!(
            "imgui-winit-glow-renderer-viewports {}",
            env!("CARGO_PKG_VERSION")
        )));
        imgui.set_renderer_name(Some(format!(
            "imgui-winit-glow-renderer-viewports {}",
            env!("CARGO_PKG_VERSION")
        )));

        let event_queue = Rc::new(RefCell::new(VecDeque::new()));

        imgui.set_platform_backend(PlatformBackend {
            event_queue: event_queue.clone(),
        });
        imgui.set_renderer_backend(RendererBackend {});

        let font_atlas = imgui.fonts().build_rgba32_texture();
        let gl_objects = GlObjects::new(
            font_atlas.width,
            font_atlas.height,
            font_atlas.data,
            gl_context,
        )?;

        Ok(Self {
            gl_objects,
            glutin_config: None,
            window_builder,
            extra_windows: HashMap::new(),
            event_queue,
            font_width: font_atlas.width,
            font_height: font_atlas.height,
            font_data: font_atlas.data.to_vec(),
            last_cursor: CursorIcon::Default,
        })
    }

    pub fn handle_event<T>(
        &mut self,
        imgui: &mut imgui::Context,
        main_window: &Window,
        event: &winit::event::Event<T>,
    ) {
        if let winit::event::Event::WindowEvent {
            window_id,
            ref event,
        } = *event
        {
            let (window, viewport) = if window_id == main_window.id() {
                (main_window, imgui.main_viewport_mut())
            } else if let Some((id, wnd)) =
                self.extra_windows.iter().find_map(|(id, (_, _, _, wnd))| {
                    if wnd.id() == window_id {
                        Some((*id, wnd))
                    } else {
                        None
                    }
                })
            {
                if let Some(viewport) = imgui.viewport_by_id_mut(id) {
                    (wnd, viewport)
                } else {
                    return;
                }
            } else {
                return;
            };

            match *event {
                winit::event::WindowEvent::Resized(new_size) => {
                    let logical_size = new_size.to_logical::<f32>(viewport.dpi_scale.into());
                    unsafe {
                        (*(viewport.platform_user_data.cast::<ViewportData>())).size =
                            logical_size.into();
                    }

                    viewport.platform_request_resize = true;

                    if window_id == main_window.id() {
                        imgui.io_mut().display_size = logical_size.into();
                    }
                }
                winit::event::WindowEvent::Moved(_) => unsafe {
                    let new_pos = window.inner_position().unwrap().cast::<f32>();
                    let logical_pos = new_pos.to_logical::<f32>(viewport.dpi_scale.into());
                    (*(viewport.platform_user_data.cast::<ViewportData>())).pos =
                        logical_pos.into();

                    viewport.platform_request_move = true;
                },
                winit::event::WindowEvent::CloseRequested if window_id != main_window.id() => {
                    viewport.platform_request_close = true;
                }
                winit::event::WindowEvent::Focused(f) => unsafe {
                    (*(viewport.platform_user_data.cast::<ViewportData>())).focus = f;
                },
                winit::event::WindowEvent::KeyboardInput { ref event, .. } => {
                    if let Some(txt) = &event.text {
                        for ch in txt.chars() {
                            imgui.io_mut().add_input_character(ch);
                        }
                    }

                    let key = event.key_without_modifiers();

                    let pressed = event.state == ElementState::Pressed;

                    // We map both left and right ctrl to `ModCtrl`, etc.
                    // imgui is told both "left control is pressed" and
                    // "consider the control key is pressed". Allows
                    // applications to use either general "ctrl" or a
                    // specific key. Same applies to other modifiers.
                    // https://github.com/ocornut/imgui/issues/5047
                    handle_key_modifier(imgui.io_mut(), &key, pressed);

                    // Add main key event
                    if let Some(key) = to_imgui_key(key, event.location) {
                        imgui.io_mut().add_key_event(key, pressed);
                    }
                }
                winit::event::WindowEvent::ModifiersChanged(modifiers) => {
                    let state = modifiers.state();

                    imgui
                        .io_mut()
                        .add_key_event(Key::ModShift, state.shift_key());
                    imgui
                        .io_mut()
                        .add_key_event(Key::ModCtrl, state.control_key());
                    imgui.io_mut().add_key_event(Key::ModAlt, state.alt_key());
                    imgui
                        .io_mut()
                        .add_key_event(Key::ModSuper, state.super_key());
                }
                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    let logical_pos = position.to_logical::<f32>(viewport.dpi_scale.into());
                    let window_pos = window.inner_position().unwrap_or_default().cast::<f32>();
                    let window_logical_pos =
                        window_pos.to_logical::<f32>(viewport.dpi_scale.into());
                    if imgui
                        .io()
                        .config_flags
                        .contains(ConfigFlags::VIEWPORTS_ENABLE)
                    {
                        imgui.io_mut().add_mouse_pos_event([
                            logical_pos.x + window_logical_pos.x,
                            logical_pos.y + window_logical_pos.y,
                        ]);
                    } else {
                        imgui
                            .io_mut()
                            .add_mouse_pos_event([position.x as f32, position.y as f32]);
                    }
                }
                winit::event::WindowEvent::MouseWheel {
                    delta,
                    phase: TouchPhase::Moved,
                    ..
                } => match delta {
                    winit::event::MouseScrollDelta::LineDelta(h, v) => {
                        imgui.io_mut().add_mouse_wheel_event([h, v]);
                    }
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        let h = if pos.x > 0.0 {
                            1.0
                        } else if pos.x < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                        let v = if pos.y > 0.0 {
                            1.0
                        } else if pos.y < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                        imgui.io_mut().add_mouse_wheel_event([h, v]);
                    }
                },
                winit::event::WindowEvent::MouseInput { state, button, .. } => {
                    let state = state == ElementState::Pressed;

                    if let Some(button) = to_imgui_mouse_button(button) {
                        imgui.io_mut().add_mouse_button_event(button, state);
                    }
                }
                winit::event::WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    viewport.dpi_scale = scale_factor as f32;
                    viewport.platform_request_resize = true;
                    if window_id == main_window.id() {
                        imgui.io_mut().display_framebuffer_scale =
                            [scale_factor as f32, scale_factor as f32];
                    }
                }
                _ => {}
            }
        }
    }

    pub fn update_viewports<T>(
        &mut self,
        imgui: &mut imgui::Context,
        window_target: &EventLoopWindowTarget<T>,
        glow: &glow::Context,
    ) -> Result<(), RendererError> {
        loop {
            let event = self.event_queue.borrow_mut().pop_front();
            let event = if let Some(event) = event {
                event
            } else {
                break;
            };

            match event {
                ViewportEvent::Create(id) => {
                    if let Some(viewport) = imgui.viewport_by_id_mut(id) {
                        let extra_window =
                            self.create_extra_window(viewport, window_target, glow)?;
                        self.extra_windows.insert(id, extra_window);
                    }
                }
                ViewportEvent::Destroy(id) => {
                    self.extra_windows.remove(&id);
                }
                ViewportEvent::SetPos(id, pos) => {
                    if let Some(viewport) = imgui.viewport_by_id_mut(id) {
                        if let Some((_, _, _, wnd)) = self.extra_windows.get(&id) {
                            let logical_pos = LogicalPosition::new(pos[0], pos[1]);
                            let physical_pos =
                                logical_pos.to_physical::<f32>(viewport.dpi_scale.into());
                            wnd.set_outer_position(physical_pos);
                        }
                    }
                }
                ViewportEvent::SetSize(id, size) => {
                    if let Some(viewport) = imgui.viewport_by_id_mut(id) {
                        if let Some((_, _, _, wnd)) = self.extra_windows.get(&id) {
                            let logical_size = LogicalSize::new(size[0], size[1]);
                            let physical_size =
                                logical_size.to_physical::<f32>(viewport.dpi_scale.into());
                            _ = wnd.request_inner_size(physical_size);
                        }
                    }
                }
                ViewportEvent::SetVisible(id) => {
                    if let Some((_, _, _, wnd)) = self.extra_windows.get(&id) {
                        wnd.set_visible(true);
                    }
                }
                ViewportEvent::SetFocus(id) => {
                    if let Some((_, _, _, wnd)) = self.extra_windows.get(&id) {
                        wnd.focus_window();
                    }
                }
                ViewportEvent::SetTitle(id, title) => {
                    if let Some((_, _, _, wnd)) = self.extra_windows.get(&id) {
                        wnd.set_title(&title);
                    }
                }
            }
        }

        Ok(())
    }

    fn to_winit_cursor(cursor: imgui::MouseCursor) -> winit::window::CursorIcon {
        match cursor {
            imgui::MouseCursor::Arrow => winit::window::CursorIcon::Default,
            imgui::MouseCursor::TextInput => winit::window::CursorIcon::Text,
            imgui::MouseCursor::ResizeAll => winit::window::CursorIcon::Move,
            imgui::MouseCursor::ResizeNS => winit::window::CursorIcon::NsResize,
            imgui::MouseCursor::ResizeEW => winit::window::CursorIcon::EwResize,
            imgui::MouseCursor::ResizeNESW => winit::window::CursorIcon::NeswResize,
            imgui::MouseCursor::ResizeNWSE => winit::window::CursorIcon::NwseResize,
            imgui::MouseCursor::Hand => winit::window::CursorIcon::Grab,
            imgui::MouseCursor::NotAllowed => winit::window::CursorIcon::NotAllowed,
        }
    }

    pub fn prepare_render(&mut self, imgui: &mut imgui::Context, main_window: &Window) {
        if let Some(cursor) = imgui.mouse_cursor() {
            let cursor = Self::to_winit_cursor(cursor);

            if self.last_cursor != cursor {
                main_window.set_cursor_icon(cursor);

                for (_, _, _, wnd) in self.extra_windows.values() {
                    wnd.set_cursor_icon(cursor);
                }

                self.last_cursor = cursor;
            }
        }
    }

    fn create_extra_window<T>(
        &mut self,
        viewport: &mut imgui::Viewport,
        window_target: &EventLoopWindowTarget<T>,
        glow: &glow::Context,
    ) -> Result<
        (
            GlObjects,
            Surface<WindowSurface>,
            Option<NotCurrentContext>,
            Window,
        ),
        RendererError,
    > {
        let logical_pos = LogicalPosition::new(viewport.pos[0], viewport.pos[1]);
        let logical_size = LogicalSize::new(viewport.size[0], viewport.size[1]);

        let window_builder = if let Some(window_builder) = &self.window_builder {
            window_builder(viewport)
        } else {
            WindowBuilder::new()
                .with_resizable(true)
                .with_transparent(true)
                .with_skip_taskbar(viewport.flags.contains(ViewportFlags::NO_TASK_BAR_ICON))
                .with_decorations(!viewport.flags.contains(ViewportFlags::NO_DECORATION))
                .with_window_level(if viewport.flags.contains(ViewportFlags::TOP_MOST) {
                    WindowLevel::AlwaysOnTop
                } else {
                    WindowLevel::Normal
                })
        }
        .with_visible(false)
        .with_position(logical_pos.to_physical::<f32>(viewport.dpi_scale.into()))
        .with_inner_size(logical_size.to_physical::<f32>(viewport.dpi_scale.into()));

        let window = if let Some(glutin_config) = &self.glutin_config {
            glutin_winit::finalize_window(window_target, window_builder, glutin_config)
                .map_err(|_| RendererError::WindowCreationFailed)?
        } else {
            let template_builder = ConfigTemplateBuilder::new();

            let (window, cfg) = DisplayBuilder::new()
                .with_window_builder(Some(window_builder))
                .build(window_target, template_builder, |mut configs| {
                    configs.next().unwrap()
                })
                .map_err(|_| RendererError::WindowCreationFailed)?;

            self.glutin_config = Some(cfg);

            window.unwrap()
        };

        if viewport.flags.contains(ViewportFlags::NO_DECORATION) {
            match window.raw_window_handle() {
                RawWindowHandle::Win32(handle) => {
                    // disable window animations for undecorated windows
                    let hwnd = handle.hwnd;
                    #[cfg(windows)]
                    unsafe {
                        use windows::Win32::Foundation::*;
                        use windows::Win32::Graphics::Dwm::*;

                        let value: BOOL = TRUE;
                        DwmSetWindowAttribute(
                            HWND(hwnd as _),
                            DWMWA_TRANSITIONS_FORCEDISABLED,
                            &value as *const BOOL as *const _,
                            std::mem::size_of::<BOOL>() as _,
                        )
                        .unwrap();
                    }
                }
                _ => {}
            };
        }

        let glutin_config = self.glutin_config.as_ref().unwrap();

        let context_attribs =
            ContextAttributesBuilder::new().build(Some(window.raw_window_handle()));
        let context = unsafe {
            glutin_config
                .display()
                .create_context(glutin_config, &context_attribs)
                .map_err(|_| RendererError::WindowContextCreationFailed)?
        };
        let viewport_w = viewport.size[0] * viewport.dpi_scale;
        let viewport_h = viewport.size[1] * viewport.dpi_scale;
        let surface_attribs = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            window.raw_window_handle(),
            NonZeroU32::new(viewport_w as u32).unwrap(),
            NonZeroU32::new(viewport_h as u32).unwrap(),
        );
        let surface = unsafe {
            glutin_config
                .display()
                .create_window_surface(glutin_config, &surface_attribs)
                .map_err(|_| RendererError::WindowSurfaceCreationFailed)?
        };

        let context = context
            .make_current(&surface)
            .map_err(|_| RendererError::MakeCurrentFailed)?;

        surface
            .set_swap_interval(
                &context,
                glutin::surface::SwapInterval::Wait(NonZeroU32::new(1).unwrap()),
            )
            .unwrap();

        let gl_objects = GlObjects::new(self.font_width, self.font_height, &self.font_data, glow)?;

        Ok((
            gl_objects,
            surface,
            Some(context.make_not_current().unwrap()),
            window,
        ))
    }

    pub fn render(
        &mut self,
        main_window: &Window,
        glow: &glow::Context,
        draw_data: &imgui::DrawData,
    ) -> Result<(), RendererError> {
        let backup = GlStateBackup::backup(glow);
        let res = Self::render_window(
            main_window,
            glow,
            draw_data,
            &self.gl_objects,
            draw_data.framebuffer_scale,
        );
        backup.restore(glow);
        res
    }

    pub fn render_viewports(
        &mut self,
        glow: &glow::Context,
        imgui: &mut imgui::Context,
    ) -> Result<(), RendererError> {
        for (id, (gl_objects, surface, context, wnd)) in &mut self.extra_windows {
            if let Some(viewport) = imgui.viewport_by_id(*id) {
                let current_context = context
                    .take()
                    .unwrap()
                    .make_current(surface)
                    .map_err(|_| RendererError::MakeCurrentFailed)?;

                unsafe {
                    glow.disable(glow::SCISSOR_TEST);
                    glow.clear(glow::COLOR_BUFFER_BIT);
                }
                let framebuffer_scale = [viewport.dpi_scale, viewport.dpi_scale];
                Self::render_window(
                    wnd,
                    glow,
                    viewport.draw_data(),
                    gl_objects,
                    framebuffer_scale,
                )?;
                surface
                    .swap_buffers(&current_context)
                    .map_err(|_| RendererError::SwapBuffersFailed)?;

                *context = Some(current_context.make_not_current().unwrap());
            }
        }

        Ok(())
    }

    fn render_window(
        window: &Window,
        glow: &glow::Context,
        draw_data: &imgui::DrawData,
        gl_objects: &GlObjects,
        framebuffer_scale: [f32; 2],
    ) -> Result<(), RendererError> {
        unsafe {
            // draw_data.framebuffer_scale is hardcoded to
            // display_framebuffer_scale, which isn't that useful.
            let viewport_w = draw_data.display_size[0] * framebuffer_scale[0];
            let viewport_h = draw_data.display_size[1] * framebuffer_scale[1];
            glow.viewport(0, 0, viewport_w as i32, viewport_h as i32);

            glow.enable(glow::BLEND);
            glow.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE_MINUS_DST_ALPHA,
                glow::ONE,
            );
            glow.blend_equation_separate(glow::FUNC_ADD, glow::FUNC_ADD);
            glow.enable(glow::SCISSOR_TEST);

            glow.bind_vertex_array(Some(gl_objects.vao));
            glow.bind_buffer(glow::ARRAY_BUFFER, Some(gl_objects.vbo));
            glow.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(gl_objects.ibo));
            glow.active_texture(glow::TEXTURE0);
            glow.bind_texture(glow::TEXTURE_2D, Some(gl_objects.font_atlas));
            glow.use_program(Some(gl_objects.program));

            let left = draw_data.display_pos[0];
            let right = draw_data.display_pos[0] + draw_data.display_size[0];
            let top = draw_data.display_pos[1];
            let bottom = draw_data.display_pos[1] + draw_data.display_size[1];

            let matrix = [
                2.0 / (right - left),
                0.0,
                0.0,
                0.0,
                0.0,
                (2.0 / (top - bottom)),
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                (right + left) / (left - right),
                (top + bottom) / (bottom - top),
                0.0,
                1.0,
            ];

            let loc = glow
                .get_uniform_location(gl_objects.program, "u_Matrix")
                .unwrap();
            glow.uniform_matrix_4_f32_slice(Some(&loc), false, &matrix);

            for list in draw_data.draw_lists() {
                glow.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    slice::from_raw_parts(
                        list.vtx_buffer().as_ptr().cast(),
                        list.vtx_buffer().len() * 20,
                    ),
                    glow::STREAM_DRAW,
                );
                glow.buffer_data_u8_slice(
                    glow::ELEMENT_ARRAY_BUFFER,
                    slice::from_raw_parts(
                        list.idx_buffer().as_ptr().cast(),
                        list.idx_buffer().len() * 2,
                    ),
                    glow::STREAM_DRAW,
                );

                for cmd in list.commands() {
                    if let imgui::DrawCmd::Elements { count, cmd_params } = cmd {
                        let clip_rect = cmd_params.clip_rect;
                        let clip_off = draw_data.display_pos;
                        let clip_scale = framebuffer_scale;

                        let clip_x1 = (clip_rect[0] - clip_off[0]) * clip_scale[0];
                        let clip_y1 = (clip_rect[1] - clip_off[1]) * clip_scale[1];
                        let clip_x2 = (clip_rect[2] - clip_off[0]) * clip_scale[0];
                        let clip_y2 = (clip_rect[3] - clip_off[1]) * clip_scale[1];

                        glow.scissor(
                            clip_x1 as i32,
                            (viewport_h - clip_y2) as i32,
                            (clip_x2 - clip_x1) as i32,
                            (clip_y2 - clip_y1) as i32,
                        );
                        glow.draw_elements_base_vertex(
                            glow::TRIANGLES,
                            count as i32,
                            glow::UNSIGNED_SHORT,
                            (cmd_params.idx_offset * 2) as i32,
                            cmd_params.vtx_offset as i32,
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

struct ViewportData {
    pos: [f32; 2],
    size: [f32; 2],
    focus: bool,
    minimized: bool,
}

struct PlatformBackend {
    event_queue: Rc<RefCell<VecDeque<ViewportEvent>>>,
}

fn handle_key_modifier(io: &mut Io, key: &WinitKey, down: bool) {
    match key {
        WinitKey::Named(NamedKey::Shift) => io.add_key_event(imgui::Key::ModShift, down),
        WinitKey::Named(NamedKey::Control) => io.add_key_event(imgui::Key::ModCtrl, down),
        WinitKey::Named(NamedKey::Alt) => io.add_key_event(imgui::Key::ModAlt, down),
        WinitKey::Named(NamedKey::Super) => io.add_key_event(imgui::Key::ModSuper, down),
        _ => {}
    }
}

impl imgui::PlatformViewportBackend for PlatformBackend {
    fn create_window(&mut self, viewport: &mut imgui::Viewport) {
        viewport.platform_user_data = Box::into_raw(Box::new(ViewportData {
            pos: viewport.pos,
            size: viewport.size,
            focus: false,
            minimized: false,
        }))
        .cast();
        self.event_queue
            .borrow_mut()
            .push_back(ViewportEvent::Create(viewport.id));
    }

    fn destroy_window(&mut self, viewport: &mut imgui::Viewport) {
        unsafe {
            drop(Box::from_raw(
                viewport.platform_user_data.cast::<ViewportData>(),
            ));
        }
        viewport.platform_user_data = null_mut();

        self.event_queue
            .borrow_mut()
            .push_back(ViewportEvent::Destroy(viewport.id));
    }

    fn show_window(&mut self, viewport: &mut imgui::Viewport) {
        self.event_queue
            .borrow_mut()
            .push_back(ViewportEvent::SetVisible(viewport.id));
    }

    fn set_window_pos(&mut self, viewport: &mut imgui::Viewport, pos: [f32; 2]) {
        self.event_queue
            .borrow_mut()
            .push_back(ViewportEvent::SetPos(viewport.id, pos));
    }

    fn get_window_pos(&mut self, viewport: &mut imgui::Viewport) -> [f32; 2] {
        unsafe { (*(viewport.platform_user_data.cast::<ViewportData>())).pos }
    }

    fn set_window_size(&mut self, viewport: &mut imgui::Viewport, size: [f32; 2]) {
        self.event_queue
            .borrow_mut()
            .push_back(ViewportEvent::SetSize(viewport.id, size));
    }

    fn get_window_size(&mut self, viewport: &mut imgui::Viewport) -> [f32; 2] {
        unsafe { (*(viewport.platform_user_data.cast::<ViewportData>())).size }
    }

    fn set_window_focus(&mut self, viewport: &mut imgui::Viewport) {
        self.event_queue
            .borrow_mut()
            .push_back(ViewportEvent::SetFocus(viewport.id));
    }

    fn get_window_focus(&mut self, viewport: &mut imgui::Viewport) -> bool {
        unsafe { (*(viewport.platform_user_data.cast::<ViewportData>())).focus }
    }

    fn get_window_minimized(&mut self, viewport: &mut imgui::Viewport) -> bool {
        unsafe { (*(viewport.platform_user_data.cast::<ViewportData>())).minimized }
    }

    fn set_window_title(&mut self, viewport: &mut imgui::Viewport, title: &str) {
        self.event_queue
            .borrow_mut()
            .push_back(ViewportEvent::SetTitle(viewport.id, title.to_owned()));
    }

    fn set_window_alpha(&mut self, _viewport: &mut imgui::Viewport, _alpha: f32) {}

    fn update_window(&mut self, _viewport: &mut imgui::Viewport) {}

    fn render_window(&mut self, _viewport: &mut imgui::Viewport) {}

    fn swap_buffers(&mut self, _viewport: &mut imgui::Viewport) {}

    fn create_vk_surface(
        &mut self,
        _viewport: &mut imgui::Viewport,
        _instance: u64,
        _out_surface: &mut u64,
    ) -> i32 {
        0
    }
}

struct RendererBackend {}

impl imgui::RendererViewportBackend for RendererBackend {
    fn create_window(&mut self, _viewport: &mut imgui::Viewport) {}

    fn destroy_window(&mut self, _viewport: &mut imgui::Viewport) {}

    fn set_window_size(&mut self, _viewport: &mut imgui::Viewport, _size: [f32; 2]) {}

    fn render_window(&mut self, _viewport: &mut imgui::Viewport) {}

    fn swap_buffers(&mut self, _viewport: &mut imgui::Viewport) {}
}

fn to_imgui_key(key: winit::keyboard::Key, location: KeyLocation) -> Option<Key> {
    match (key.as_ref(), location) {
        (WinitKey::Named(NamedKey::Tab), _) => Some(Key::Tab),
        (WinitKey::Named(NamedKey::ArrowLeft), _) => Some(Key::LeftArrow),
        (WinitKey::Named(NamedKey::ArrowRight), _) => Some(Key::RightArrow),
        (WinitKey::Named(NamedKey::ArrowUp), _) => Some(Key::UpArrow),
        (WinitKey::Named(NamedKey::ArrowDown), _) => Some(Key::DownArrow),
        (WinitKey::Named(NamedKey::PageUp), _) => Some(Key::PageUp),
        (WinitKey::Named(NamedKey::PageDown), _) => Some(Key::PageDown),
        (WinitKey::Named(NamedKey::Home), _) => Some(Key::Home),
        (WinitKey::Named(NamedKey::End), _) => Some(Key::End),
        (WinitKey::Named(NamedKey::Insert), _) => Some(Key::Insert),
        (WinitKey::Named(NamedKey::Delete), _) => Some(Key::Delete),
        (WinitKey::Named(NamedKey::Backspace), _) => Some(Key::Backspace),
        (WinitKey::Named(NamedKey::Space), _) => Some(Key::Space),
        (WinitKey::Named(NamedKey::Enter), KeyLocation::Standard) => Some(Key::Enter),
        (WinitKey::Named(NamedKey::Enter), KeyLocation::Numpad) => Some(Key::KeypadEnter),
        (WinitKey::Named(NamedKey::Escape), _) => Some(Key::Escape),
        (WinitKey::Named(NamedKey::Control), KeyLocation::Left) => Some(Key::LeftCtrl),
        (WinitKey::Named(NamedKey::Control), KeyLocation::Right) => Some(Key::RightCtrl),
        (WinitKey::Named(NamedKey::Shift), KeyLocation::Left) => Some(Key::LeftShift),
        (WinitKey::Named(NamedKey::Shift), KeyLocation::Right) => Some(Key::RightShift),
        (WinitKey::Named(NamedKey::Alt), KeyLocation::Left) => Some(Key::LeftAlt),
        (WinitKey::Named(NamedKey::Alt), KeyLocation::Right) => Some(Key::RightAlt),
        (WinitKey::Named(NamedKey::Super), KeyLocation::Left) => Some(Key::LeftSuper),
        (WinitKey::Named(NamedKey::Super), KeyLocation::Right) => Some(Key::RightSuper),
        (WinitKey::Named(NamedKey::ContextMenu), _) => Some(Key::Menu),
        (WinitKey::Named(NamedKey::F1), _) => Some(Key::F1),
        (WinitKey::Named(NamedKey::F2), _) => Some(Key::F2),
        (WinitKey::Named(NamedKey::F3), _) => Some(Key::F3),
        (WinitKey::Named(NamedKey::F4), _) => Some(Key::F4),
        (WinitKey::Named(NamedKey::F5), _) => Some(Key::F5),
        (WinitKey::Named(NamedKey::F6), _) => Some(Key::F6),
        (WinitKey::Named(NamedKey::F7), _) => Some(Key::F7),
        (WinitKey::Named(NamedKey::F8), _) => Some(Key::F8),
        (WinitKey::Named(NamedKey::F9), _) => Some(Key::F9),
        (WinitKey::Named(NamedKey::F10), _) => Some(Key::F10),
        (WinitKey::Named(NamedKey::F11), _) => Some(Key::F11),
        (WinitKey::Named(NamedKey::F12), _) => Some(Key::F12),
        (WinitKey::Named(NamedKey::CapsLock), _) => Some(Key::CapsLock),
        (WinitKey::Named(NamedKey::ScrollLock), _) => Some(Key::ScrollLock),
        (WinitKey::Named(NamedKey::NumLock), _) => Some(Key::NumLock),
        (WinitKey::Named(NamedKey::PrintScreen), _) => Some(Key::PrintScreen),
        (WinitKey::Named(NamedKey::Pause), _) => Some(Key::Pause),
        (WinitKey::Character("0"), KeyLocation::Standard) => Some(Key::Alpha0),
        (WinitKey::Character("1"), KeyLocation::Standard) => Some(Key::Alpha1),
        (WinitKey::Character("2"), KeyLocation::Standard) => Some(Key::Alpha2),
        (WinitKey::Character("3"), KeyLocation::Standard) => Some(Key::Alpha3),
        (WinitKey::Character("4"), KeyLocation::Standard) => Some(Key::Alpha4),
        (WinitKey::Character("5"), KeyLocation::Standard) => Some(Key::Alpha5),
        (WinitKey::Character("6"), KeyLocation::Standard) => Some(Key::Alpha6),
        (WinitKey::Character("7"), KeyLocation::Standard) => Some(Key::Alpha7),
        (WinitKey::Character("8"), KeyLocation::Standard) => Some(Key::Alpha8),
        (WinitKey::Character("9"), KeyLocation::Standard) => Some(Key::Alpha9),
        (WinitKey::Character("0"), KeyLocation::Numpad) => Some(Key::Keypad0),
        (WinitKey::Character("1"), KeyLocation::Numpad) => Some(Key::Keypad1),
        (WinitKey::Character("2"), KeyLocation::Numpad) => Some(Key::Keypad2),
        (WinitKey::Character("3"), KeyLocation::Numpad) => Some(Key::Keypad3),
        (WinitKey::Character("4"), KeyLocation::Numpad) => Some(Key::Keypad4),
        (WinitKey::Character("5"), KeyLocation::Numpad) => Some(Key::Keypad5),
        (WinitKey::Character("6"), KeyLocation::Numpad) => Some(Key::Keypad6),
        (WinitKey::Character("7"), KeyLocation::Numpad) => Some(Key::Keypad7),
        (WinitKey::Character("8"), KeyLocation::Numpad) => Some(Key::Keypad8),
        (WinitKey::Character("9"), KeyLocation::Numpad) => Some(Key::Keypad9),
        (WinitKey::Character("a"), _) => Some(Key::A),
        (WinitKey::Character("b"), _) => Some(Key::B),
        (WinitKey::Character("c"), _) => Some(Key::C),
        (WinitKey::Character("d"), _) => Some(Key::D),
        (WinitKey::Character("e"), _) => Some(Key::E),
        (WinitKey::Character("f"), _) => Some(Key::F),
        (WinitKey::Character("g"), _) => Some(Key::G),
        (WinitKey::Character("h"), _) => Some(Key::H),
        (WinitKey::Character("i"), _) => Some(Key::I),
        (WinitKey::Character("j"), _) => Some(Key::J),
        (WinitKey::Character("k"), _) => Some(Key::K),
        (WinitKey::Character("l"), _) => Some(Key::L),
        (WinitKey::Character("m"), _) => Some(Key::M),
        (WinitKey::Character("n"), _) => Some(Key::N),
        (WinitKey::Character("o"), _) => Some(Key::O),
        (WinitKey::Character("p"), _) => Some(Key::P),
        (WinitKey::Character("q"), _) => Some(Key::Q),
        (WinitKey::Character("r"), _) => Some(Key::R),
        (WinitKey::Character("s"), _) => Some(Key::S),
        (WinitKey::Character("t"), _) => Some(Key::T),
        (WinitKey::Character("u"), _) => Some(Key::U),
        (WinitKey::Character("v"), _) => Some(Key::V),
        (WinitKey::Character("w"), _) => Some(Key::W),
        (WinitKey::Character("x"), _) => Some(Key::X),
        (WinitKey::Character("y"), _) => Some(Key::Y),
        (WinitKey::Character("z"), _) => Some(Key::Z),
        (WinitKey::Character("'"), _) => Some(Key::Apostrophe),
        (WinitKey::Character(","), KeyLocation::Standard) => Some(Key::Comma),
        (WinitKey::Character("-"), KeyLocation::Standard) => Some(Key::Minus),
        (WinitKey::Character("-"), KeyLocation::Numpad) => Some(Key::KeypadSubtract),
        (WinitKey::Character("."), KeyLocation::Standard) => Some(Key::Period),
        (WinitKey::Character("."), KeyLocation::Numpad) => Some(Key::KeypadDecimal),
        (WinitKey::Character("/"), KeyLocation::Standard) => Some(Key::Slash),
        (WinitKey::Character("/"), KeyLocation::Numpad) => Some(Key::KeypadDivide),
        (WinitKey::Character(";"), _) => Some(Key::Semicolon),
        (WinitKey::Character("="), KeyLocation::Standard) => Some(Key::Equal),
        (WinitKey::Character("="), KeyLocation::Numpad) => Some(Key::KeypadEqual),
        (WinitKey::Character("["), _) => Some(Key::LeftBracket),
        (WinitKey::Character("\\"), _) => Some(Key::Backslash),
        (WinitKey::Character("]"), _) => Some(Key::RightBracket),
        (WinitKey::Character("`"), _) => Some(Key::GraveAccent),
        (WinitKey::Character("*"), KeyLocation::Numpad) => Some(Key::KeypadMultiply),
        (WinitKey::Character("+"), KeyLocation::Numpad) => Some(Key::KeypadAdd),
        _ => None,
    }
}

fn to_imgui_mouse_button(button: winit::event::MouseButton) -> Option<MouseButton> {
    match button {
        winit::event::MouseButton::Left | winit::event::MouseButton::Other(0) => {
            Some(imgui::MouseButton::Left)
        }
        winit::event::MouseButton::Right | winit::event::MouseButton::Other(1) => {
            Some(imgui::MouseButton::Right)
        }
        winit::event::MouseButton::Middle | winit::event::MouseButton::Other(2) => {
            Some(imgui::MouseButton::Middle)
        }
        winit::event::MouseButton::Other(3) => Some(imgui::MouseButton::Extra1),
        winit::event::MouseButton::Other(4) => Some(imgui::MouseButton::Extra2),
        _ => None,
    }
}
