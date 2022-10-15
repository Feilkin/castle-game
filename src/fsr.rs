//! AMD FidelityFX Super Resolution 1.0 implementation
use crate::widgets::Widget;
use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::camera::{RenderTarget, ScalingMode, Viewport};
use bevy::render::mesh::MeshVertexBufferLayout;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, RenderPipelineDescriptor, ShaderRef, ShaderType,
    SpecializedMeshPipelineError, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages,
};
use bevy::render::texture::BevyDefault;
use bevy::render::view::RenderLayers;
use bevy::sprite::{Material2d, Material2dKey, Material2dPlugin, MaterialMesh2dBundle};
use bevy_egui::egui::{Slider, Ui};
use bevy_egui::{egui, EguiContext};
use std::any::Any;

pub struct FsrPlugin;

/// Resources needed for low-resolution rendering.
pub struct LowResTexture {
    /// Handle to the low-resolution frame buffer texture.
    pub image: Handle<Image>,
    /// Handle to the quad that covers the region of the low-resolution texture that should be used.
    pub quad: Handle<Mesh>,
    pub size: (u32, u32),
}

#[derive(ShaderType, Clone)]
pub struct FsrSettings {
    /// How much the low-resolution image will be scaled down from render resolution. 0.50 render
    /// scaling means the dimensions of the low-resolution image will be half of the full viewport
    /// size.
    pub render_scaling: f32,
    /// Amount of sharpening to be performed after FSR up-scaling.
    pub sharpening_amount: f32,
    // TODO: film grain
}

pub struct UpscaleLayer(pub u8);
struct SharpenTexture(Handle<Image>);
struct UpscaleEntity(Entity);
struct SharpenEntity(Entity);
struct CameraEntity(Entity);

/// FSR 1.0 Up-scaling Material
#[derive(AsBindGroup, TypeUuid, Clone)]
#[uuid = "145b7537-9daa-4307-9289-3b01260562fb"]
pub struct UpscaleMaterial {
    #[texture(0)]
    #[sampler(1)]
    low_res: Handle<Image>,
    #[uniform(2)]
    settings: FsrSettings,
}

/// FSR 1.0 Sharpening Material
#[derive(AsBindGroup, TypeUuid, Clone)]
#[uuid = "ef03e778-70dd-4b77-a315-a79393d67e47"]
pub struct SharpeningMaterial {
    #[texture(0)]
    #[sampler(1)]
    low_res: Handle<Image>,
    #[uniform(2)]
    settings: FsrSettings,
}

impl Material2d for UpscaleMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/fsr1.wgsl".into()
    }
}

impl Material2d for SharpeningMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/fsr1.wgsl".into()
    }

    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayout,
        _key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.fragment.as_mut().unwrap().entry_point = "sharpen".into();
        Ok(())
    }
}

impl Plugin for FsrPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FsrSettings::default())
            .insert_resource(UpscaleLayer(2))
            .add_plugin(Material2dPlugin::<UpscaleMaterial>::default())
            .add_plugin(Material2dPlugin::<SharpeningMaterial>::default())
            .add_startup_system_to_stage(StartupStage::PreStartup, setup_fsr)
            .add_system(update_fsr_settings)
            .add_system(draw_settings::<FsrSettings>);
    }
}

impl Default for FsrSettings {
    fn default() -> Self {
        FsrSettings {
            render_scaling: 0.75,
            sharpening_amount: 0.0,
        }
    }
}

impl Widget for FsrSettings {
    fn draw(&mut self, ui: &mut Ui) {
        ui.add(Slider::new(&mut self.render_scaling, 0.01..=1.0).text("Render scale"));
    }
}

/// Set up textures needed for FSR up-scaling.
pub fn setup_fsr(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut upscale_materials: ResMut<Assets<UpscaleMaterial>>,
    mut sharpening_materials: ResMut<Assets<SharpeningMaterial>>,
    settings: Res<FsrSettings>,
    upscale_layer: Res<UpscaleLayer>,
    windows: Res<Windows>,
) {
    let window = windows.primary();
    let full_res_size = Extent3d {
        width: window.physical_width(),
        height: window.physical_height(),
        ..default()
    };

    let full_res_quad = meshes.add(Mesh::from(shape::Quad::new(Vec2::new(
        full_res_size.width as f32,
        full_res_size.height as f32,
    ))));
    // create low-resolution target texture
    let mut low_res_image = Image::new_fill(
        full_res_size,
        TextureDimension::D2,
        &[255, 255, 255, 255],
        TextureFormat::bevy_default(),
    );
    low_res_image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
    let low_res_image_handle = images.add(low_res_image);

    let quad = meshes.add(Mesh::from(shape::Quad::new(Vec2::new(2., 2.))));

    commands.insert_resource(LowResTexture {
        image: low_res_image_handle.clone(),
        quad: quad.clone(),
        size: (
            (full_res_size.width as f32 * settings.render_scaling).floor() as u32,
            (full_res_size.height as f32 * settings.render_scaling).floor() as u32,
        ),
    });

    // create sharpening target texture
    let mut sharpen_image = Image::new_fill(
        full_res_size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::bevy_default(),
    );
    sharpen_image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT;
    let sharpen_image_handle = images.add(sharpen_image);

    commands.insert_resource(SharpenTexture(sharpen_image_handle.clone()));

    // create upscale material
    let upscale_material = upscale_materials.add(UpscaleMaterial {
        low_res: low_res_image_handle.clone(),
        settings: settings.clone(),
    });

    // create material that draws the low-resolution texture to sharpening texture using up-scaling
    let upscale_layer = RenderLayers::layer(upscale_layer.0);
    let upscale_entity = commands
        .spawn_bundle(MaterialMesh2dBundle {
            mesh: full_res_quad.clone().into(),
            material: upscale_material,
            transform: Transform {
                translation: Vec3::new(0.0, 0.0, 1.5),
                ..default()
            },
            ..default()
        })
        .insert(upscale_layer)
        .id();
    commands.insert_resource(UpscaleEntity(upscale_entity));

    // create camera that renders the upscale material to sharpening texture
    commands
        .spawn_bundle(Camera2dBundle {
            camera: Camera {
                priority: 1,
                target: RenderTarget::Image(sharpen_image_handle.clone()),
                ..default()
            },
            camera_2d: Camera2d {
                clear_color: ClearColorConfig::None,
            },
            ..Camera2dBundle::default()
        })
        .insert(UiCameraConfig { show_ui: false })
        .insert(upscale_layer);

    // create sharpening material
    let sharpening_material = sharpening_materials.add(SharpeningMaterial {
        low_res: sharpen_image_handle,
        settings: settings.clone(),
    });

    let sharpening_entity = commands
        .spawn_bundle(MaterialMesh2dBundle {
            mesh: full_res_quad.into(),
            material: sharpening_material,
            transform: Transform::from_translation(Vec3::new(0., 0., 1.5)),
            ..default()
        })
        .id();

    commands.insert_resource(SharpenEntity(sharpening_entity));

    let mut projection = OrthographicProjection::default();
    projection.scaling_mode = ScalingMode::None;

    let camera_entity = commands
        .spawn_bundle(Camera2dBundle {
            camera: Camera {
                // renders after the first main camera which has default value: 0.
                priority: 0,
                target: RenderTarget::Image(low_res_image_handle.clone()),
                viewport: Some(Viewport {
                    physical_position: Default::default(),
                    physical_size: UVec2::new(
                        (full_res_size.width as f32 * settings.render_scaling) as u32,
                        (full_res_size.height as f32 * settings.render_scaling) as u32,
                    ),
                    depth: 0.0..1.0,
                }),
                ..default()
            },
            projection,
            ..Camera2dBundle::default()
        })
        .insert(UiCameraConfig { show_ui: false })
        .insert(RenderLayers::layer(1))
        .id();
    commands.insert_resource(CameraEntity(camera_entity));
}

fn update_fsr_settings(
    settings: Res<FsrSettings>,
    upscale_entity: Res<UpscaleEntity>,
    camera_entity: Res<CameraEntity>,
    mut upscale_query: Query<&Handle<UpscaleMaterial>>,
    mut upscale_mats: ResMut<Assets<UpscaleMaterial>>,
    mut camera_query: Query<&mut Camera>,
    windows: Res<Windows>,
) {
    if settings.is_changed() {
        let window = windows.primary();
        let full_res_size = Extent3d {
            width: window.physical_width(),
            height: window.physical_height(),
            ..default()
        };

        let mat = upscale_query.get(upscale_entity.0).unwrap();
        let mat = upscale_mats.get_mut(mat).unwrap();
        mat.settings = settings.clone();

        let mut cam = camera_query.get_mut(camera_entity.0).unwrap();
        cam.viewport = Some(Viewport {
            physical_position: Default::default(),
            physical_size: UVec2::new(
                (full_res_size.width as f32 * settings.render_scaling) as u32,
                (full_res_size.height as f32 * settings.render_scaling) as u32,
            ),
            depth: 0.0..1.0,
        });
    }
}

fn draw_settings<T>(mut settings: ResMut<T>, mut egui_context: ResMut<EguiContext>)
where
    T: Widget + Any + Send + Sync,
{
    egui::Window::new(std::any::type_name::<T>()).show(egui_context.ctx_mut(), |ui| {
        settings.draw(ui);
    });
}
