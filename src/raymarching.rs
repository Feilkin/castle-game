//! Raymarching stuff
use bevy::core_pipeline::core_2d::Transparent2d;
use bevy::core_pipeline::core_3d::Transparent3d;
use bevy::ecs::system::lifetimeless::{SQuery, SRes};
use bevy::ecs::system::SystemParamItem;
use std::any::Any;
use std::ops::RangeInclusive;

use crate::bake::{HeightMap, VoxelData};
use crate::buildings;
use crate::bvh::BvhBuffer;
use crate::fsr::LowResTexture;
use bevy::prelude::*;
use bevy::render::camera::Projection;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::mesh::MeshVertexBufferLayout;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, EntityRenderCommand, RenderCommandResult, RenderPhase,
    SetItemPipeline, TrackedRenderPass,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, DynamicStorageBuffer,
    PipelineCache, RenderPipelineDescriptor, SamplerBindingType, ShaderStages, ShaderType,
    SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines,
    SpecializedRenderPipeline, SpecializedRenderPipelines, TextureSampleType, TextureViewDimension,
    UniformBuffer,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::view::{RenderLayers, VisibleEntities};
use bevy::render::{Extract, RenderApp, RenderStage};
use bevy::sprite::{
    DrawMesh2d, Mesh2dHandle, Mesh2dPipeline, Mesh2dPipelineKey, Mesh2dUniform, SetMesh2dBindGroup,
    SetMesh2dViewBindGroup,
};
use bevy::utils::FloatOrd;
use bevy_egui::egui::plot::Plot;
use bevy_egui::egui::{Color32, Ui};
use bevy_egui::{egui, EguiContext};

use crate::widgets::{Curve, Widget};

// Raymarching related settings

#[derive(Copy, Clone)]
struct DensityFalloff(f64);

#[derive(ShaderType, Clone, ExtractResource)]
pub struct SkySettings {
    planet_center: Vec3,
    planet_radius: f32,
    atmosphere_radius: f32,
    sun_distance: f32,
    sun_axis: Vec3,
    density_falloff: f32,
    wavelengths: Vec3,
    scatter_strength: f32,
    scatter_coefficients: Vec3,
}

#[derive(Component, Debug)]
pub struct EntityBufferIndex(pub i32);

#[derive(ShaderType, Clone, Component)]
pub struct EntityData {
    /// Position in world space
    position: Vec3,
    /// Rotation around the Z axis
    rotation: f32,
    /// Type of this entity
    kind: i32,
    /// Additional entity specific data
    data: Vec3,
    data2: Vec4,
}

#[derive(Component)]
pub struct RaymarchCameraEntity;

pub struct EntityBuffer(DynamicStorageBuffer<EntityData>);

struct EntityBindGroup(BindGroup);

#[derive(ShaderType, Clone, ExtractResource)]
struct RaymarchCamera {
    fov: f32,
    view_matrix: Mat4,
    inverse_view_matrix: Mat4,
    projection_matrix: Mat4,
    inverse_projection_matrix: Mat4,
}

impl Default for RaymarchCamera {
    fn default() -> Self {
        let view_matrix =
            Mat4::look_at_rh(Vec3::new(4.5, 13.0, 5.5), Vec3::new(0., 0., 0.), Vec3::Z);
        RaymarchCamera {
            fov: 75.,
            inverse_view_matrix: view_matrix.inverse(),
            view_matrix,
            projection_matrix: Default::default(),
            inverse_projection_matrix: Default::default(),
        }
    }
}

pub struct RaymarchPlugin;

impl Plugin for RaymarchPlugin {
    fn build(&self, app: &mut App) {
        dbg!(EntityData::min_size());

        app.insert_resource(SkySettings::default())
            .insert_resource(RaymarchCamera::default())
            .insert_resource(SkySettings::default())
            .add_plugin(ExtractResourcePlugin::<RaymarchCamera>::default())
            .add_plugin(ExtractResourcePlugin::<crate::World>::default())
            .add_plugin(ExtractResourcePlugin::<SkySettings>::default())
            .add_startup_system(setup_raymarching)
            .add_system(draw_settings::<SkySettings>)
            .add_system(update_tower_data)
            .add_system(update_camera);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            // .add_stage_after(
            //     RenderStage::Extract,
            //     "AfterExtract",
            //     SystemStage::single_threaded(),
            // )
            .add_render_command::<Transparent2d, DrawRaymarchQuad>()
            .init_resource::<RaymarchPipeline>()
            .init_resource::<SpecializedMeshPipelines<RaymarchPipeline>>()
            .insert_resource(RaymarchBindGroup::default())
            .insert_resource(EntityBuffer(DynamicStorageBuffer::default()))
            .add_system_to_stage(RenderStage::Extract, extract_entity_data)
            .add_system_to_stage(RenderStage::Extract, extract_raymarch_quad)
            // .add_system_to_stage("AfterExtract", push_entities_to_buffer)
            .add_system_to_stage(RenderStage::Prepare, write_entity_buffer)
            .add_system_to_stage(RenderStage::Prepare, prepare_globals)
            .add_system_to_stage(RenderStage::Queue, prepare_raymarching_bind_group)
            .add_system_to_stage(
                RenderStage::Queue,
                queue_raymarching.after(prepare_raymarching_bind_group),
            );
    }
}

fn setup_raymarching(mut commands: Commands, low_res: Res<LowResTexture>) {
    let raymarch_layer = RenderLayers::layer(1);
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(0., 0., 1.5)),
        ))
        .insert_bundle(VisibilityBundle::default())
        .insert(Mesh2dHandle(low_res.quad.clone()))
        .insert(RaymarchQuad)
        .insert(raymarch_layer);
}

fn update_tower_data(
    mut commands: Commands,
    mut towers: Query<(
        Entity,
        &Transform,
        &buildings::Tower,
        Option<&mut EntityData>,
    )>,
) {
    for (entity, transform, tower, maybe_data) in towers.iter_mut() {
        let transform: &Transform = transform;
        let tower: &buildings::Tower = tower;

        if let Some(mut data) = maybe_data {
            data.position = transform.translation;
            data.rotation = 0.; // TODO: rotation
            data.data.x = tower.height;
        } else {
            commands.entity(entity).insert(EntityData {
                position: transform.translation,
                rotation: 0.0, // TODO: rotation
                kind: buildings::Tower::KIND,
                data: Vec3::new(tower.height, 0., 0.),
                data2: Default::default(),
            });
        }
    }
}

fn update_camera(
    query: Query<(&Projection, &Transform), With<RaymarchCameraEntity>>,
    mut camera: ResMut<RaymarchCamera>,
) {
    use bevy::render::camera::CameraProjection;
    let (projection, transform) = query.iter().next().unwrap();
    let transform: &Transform = transform;

    let view_matrix = transform.clone();
    camera.view_matrix = view_matrix.compute_matrix().inverse();
    camera.inverse_view_matrix = view_matrix.compute_matrix();
    camera.projection_matrix = projection.get_projection_matrix();
    camera.inverse_projection_matrix = camera.projection_matrix.inverse();

    if let Projection::Perspective(proj) = projection {
        camera.fov = proj.fov;
    }
}

fn extract_entity_data(
    mut commands: Commands,
    entities: Extract<Query<(Entity, &EntityData)>>,
    mut buffer: ResMut<EntityBuffer>,
) {
    buffer.0.clear();
    let mut values = Vec::new();

    let mut i = 0;
    for (entity, data) in entities.iter() {
        let _offset = buffer.0.push(data.clone());
        values.push((entity, (data.clone(), EntityBufferIndex(i))));
        i += 1;
    }
    commands.insert_or_spawn_batch(values);
}

fn write_entity_buffer(
    mut buffer: ResMut<EntityBuffer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    buffer.0.write_buffer(&render_device, &render_queue);
}

impl Default for SkySettings {
    fn default() -> Self {
        SkySettings {
            planet_center: Vec3::default(),
            planet_radius: 1.0,
            atmosphere_radius: 0.27,
            sun_distance: 150_000_000.0,
            sun_axis: Vec3::Y,
            density_falloff: 6.0,
            wavelengths: Vec3::new(700., 530., 440.),
            scatter_strength: 11.7,
            scatter_coefficients: Default::default(),
        }
    }
}

impl Widget for SkySettings {
    fn draw(&mut self, ui: &mut Ui) {
        ui.label("Earth");
        ui.add(egui::Slider::new(&mut self.planet_radius, 0.1..=2.).text("Radius"));
        ui.add(egui::Slider::new(&mut self.atmosphere_radius, 0.0001..=1.).text("Atmosphere"));

        ui.label("Density Falloff");
        ui.add(egui::Slider::new(&mut self.density_falloff, 0.1..=15.));
        let mut falloff = DensityFalloff(self.density_falloff as f64);
        falloff.draw(ui);

        ui.label("Scattering");
        ui.add(egui::Slider::new(&mut self.scatter_strength, 0.1..=15.).text("Scatter strength"));
        ui.add(egui::Slider::new(&mut self.wavelengths.x, 100.0..=800.).text("Red wavelength"));
        ui.add(egui::Slider::new(&mut self.wavelengths.y, 100.0..=800.).text("Green wavelength"));
        ui.add(egui::Slider::new(&mut self.wavelengths.z, 100.0..=800.).text("Blue wavelength"));

        self.scatter_coefficients.x = (400. / self.wavelengths.x).powf(4.0) * self.scatter_strength;
        self.scatter_coefficients.y = (400. / self.wavelengths.y).powf(4.0) * self.scatter_strength;
        self.scatter_coefficients.z = (400. / self.wavelengths.z).powf(4.0) * self.scatter_strength;

        let mut scatter_curves = ScatterCurves(self.scatter_coefficients);
        scatter_curves.draw(ui);
    }
}

impl Curve for DensityFalloff {
    const VALUE_RANGE: RangeInclusive<f64> = 0. ..=2.;

    fn value_at(&self, x: f64) -> f64 {
        (-x * self.0).exp()
    }

    fn customize_plot(&self, plot: Plot) -> Plot {
        plot.width(400.).height(100.)
    }
}

struct ScatterCurve(f64);

impl Curve for ScatterCurve {
    const VALUE_RANGE: RangeInclusive<f64> = 0.0..=3.0;

    fn value_at(&self, x: f64) -> f64 {
        (-x * self.0).exp()
    }
}

struct ScatterCurves(Vec3);

impl Widget for ScatterCurves {
    fn draw(&mut self, ui: &mut Ui) {
        let r_curve = ScatterCurve(self.0.x as f64);
        let g_curve = ScatterCurve(self.0.y as f64);
        let b_curve = ScatterCurve(self.0.z as f64);

        let r_line = r_curve.line().color(Color32::RED);
        let g_line = g_curve.line().color(Color32::GREEN);
        let b_line = b_curve.line().color(Color32::BLUE);

        Plot::new("ScatterCurves")
            .width(400.)
            .height(100.)
            .show(ui, |plot_ui| {
                plot_ui.line(r_line);
                plot_ui.line(g_line);
                plot_ui.line(b_line);
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

#[derive(Component)]
pub struct RaymarchQuad;

fn extract_raymarch_quad(
    mut commands: Commands,
    query: Extract<Query<(Entity, &ComputedVisibility), With<RaymarchQuad>>>,
) {
    for (entity, computed_visibility) in query.iter() {
        if computed_visibility.is_visible() {
            commands.get_or_spawn(entity).insert(RaymarchQuad);
        }
    }
}

pub struct RaymarchPipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    fragment_shader: Handle<Shader>,
    binds_layout: BindGroupLayout,
}

#[derive(ShaderType)]
pub struct Globals {
    camera: RaymarchCamera,
    world: crate::World,
    sky_settings: SkySettings,
}

pub struct GlobalsBuffer(UniformBuffer<Globals>);

fn prepare_globals(
    mut commands: Commands,
    camera: Res<RaymarchCamera>,
    world: Res<crate::World>,
    sky_settings: Res<SkySettings>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let globals = Globals {
        camera: camera.clone(),
        world: world.clone(),
        sky_settings: sky_settings.clone(),
    };

    let mut globals_buffer = UniformBuffer::from(globals);
    globals_buffer.write_buffer(&render_device, &render_queue);

    commands.insert_resource(GlobalsBuffer(globals_buffer));
}

impl FromWorld for RaymarchPipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh2d_pipeline = Mesh2dPipeline::from_world(world);

        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();

        let binds_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("".into()),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        Self {
            mesh2d_pipeline,
            fragment_shader: asset_server.load("shaders/raymarching.wgsl"),
            binds_layout,
        }
    }
}

impl SpecializedMeshPipeline for RaymarchPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh2d_pipeline.specialize(key, layout)?;
        descriptor.fragment.as_mut().unwrap().shader = self.fragment_shader.clone();
        descriptor.layout = Some(vec![
            self.mesh2d_pipeline.view_layout.clone(),
            self.binds_layout.clone(),
            self.mesh2d_pipeline.mesh_layout.clone(),
        ]);

        Ok(descriptor)
    }
}

#[derive(Default)]
struct RaymarchBindGroup(Option<BindGroup>);

struct SetRaymarchingBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetRaymarchingBindGroup<I> {
    type Param = SRes<RaymarchBindGroup>;
    #[inline]
    fn render<'w>(
        _view: Entity,
        _item: Entity,
        raymarching_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let bind_group = &raymarching_bind_group.into_inner().0.as_ref();

        if let Some(bind_group) = bind_group {
            pass.set_bind_group(I, bind_group, &[]);
            RenderCommandResult::Success
        } else {
            println!("Failed: no bind group");
            RenderCommandResult::Failure
        }
    }
}

type DrawRaymarchQuad = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetRaymarchingBindGroup<1>,
    SetMesh2dBindGroup<2>,
    DrawMesh2d,
);

fn prepare_raymarching_bind_group(
    mut raymarch_bind_group: ResMut<RaymarchBindGroup>,
    pipeline: Res<RaymarchPipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    voxel_data: Res<VoxelData>,
    height_map: Res<HeightMap>,
    globals_buffer: Res<GlobalsBuffer>,
    entity_buffer: Res<EntityBuffer>,
    bvh_buffer: Res<BvhBuffer>,
    render_device: Res<RenderDevice>,
) {
    let voxel_view: &GpuImage = &gpu_images[&voxel_data.0];
    let map_view: &GpuImage = &gpu_images[&height_map.0];

    if let (Some(globals_buffer), Some(entity_buffer), Some(bvh_buffer)) = (
        globals_buffer.0.binding(),
        entity_buffer.0.binding(),
        bvh_buffer.0.binding(),
    ) {
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("raymarch bind group".into()),
            layout: &pipeline.binds_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: entity_buffer,
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bvh_buffer,
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&map_view.texture_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&map_view.sampler),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&voxel_view.texture_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Sampler(&voxel_view.sampler),
                },
            ],
        });
        raymarch_bind_group.0 = Some(bind_group);
    }
}

fn queue_raymarching(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    raymarching_pipeline: Res<RaymarchPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<RaymarchPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    msaa: Res<Msaa>,
    render_meshes: Res<RenderAssets<Mesh>>,
    raymarch_quads: Query<(&Mesh2dHandle, &Mesh2dUniform), With<RaymarchQuad>>,
    mut views: Query<(&VisibleEntities, &mut RenderPhase<Transparent2d>)>,
) {
    if raymarch_quads.is_empty() {
        return;
    }

    for (visible_entities, mut transparent_phase) in &mut views {
        let draw_raymarching = transparent_draw_functions
            .read()
            .get_id::<DrawRaymarchQuad>()
            .unwrap();

        let mesh_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples);

        for visible_entity in &visible_entities.entities {
            if let Ok((mesh2d_handle, mesh2d_uniform)) = raymarch_quads.get(*visible_entity) {
                let mut mesh2d_key = mesh_key;
                if let Some(mesh) = render_meshes.get(&mesh2d_handle.0) {
                    mesh2d_key |=
                        Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology);

                    let pipeline_id = pipelines
                        .specialize(
                            &mut pipeline_cache,
                            &raymarching_pipeline,
                            mesh2d_key,
                            &mesh.layout,
                        )
                        .unwrap();

                    let mesh_z = mesh2d_uniform.transform.w_axis.z;
                    transparent_phase.add(Transparent2d {
                        sort_key: FloatOrd(mesh_z),
                        entity: *visible_entity,
                        pipeline: pipeline_id,
                        draw_function: draw_raymarching,
                        batch_range: None,
                    });
                }
            }
        }
    }
}
