//! A shaders that uses dynamic data like the time since startup.
//!
//! This example uses a specialized pipeline.
#![feature(trivial_bounds)]

mod bake;
mod camera;
mod fsr;
mod npc;
mod raymarching;
mod widgets;

use crate::bake::{BakeVoxelPlugin, HeightMap, VoxelData};
use crate::camera::{CameraPlugin, PanOrbitCamera};
use crate::fsr::{FsrPlugin, FsrSettings, LowResTexture};
use crate::raymarching::{RaymarchPlugin, SkySettings};
use bevy::asset::AssetServerSettings;
use bevy::diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::math::Vec3Swizzles;
use bevy::reflect::Uuid;
use bevy::render::camera::{Projection, ScalingMode, Viewport};
use bevy::render::mesh::MeshVertexBufferLayout;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::encase::ArrayLength;
use bevy::render::render_resource::{
    AsBindGroupError, BindGroupLayout, PreparedBindGroup, RenderPipelineDescriptor,
    SpecializedMeshPipelineError,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::{FallbackImage, ImageSampler};
use bevy::ui::UiPlugin;
use bevy::window::PresentMode;
use bevy::{
    core_pipeline::clear_color::ClearColorConfig,
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::RenderTarget,
        render_resource::{
            AsBindGroup, Extent3d, ShaderRef, ShaderType, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages,
        },
        texture::BevyDefault,
        view::RenderLayers,
    },
    sprite::{Material2d, Material2dKey, Material2dPlugin, MaterialMesh2dBundle},
};
use bevy_egui::egui::plot::{HLine, Line, Plot, PlotPoints};
use bevy_egui::{egui, EguiContext, EguiPlugin};
use noise::Seedable;
use std::collections::VecDeque;

fn main() {
    App::new()
        .insert_resource(AssetServerSettings {
            watch_for_changes: true,
            ..default()
        })
        .insert_resource(WindowDescriptor {
            width: 1920.0,
            height: 1080.0,
            present_mode: PresentMode::AutoVsync,
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(FsrPlugin)
        .add_plugin(CameraPlugin)
        .add_plugin(Material2dPlugin::<RaymarchMaterial>::default())
        .add_plugin(EguiPlugin)
        .add_plugin(BakeVoxelPlugin::default())
        .add_plugin(RaymarchPlugin)
        .add_startup_system(setup)
        .add_startup_system(print_render_limits)
        .add_system(update_camera)
        .add_system(text_update_system)
        .add_system(update_towers)
        .add_system(update_walls)
        .add_system(update_voxel_data)
        .add_system(update_time)
        .add_system(update_sky_settings)
        .run();
}

fn print_render_limits(dev: Res<RenderDevice>) {
    println!("{:?}", dev.limits());
}

fn setup(
    mut commands: Commands,
    mut post_processing_materials: ResMut<Assets<RaymarchMaterial>>,
    mut images: ResMut<Assets<Image>>,
    low_res: Res<LowResTexture>,
    asset_server: Res<AssetServer>,
) {
    let image = generate_perlin_noise(1024, 1024);
    let image_handle = images.add(image.clone());
    commands.insert_resource(HeightMap(image_handle.clone()));

    let dummy_image = Image::new_fill(
        Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 2,
        },
        TextureDimension::D3,
        &(0f32).to_le_bytes(),
        TextureFormat::R32Float,
    );
    let dummy_handle = images.add(dummy_image);

    let material_handle = post_processing_materials.add(RaymarchMaterial {
        raymarch_camera: Default::default(),
        voxel_data: dummy_handle,
        towers: Default::default(),
        walls: Default::default(),
        hieghtmap: image_handle,
        world: Default::default(),
        sky_settings: Default::default(),
    });
    commands.insert_resource(RaymarchMaterialHandle(material_handle.clone()));
    let raymarch_layer = RenderLayers::layer(1);
    commands
        .spawn_bundle(MaterialMesh2dBundle {
            mesh: low_res.quad.clone().into(),
            material: material_handle.clone(),
            transform: Transform {
                translation: Vec3::new(0.0, 0.0, 1.5),
                ..default()
            },
            ..default()
        })
        .insert(raymarch_layer);

    // ui camera
    commands.spawn_bundle(Camera2dBundle {
        camera: Camera {
            priority: 2,
            ..default()
        },
        camera_2d: Camera2d {
            clear_color: ClearColorConfig::None,
        },
        ..default()
    });
    commands
        .spawn_bundle(
            // Create a TextBundle that has a Text with a list of sections.
            TextBundle::from_sections([
                TextSection::new(
                    "FPS: ",
                    TextStyle {
                        font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                        font_size: 24.0,
                        color: Color::BLACK,
                    },
                ),
                TextSection::from_style(TextStyle {
                    font: asset_server.load("fonts/FiraSans-Medium.ttf"),
                    font_size: 24.0,
                    color: Color::BLACK,
                }),
            ])
            .with_style(Style {
                align_self: AlignSelf::FlexEnd,
                ..default()
            }),
        )
        .insert(FpsText);

    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(-12., 0., 0.)),
        ))
        .insert(Tower { height: 4. });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(12., 0., 0.)),
        ))
        .insert(Tower { height: 4. });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(-12., -20., 0.)),
        ))
        .insert(Tower { height: 4. });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(12., -20., 0.)),
        ))
        .insert(Tower { height: 4. });

    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(-8., -10., 0.)),
        ))
        .insert(Tower { height: 7. });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(8., -10., 0.)),
        ))
        .insert(Tower { height: 7. });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(-8., -17., 0.)),
        ))
        .insert(Tower { height: 7. });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(8., -17., 0.)),
        ))
        .insert(Tower { height: 7. });

    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(12., 0., 0.)),
        ))
        .insert(Wall {
            rotation: 0.0,
            length: 24.0,
        });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(12., 0., 0.)),
        ))
        .insert(Wall {
            rotation: std::f32::consts::FRAC_PI_2,
            length: 20.0,
        });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(-12., 0., 0.)),
        ))
        .insert(Wall {
            rotation: std::f32::consts::FRAC_PI_2,
            length: 20.0,
        });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(12., -20., 0.)),
        ))
        .insert(Wall {
            rotation: 0.0,
            length: 24.0,
        });

    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(8., -10., 0.)),
        ))
        .insert(Wall {
            rotation: 0.0,
            length: 16.0,
        });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(8., -10., 0.)),
        ))
        .insert(Wall {
            rotation: std::f32::consts::FRAC_PI_2,
            length: 7.0,
        });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(-8., -10., 0.)),
        ))
        .insert(Wall {
            rotation: std::f32::consts::FRAC_PI_2,
            length: 7.0,
        });
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(8., -17., 0.)),
        ))
        .insert(Wall {
            rotation: 0.0,
            length: 16.0,
        });
}

#[derive(ShaderType, Clone)]
struct RaymarchCamera {
    fov: f32,
    view_matrix: Mat4,
    inverse_view_matrix: Mat4,
}

impl Default for RaymarchCamera {
    fn default() -> Self {
        let view_matrix =
            Mat4::look_at_rh(Vec3::new(4.5, 13.0, 5.5), Vec3::new(0., 0., 0.), Vec3::Z);
        RaymarchCamera {
            fov: 75.,
            inverse_view_matrix: view_matrix.inverse(),
            view_matrix,
        }
    }
}

#[derive(ShaderType, Clone, Default)]
struct Towers {
    length: u32,
    towers: [Vec4; 32],
}

#[derive(ShaderType, Clone, Default, Component)]
struct Wall {
    rotation: f32,
    length: f32,
}

#[derive(ShaderType, Clone, Default)]
struct Walls {
    length: u32,
    walls: [Vec4; 32],
}

#[derive(ShaderType, Clone, Default)]
struct World {
    time: f32,
}

struct RaymarchMaterialHandle(Handle<RaymarchMaterial>);

/// Our custom post processing material
#[derive(AsBindGroup, TypeUuid, Clone)]
#[uuid = "bc2f08eb-a0fb-43f1-a908-54871ea597d5"]
struct RaymarchMaterial {
    #[uniform(0)]
    raymarch_camera: RaymarchCamera,
    #[texture(1)]
    #[sampler(2)]
    hieghtmap: Handle<Image>,
    #[uniform(3)]
    towers: Towers,
    #[uniform(4)]
    walls: Walls,
    #[texture(5, dimension = "3d", filterable = true)]
    #[sampler(6, sampler_type = "filtering")]
    voxel_data: Handle<Image>,
    #[uniform(7)]
    world: World,
    #[uniform(8)]
    sky_settings: SkySettings,
}

fn generate_perlin_noise(width: u32, height: u32) -> Image {
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    use noise::{NoiseFn, Perlin};
    // let seed = rand::random();
    let seed = 290586575;
    let perlin = Perlin::new().set_seed(seed);
    println!("seed: {}", seed);
    // let mut data = Vec::with_capacity((width * height * 4) as usize);
    let mut data = vec![32.; (width * height * 4) as usize];

    let amplitudes_and_scales = [
        (1., 32.),
        (2., 16.),
        (4., 8.),
        (8., 4.),
        (16., 2.),
        (32., 1.),
    ];

    for (scale, amplitude) in amplitudes_and_scales {
        for y in 0..height as usize {
            for x in 0..width as usize {
                let sx = (x as f64 * scale / width as f64);
                let sy = (y as f64 * scale / height as f64);
                let val = (perlin.get([sx, sy]) * amplitude) as f32;
                let i = y * (width as usize) + x;
                data[i] += val;
                data[i] = data[i].min(63.);
            }
        }
    }

    let data: Vec<u8> = unsafe { std::mem::transmute(data) };

    image::save_buffer("heightmap.png", &data, width, height, image::ColorType::L8);

    Image {
        data,
        texture_descriptor: TextureDescriptor {
            label: Some("perlin noise"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC
                | TextureUsages::STORAGE_BINDING,
        },
        sampler_descriptor: ImageSampler::linear(),
        texture_view_descriptor: None,
    }
}

impl Material2d for RaymarchMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/raymarching.wgsl".into()
    }
}

fn update_camera(
    query: Query<(&Projection, &Transform, &PanOrbitCamera)>,
    handle: Res<RaymarchMaterialHandle>,
    mut mats: ResMut<Assets<RaymarchMaterial>>,
) {
    let (projection, transform, panorbit) = query.iter().next().unwrap();
    let transform: &Transform = transform;
    let panorbit: &PanOrbitCamera = panorbit;

    let mat = mats.get_mut(&handle.0).unwrap();
    let view_matrix = transform.clone().looking_at(panorbit.focus, Vec3::Z);
    mat.raymarch_camera.view_matrix = view_matrix.compute_matrix().inverse();
    mat.raymarch_camera.inverse_view_matrix = view_matrix.compute_matrix();

    if let Projection::Perspective(proj) = projection {
        mat.raymarch_camera.fov = proj.fov;
    }
}

#[derive(Component)]
struct Tower {
    height: f32,
}

fn update_towers(
    tower_query: Query<(&Transform, &Tower), Added<Tower>>,
    mat_query: Query<(&Handle<RaymarchMaterial>,)>,
    mut mats: ResMut<Assets<RaymarchMaterial>>,
) {
    let (handle,) = mat_query.iter().next().unwrap();
    let mat = mats.get_mut(handle).unwrap();

    for (transform, tower) in tower_query.iter() {
        if mat.towers.length >= 32 {
            break;
        }

        let transform: &Transform = transform;
        let tower = transform.translation.extend(tower.height);

        mat.towers.towers[mat.towers.length as usize] = tower;
        mat.towers.length += 1;
    }
}

fn update_walls(
    walls_query: Query<(&Transform, &Wall), Added<Wall>>,
    mat_query: Query<(&Handle<RaymarchMaterial>,)>,
    mut mats: ResMut<Assets<RaymarchMaterial>>,
) {
    let (handle,) = mat_query.iter().next().unwrap();
    let mat = mats.get_mut(handle).unwrap();

    for (transform, wall) in walls_query.iter() {
        if mat.walls.length >= 32 {
            break;
        }

        let transform: &Transform = transform;
        let wall = transform
            .translation
            .xy()
            .extend(wall.rotation)
            .extend(wall.length);

        mat.walls.walls[mat.walls.length as usize] = wall;
        mat.walls.length += 1;
    }
}

fn update_voxel_data(
    mat_query: Query<(&Handle<RaymarchMaterial>,)>,
    mut mats: ResMut<Assets<RaymarchMaterial>>,
    voxel_data: Res<VoxelData>,
) {
    if voxel_data.is_changed() {
        let (handle,) = mat_query.iter().next().unwrap();
        let mat = mats.get_mut(handle).unwrap();
        mat.voxel_data = voxel_data.0.clone();
    }
}

fn update_time(
    mat_query: Query<(&Handle<RaymarchMaterial>,)>,
    mut mats: ResMut<Assets<RaymarchMaterial>>,
    time: Res<Time>,
) {
    let (handle,) = mat_query.iter().next().unwrap();
    let mat = mats.get_mut(handle).unwrap();
    mat.world.time += time.delta_seconds();
}

fn update_sky_settings(
    mat_query: Query<(&Handle<RaymarchMaterial>,)>,
    mut mats: ResMut<Assets<RaymarchMaterial>>,
    settings: Res<SkySettings>,
) {
    let (handle,) = mat_query.iter().next().unwrap();
    let mat = mats.get_mut(handle).unwrap();
    mat.sky_settings = settings.clone();
}

#[derive(Component)]
struct FpsText;

fn text_update_system(
    diagnostics: Res<Diagnostics>,
    mut query: Query<&mut Text, With<FpsText>>,
    mut egui_context: ResMut<EguiContext>,
    mut averages: Local<VecDeque<f64>>,
    mut frame_times: Local<VecDeque<f64>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(average) = fps.average() {
                // Update the value of the second section
                text.sections[1].value = format!("{:.2}", average);
            }
        }
    }

    egui::Window::new("FPS")
        .default_height(400.)
        .default_pos([0., 720.])
        .show(egui_context.ctx_mut(), |ui| {
            if let Some(fps) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
                if let Some(value) = fps.value() {
                    averages.push_front(value);
                    averages.truncate(2000);

                    let points: PlotPoints = averages
                        .iter()
                        .rev()
                        .enumerate()
                        .map(|(i, v)| [i as f64, *v])
                        .collect();
                    let line = Line::new(points);

                    Plot::new("averages")
                        .height(200.)
                        .show(ui, |plot_ui| plot_ui.line(line));
                }
            }

            if let Some(frame_time) = diagnostics.get(FrameTimeDiagnosticsPlugin::FRAME_TIME) {
                if let Some(value) = frame_time.value() {
                    // value is in seconds, make it more reasonable ms
                    frame_times.push_front(value * 1000.);
                    frame_times.truncate(2000);

                    let points: PlotPoints = frame_times
                        .iter()
                        .rev()
                        .enumerate()
                        .map(|(i, v)| [i as f64, *v])
                        .collect();
                    let line = Line::new(points);

                    let average = frame_times.iter().sum::<f64>() / frame_times.len() as f64;
                    let average_line = HLine::new(average);

                    Plot::new("frame times").height(200.).show(ui, |plot_ui| {
                        plot_ui.line(line);
                        plot_ui.hline(average_line)
                    });
                }
            }
        });
}
