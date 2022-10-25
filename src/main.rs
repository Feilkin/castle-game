//! A shaders that uses dynamic data like the time since startup.
//!
//! This example uses a specialized pipeline.
#![feature(trivial_bounds)]

use std::collections::VecDeque;

use bevy::asset::AssetServerSettings;
use bevy::diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin};
use bevy::math::Vec3Swizzles;
use bevy::render::camera::Projection;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::ImageSampler;
use bevy::window::PresentMode;
use bevy::{
    core_pipeline::clear_color::ClearColorConfig,
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_resource::{
            AsBindGroup, Extent3d, ShaderRef, ShaderType, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages,
        },
        view::RenderLayers,
    },
    sprite::{Material2d, Material2dPlugin, MaterialMesh2dBundle},
};
use bevy_egui::egui::plot::{HLine, Line, Plot, PlotPoints};
use bevy_egui::{egui, EguiContext, EguiPlugin};
use noise::Seedable;

use crate::bake::{BakeVoxelPlugin, HeightMap, VoxelData};
use crate::bvh::{BvhPlugin, CalculateBvh, LocalBoundingBox};
use crate::camera::{CameraPlugin, PanOrbitCamera};
use crate::debug::DebugPlugin;
use crate::fsr::{FsrPlugin, LowResTexture};
use crate::raymarching::{RaymarchPlugin, SkySettings};

mod bake;
mod buildings;
mod bvh;
mod camera;
mod debug;
mod fsr;
mod npc;
mod raymarching;
mod widgets;

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
        .insert_resource(World::default())
        .add_plugins(DefaultPlugins)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(FsrPlugin)
        .add_plugin(CameraPlugin)
        .add_plugin(EguiPlugin)
        .add_plugin(BakeVoxelPlugin::default())
        .add_plugin(RaymarchPlugin)
        .add_plugin(BvhPlugin)
        .add_plugin(DebugPlugin)
        .add_plugin(ExtractResourcePlugin::<World>::default())
        .add_startup_system(setup)
        .add_startup_system(print_render_limits)
        .add_system(update_world)
        .run();
}

fn print_render_limits(dev: Res<RenderDevice>) {
    println!("{:?}", dev.limits());
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
) {
    let image = generate_perlin_noise(1024, 1024);
    let image_handle = images.add(image.clone());
    commands.insert_resource(HeightMap(image_handle.clone()));

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

    // spawn_tower(
    //     &mut commands,
    //     Transform::from_translation(Vec3::new(-12., 0., 0.)),
    //     4.,
    //     &image,
    // );
    // spawn_tower(
    //     &mut commands,
    //     Transform::from_translation(Vec3::new(12., 0., 0.)),
    //     4.,
    //     &image,
    // );
    // spawn_tower(
    //     &mut commands,
    //     Transform::from_translation(Vec3::new(-12., -20., 0.)),
    //     4.,
    //     &image,
    // );
    // spawn_tower(
    //     &mut commands,
    //     Transform::from_translation(Vec3::new(12., -20., 0.)),
    //     4.,
    //     &image,
    // );
    //
    // spawn_tower(
    //     &mut commands,
    //     Transform::from_translation(Vec3::new(-8., -10., 0.)),
    //     7.,
    //     &image,
    // );
    // spawn_tower(
    //     &mut commands,
    //     Transform::from_translation(Vec3::new(8., -10., 0.)),
    //     7.,
    //     &image,
    // );
    // spawn_tower(
    //     &mut commands,
    //     Transform::from_translation(Vec3::new(-8., -17., 0.)),
    //     7.,
    //     &image,
    // );
    // spawn_tower(
    //     &mut commands,
    //     Transform::from_translation(Vec3::new(8., -17., 0.)),
    //     7.,
    //     &image,
    // );

    for y in 0..10 {
        for x in 0..10 {
            spawn_tower(
                &mut commands,
                Transform::from_translation(Vec3::new(
                    x as f32 * 10. - 150.,
                    y as f32 * 10. - 150.,
                    0.,
                )),
                14.,
                &image,
            );
        }
    }

    spawn_tower(
        &mut commands,
        Transform::from_translation(Vec3::new(-90., 10., 0.)),
        14.,
        &image,
    );
}

fn spawn_tower(commands: &mut Commands, mut transform: Transform, height: f32, heightmap: &Image) {
    let data_index = ((transform.translation.y + heightmap.size().y * 0.5 - 1.)
        * heightmap.size().x
        + (transform.translation.x + heightmap.size().x * 0.5 - 1.)) as usize
        * 4;
    let ground_level_data = &heightmap.data[data_index..data_index + 4];
    let ground_level = f32::from_le_bytes(ground_level_data.try_into().unwrap());
    transform.translation.z += ground_level;

    commands
        .spawn_bundle(TransformBundle::from_transform(transform))
        .insert(buildings::Tower { height })
        .insert(LocalBoundingBox {
            min: Vec3::new(-2., -2., -1.),
            max: Vec3::new(2., 2., height * 2. - 3.),
        })
        .insert(CalculateBvh);
}

#[derive(ShaderType, Clone, Default, ExtractResource)]
struct World {
    time: f32,
}

fn update_world(mut world: ResMut<World>, time: Res<Time>) {
    world.time += time.delta_seconds();
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
                let sx = x as f64 * scale / width as f64;
                let sy = y as f64 * scale / height as f64;
                let val = (perlin.get([sx, sy]) * amplitude) as f32;
                let i = y * width as usize + x;
                data[i] += val;
                data[i] = data[i].min(63.);
            }
        }
    }

    let data: Vec<u8> = unsafe { std::mem::transmute(data) };

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
