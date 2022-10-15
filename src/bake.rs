//! Sdf Voxel Baking

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::RenderGraph;
use bevy::render::render_graph::{Node as RenderNode, NodeRunError, RenderGraphContext};
use bevy::render::render_resource::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferInitDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
    ComputePassDescriptor, ComputePipelineDescriptor, DynamicStorageBuffer, Extent3d,
    ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, PipelineCache, SamplerBindingType,
    ShaderStages, ShaderType, StorageBuffer, StorageTextureAccess, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDimension,
    UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{RenderApp, RenderStage};
use std::num::NonZeroU32;

const WORKGROUP_SIZE: u32 = 8;

pub struct BakeVoxelPlugin {
    voxel_res: [u32; 3],
}

impl Default for BakeVoxelPlugin {
    fn default() -> Self {
        BakeVoxelPlugin {
            voxel_res: [1024, 1024, 64],
        }
    }
}

#[derive(Clone, Deref, ExtractResource)]
pub struct VoxelData(pub Handle<Image>);

#[derive(Clone, Deref, ExtractResource)]
pub struct HeightMap(pub Handle<Image>);

struct HeightMapBuffer(StorageBuffer<Vec<f32>>);

struct ErodeBindGroup(BindGroup);
struct BakeSdfVoxelBindGroup(BindGroup);

#[derive(Clone, ExtractResource)]
pub struct VoxelSettings {
    voxel_res: [u32; 3],
}

#[derive(ShaderType, Clone)]
struct Droplet {
    pos: Vec2,
    dir: Vec2,
    vel: f32,
    water: f32,
    sediment: f32,
    dead: f32,
}

impl Droplet {
    fn random(settings: &BakeSettings) -> Self {
        let x = rand::random::<f32>() * settings.heightmap_size;
        let y = rand::random::<f32>() * settings.heightmap_size;

        Droplet {
            pos: Vec2::new(x, y),
            dir: Default::default(),
            vel: 1.0,
            water: 1.0,
            sediment: 0.0,
            dead: 0.,
        }
    }
}

struct BakeSettingsBuffer(UniformBuffer<BakeSettings>);
struct DropletsBuffer(StorageBuffer<Vec<Droplet>>);

#[derive(ShaderType, Clone, ExtractResource)]
pub struct BakeSettings {
    voxel_size: Vec3,
    heightmap_size: f32,
    erosion_inertia: f32,
    erosion_capacity: f32,
    erosion_deposition: f32,
    erosion_erosion: f32,
    erosion_gravity: f32,
    erosion_evaporation: f32,
    erosion_radius: i32,
}

impl Default for BakeSettings {
    fn default() -> Self {
        BakeSettings {
            voxel_size: Vec3::new(512., 512., 64.),
            heightmap_size: 1024.,
            erosion_inertia: 0.05,
            erosion_capacity: 8.,
            erosion_deposition: 0.3,
            erosion_erosion: 0.3,
            erosion_gravity: 0.25,
            erosion_evaporation: 0.01,
            erosion_radius: 4,
        }
    }
}

impl Default for VoxelSettings {
    fn default() -> Self {
        VoxelSettings {
            voxel_res: [512, 512, 64],
        }
    }
}

impl Plugin for BakeVoxelPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(VoxelSettings {
            voxel_res: self.voxel_res.clone(),
        })
        .insert_resource(BakeSettings::default())
        .add_plugin(ExtractResourcePlugin::<VoxelData>::default())
        .add_plugin(ExtractResourcePlugin::<HeightMap>::default())
        .add_plugin(ExtractResourcePlugin::<VoxelSettings>::default())
        .add_plugin(ExtractResourcePlugin::<BakeSettings>::default())
        .add_startup_system(setup_bake_stuff);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<BakeSdfPipeline>()
            .add_system_to_stage(RenderStage::Prepare, prepare_buffers)
            .add_system_to_stage(RenderStage::Queue, queue_bind_group);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("bake_sdf_voxel", BakeSdfVoxelNode::default());
        render_graph
            .add_node_edge(
                "bake_sdf_voxel",
                bevy::render::main_graph::node::CAMERA_DRIVER,
            )
            .unwrap();
    }
}

pub fn setup_bake_stuff(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    voxel_settings: Res<VoxelSettings>,
) {
    let voxel_size = Extent3d {
        width: voxel_settings.voxel_res[0],
        height: voxel_settings.voxel_res[1],
        depth_or_array_layers: voxel_settings.voxel_res[2],
    };

    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size: voxel_size.clone(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
        },
        ..default()
    };

    image.resize(voxel_size);

    let image_handle = images.add(image);

    commands.insert_resource(VoxelData(image_handle));
}

fn prepare_buffers(
    mut commands: Commands,
    bake_settings: Res<BakeSettings>,
    mut added: Local<bool>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if *added {
        return;
    }
    let mut heightmap_buffer = StorageBuffer::from(vec![0.; 1024 * 1024]);

    heightmap_buffer.write_buffer(&render_device, &render_queue);
    commands.insert_resource(HeightMapBuffer(heightmap_buffer));

    let mut buffer = UniformBuffer::from(bake_settings.clone());
    buffer.write_buffer(&render_device, &render_queue);
    commands.insert_resource(BakeSettingsBuffer(buffer));

    let mut droplets = Vec::with_capacity(4_000_000);

    for _ in 0..400_000 {
        droplets.push(Droplet::random(&bake_settings));
    }

    let mut droplet_buffer = StorageBuffer::from(droplets);

    droplet_buffer.write_buffer(&render_device, &render_queue);

    commands.insert_resource(DropletsBuffer(droplet_buffer));

    *added = true;
}

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<BakeSdfPipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    voxel_data: Res<VoxelData>,
    height_map: Res<HeightMap>,
    render_device: Res<RenderDevice>,
    bake_settings_buffer: Res<BakeSettingsBuffer>,
    droplets_buffer: Res<DropletsBuffer>,
    heightmap_buffer: Res<HeightMapBuffer>,
) {
    let voxel_view: &GpuImage = &gpu_images[&voxel_data.0];
    let map_view: &GpuImage = &gpu_images[&height_map.0];

    if let Some(binding) = bake_settings_buffer.0.binding() {
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.bake_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&voxel_view.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&map_view.texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&map_view.sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: binding.clone(),
                },
            ],
        });
        commands.insert_resource(BakeSdfVoxelBindGroup(bind_group));

        if let Some(droplet_binding) = droplets_buffer.0.binding() {
            let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.erode_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: heightmap_buffer.0.binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: binding.clone(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: droplet_binding.clone(),
                    },
                ],
            });

            commands.insert_resource(ErodeBindGroup(bind_group));
        }
    }
}

struct BakeSdfPipeline {
    bake_bind_group_layout: BindGroupLayout,
    erode_bind_group_layout: BindGroupLayout,
    erode_pipeline: CachedComputePipelineId,
    bake_pipeline: CachedComputePipelineId,
}

impl FromWorld for BakeSdfPipeline {
    fn from_world(world: &mut World) -> Self {
        let bake_bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::WriteOnly,
                                format: TextureFormat::R32Float,
                                view_dimension: TextureViewDimension::D3,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: true },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Sampler(SamplerBindingType::Filtering),
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let erode_bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/bake_heightmap.wgsl");

        let erode_shader = world
            .resource::<AssetServer>()
            .load("shaders/erode_heightmap.wgsl");

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let bake_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("bake_sdf_voxel_pipeline".into()),
            layout: Some(vec![bake_bind_group_layout.clone()]),
            shader,
            shader_defs: vec![],
            entry_point: "bake_sdf_voxel".into(),
        });

        let erode_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("erode_pipeline".into()),
            layout: Some(vec![erode_bind_group_layout.clone()]),
            shader: erode_shader,
            shader_defs: vec![],
            entry_point: "erode_heightmap".into(),
        });

        BakeSdfPipeline {
            bake_bind_group_layout,
            erode_bind_group_layout,
            erode_pipeline,
            bake_pipeline,
        }
    }
}

enum BakeSdfVoxelState {
    Loading,
    Eroding { iteration: usize },
    Baking,
    Done,
}

struct BakeSdfVoxelNode {
    state: BakeSdfVoxelState,
}

impl Default for BakeSdfVoxelNode {
    fn default() -> Self {
        BakeSdfVoxelNode {
            state: BakeSdfVoxelState::Loading,
        }
    }
}

impl RenderNode for BakeSdfVoxelNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<BakeSdfPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        match &mut self.state {
            BakeSdfVoxelState::Loading => {
                if world.get_resource::<BakeSettingsBuffer>().is_none() {
                    return;
                }

                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.erode_pipeline)
                {
                    self.state = BakeSdfVoxelState::Eroding { iteration: 0 };
                    println!("Node loading done");
                }
            }
            BakeSdfVoxelState::Eroding { iteration } => {
                if *iteration >= 30 {
                    if let CachedPipelineState::Ok(_) =
                        pipeline_cache.get_compute_pipeline_state(pipeline.bake_pipeline)
                    {
                        self.state = BakeSdfVoxelState::Baking;
                        println!("Erosion done");
                    }
                } else {
                    *iteration += 1;
                }
            }
            BakeSdfVoxelState::Baking => {
                self.state = BakeSdfVoxelState::Done;
            }
            _ => {}
        }
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if world.get_resource::<BakeSdfVoxelBindGroup>().is_none() {
            return Ok(());
        }

        let bake_bind_group = &world.resource::<BakeSdfVoxelBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<BakeSdfPipeline>();
        let voxel_settings = world.resource::<VoxelSettings>();
        let gpu_images = world.resource::<RenderAssets<Image>>();
        let heightmap_buffer = world.resource::<HeightMapBuffer>();
        let heightmap = world.resource::<HeightMap>();
        let heightmap_texture = &gpu_images[&heightmap.0];

        // copy buffers
        match self.state {
            BakeSdfVoxelState::Loading => {}
            BakeSdfVoxelState::Eroding { iteration } => {
                if iteration == 0 {
                    println!("copying texxture to buffer");
                    render_context.command_encoder.copy_texture_to_buffer(
                        ImageCopyTexture {
                            texture: &heightmap_texture.texture,
                            mip_level: 0,
                            origin: Default::default(),
                            aspect: Default::default(),
                        },
                        ImageCopyBuffer {
                            buffer: heightmap_buffer.0.buffer().unwrap(),
                            layout: ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(unsafe { NonZeroU32::new_unchecked(1024 * 4) }),
                                rows_per_image: None,
                            },
                        },
                        Extent3d {
                            width: 1024,
                            height: 1024,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            BakeSdfVoxelState::Baking => {
                println!("copying buffer to texture");
                render_context.command_encoder.copy_buffer_to_texture(
                    ImageCopyBuffer {
                        buffer: heightmap_buffer.0.buffer().unwrap(),
                        layout: ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(unsafe { NonZeroU32::new_unchecked(1024 * 4) }),
                            rows_per_image: None,
                        },
                    },
                    ImageCopyTexture {
                        texture: &heightmap_texture.texture,
                        mip_level: 0,
                        origin: Default::default(),
                        aspect: Default::default(),
                    },
                    Extent3d {
                        width: 1024,
                        height: 1024,
                        depth_or_array_layers: 1,
                    },
                );
            }
            BakeSdfVoxelState::Done => {}
        }

        match self.state {
            BakeSdfVoxelState::Loading => {}
            BakeSdfVoxelState::Eroding { iteration } => {
                println!("eroding: {}", iteration);
                {
                    let bind_group = &world.resource::<ErodeBindGroup>().0;

                    let mut pass = render_context
                        .command_encoder
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_bind_group(0, bind_group, &[]);
                    let erode_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.erode_pipeline)
                        .unwrap();
                    pass.set_pipeline(erode_pipeline);
                    pass.dispatch_workgroups(4_000_000 / 512, 1, 1);
                }
            }
            BakeSdfVoxelState::Baking => {
                let mut pass = render_context
                    .command_encoder
                    .begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_bind_group(0, bake_bind_group, &[]);
                let bake_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.bake_pipeline)
                    .unwrap();
                pass.set_pipeline(bake_pipeline);
                pass.dispatch_workgroups(
                    dbg!(voxel_settings.voxel_res[0] / WORKGROUP_SIZE),
                    voxel_settings.voxel_res[1] / WORKGROUP_SIZE,
                    voxel_settings.voxel_res[2] / WORKGROUP_SIZE,
                );
            }
            BakeSdfVoxelState::Done => {}
        }

        Ok(())
    }
}
