//! Bounding volume hierarchy
use crate::raymarching::EntityBufferIndex;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::mesh::MeshVertexBufferLayout;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderRef, ShaderType, SpecializedMeshPipelineError,
    StorageBuffer,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{extract_resource::ExtractResource, Extract, RenderApp, RenderStage};

#[derive(Component)]
pub struct CalculateBvh;

/// Bounding box in model space (not rotated, not translated)
#[derive(Component)]
pub struct LocalBoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

/// Axis-aligned bounding box in world space
#[derive(Component, Copy, Clone, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn centroid(&self) -> Vec3 {
        self.min + (self.max - self.min) * 0.5
    }

    pub fn total_surface_area(&self) -> f32 {
        let extents = self.max - self.min;
        return extents.x * extents.y * 2.
            + extents.x * extents.z * 2.
            + extents.y * extents.z * 2.;
    }
}

#[derive(Clone, ExtractResource)]
pub struct BvhTree {
    root: BvhNode,
}

impl Default for BvhTree {
    fn default() -> Self {
        BvhTree {
            root: BvhNode {
                aabb: Aabb {
                    min: Default::default(),
                    max: Default::default(),
                },
                kind: BvhNodeKind::Leaf(Entity::from_raw(0)),
            },
        }
    }
}

#[derive(Clone)]
pub struct BvhNode {
    aabb: Aabb,
    kind: BvhNodeKind,
}

#[derive(Clone)]
pub enum BvhNodeKind {
    Leaf(Entity),
    Branch(Box<BvhNode>, Box<BvhNode>),
}

pub struct BvhBuffer(pub StorageBuffer<GpuTree>);

pub struct BvhPlugin;

impl Plugin for BvhPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(MaterialPlugin::<DebugMaterial>::default())
            // .add_plugin(ExtractResourcePlugin::<BvhTree>::default())
            .add_startup_system(setup_bvh)
            .add_system(update_bvh_aabb);
        // .add_system(update_bvh_debug_mesh)

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(BvhTree::default())
            .add_system_to_stage(RenderStage::Extract, extract_aabb)
            .add_system_to_stage(RenderStage::Prepare, update_bvh)
            .add_system_to_stage(RenderStage::Prepare, update_bvh_buffer.after(update_bvh));
    }
}

#[derive(Debug, Clone, ShaderType)]
pub struct GpuNode {
    /// Minimum of the AABB
    min: Vec3,
    /// Maximum of the AABB
    max: Vec3,
    /// Left child index, or -1 if leaf node
    left: i32,
    /// Right child index, or entity index if leaf node
    right: i32,
}

#[derive(Debug, Clone, ShaderType)]
pub struct GpuTree {
    #[size(runtime)]
    tree: Vec<GpuNode>,
}

#[derive(AsBindGroup, TypeUuid, Clone)]
#[uuid = "2c16e30a-7e1d-4ef0-a02e-22744b489cae"]
struct DebugMaterial {}

impl Material for DebugMaterial {
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/debug_material.wgsl".into()
    }
}

struct BvhDebugMaterial(Handle<DebugMaterial>);

fn extract_aabb(
    mut commands: Commands,
    entities: Extract<Query<(Entity, &Aabb), With<CalculateBvh>>>,
) {
    let mut values = Vec::new();

    for (entity, aabb) in entities.iter() {
        values.push((entity, (aabb.clone(), CalculateBvh)));
    }
    commands.insert_or_spawn_batch(values);
}

fn setup_bvh(mut commands: Commands, mut materials: ResMut<Assets<DebugMaterial>>) {
    let mat = materials.add(DebugMaterial {});

    commands.insert_resource(BvhDebugMaterial(mat.clone()));
}

fn update_bvh_aabb(
    mut query: Query<
        (Entity, &LocalBoundingBox, &Transform, Option<&mut Aabb>),
        (
            With<CalculateBvh>,
            Or<(Changed<Transform>, Changed<LocalBoundingBox>)>,
        ),
    >,
    mut commands: Commands,
) {
    for (entity, local_bb, transform, maybe_aabb) in query.iter_mut() {
        let local_bb: &LocalBoundingBox = local_bb;
        let transform: &Transform = transform;
        let maybe_aabb: Option<Mut<Aabb>> = maybe_aabb;

        // TODO: rotation
        let new_aabb = &local_bb.into() + transform.translation;

        if let Some(mut aabb) = maybe_aabb {
            *aabb = new_aabb
        } else {
            commands.entity(entity).insert(new_aabb);
        }
    }
}

fn update_bvh_debug_mesh(
    mut query: Query<(Entity, &Aabb, &Transform, Option<&mut Handle<Mesh>>), (Changed<Aabb>)>,
    debug_material: Res<BvhDebugMaterial>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
) {
    for (entity, aabb, transform, maybe_mesh) in query.iter_mut() {
        let aabb: &Aabb = aabb;
        let transform: &Transform = transform;

        let aabb_in_local_space: Aabb = aabb - transform.translation;

        let new_mesh = meshes.add(Mesh::from(shape::Box {
            min_x: aabb_in_local_space.min.x,
            max_x: aabb_in_local_space.max.x,
            min_y: aabb_in_local_space.min.y,
            max_y: aabb_in_local_space.max.y,
            min_z: aabb_in_local_space.min.z,
            max_z: aabb_in_local_space.max.z,
        }));

        if let Some(mut mesh) = maybe_mesh {
            *mesh = new_mesh;
        } else {
            commands.entity(entity).insert_bundle(MaterialMeshBundle {
                mesh: new_mesh,
                material: debug_material.0.clone(),
                transform: transform.clone(),
                ..default()
            });
        }
    }
}

fn update_bvh(
    mut commands: Commands,
    objects: Query<(Entity, &Aabb), With<CalculateBvh>>,
    mut entities: Local<Vec<(Entity, Aabb)>>,
    mut finished: Local<bool>,
) {
    if *finished {
        return;
    }

    entities.clear();
    // collect all entities
    for (entity, aabb) in objects.iter() {
        entities.push((entity, aabb.clone()));
    }

    if entities.is_empty() {
        println!("no entities for BVH");
        return;
    }

    // make root node
    let root = split_node(&mut entities);

    if let BvhNodeKind::Branch(left, right) = &root.kind {
        spawn_debug_cubes(&mut commands, left);
        spawn_debug_cubes(&mut commands, right);
    }

    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(0., 0., 0.)),
        ))
        .insert(root.aabb.clone());

    commands.insert_resource(BvhTree { root });
    *finished = true;
}

fn update_bvh_buffer(
    mut commands: Commands,
    tree: Res<BvhTree>,
    entity_to_index: Query<&EntityBufferIndex>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let mut nodes = Vec::new();

    push_node_to_buffer(&tree.root, &mut nodes, &entity_to_index);

    let gpu_tree = GpuTree { tree: nodes };

    let mut buffer = StorageBuffer::from(gpu_tree);
    buffer.write_buffer(&render_device, &render_queue);

    commands.insert_resource(BvhBuffer(buffer));
}

fn push_node_to_buffer(
    node: &BvhNode,
    buffer: &mut Vec<GpuNode>,
    entity_to_index: &Query<&EntityBufferIndex>,
) {
    match &node.kind {
        BvhNodeKind::Leaf(entity) => buffer.push(GpuNode {
            min: node.aabb.min,
            max: node.aabb.max,
            left: -1,
            right: entity_to_index
                .get(*entity)
                .unwrap_or(&EntityBufferIndex(-1))
                .0,
        }),
        BvhNodeKind::Branch(left, right) => {
            let own_index = buffer.len();
            buffer.push(GpuNode {
                min: node.aabb.min,
                max: node.aabb.max,
                left: 0,
                right: 0,
            });

            let left_index = buffer.len();
            push_node_to_buffer(left, buffer, &entity_to_index);

            let right_index = buffer.len();
            push_node_to_buffer(right, buffer, &entity_to_index);

            buffer[own_index].left = left_index as i32;
            buffer[own_index].right = right_index as i32;
        }
    }
}

fn spawn_debug_cubes(commands: &mut Commands, node: &BvhNode) {
    commands
        .spawn_bundle(TransformBundle::from_transform(
            Transform::from_translation(Vec3::new(0., 0., 0.)),
        ))
        .insert(node.aabb.clone());

    if let BvhNodeKind::Branch(left, right) = &node.kind {
        spawn_debug_cubes(commands, left);
        spawn_debug_cubes(commands, right);
    }
}

fn split_node(aabbs: &mut [(Entity, Aabb)]) -> BvhNode {
    assert!(aabbs.len() > 0);

    if aabbs.len() == 1 {
        return BvhNode {
            aabb: aabbs[0].1,
            kind: BvhNodeKind::Leaf(aabbs[0].0),
        };
    }

    let x_index_and_cost = {
        aabbs.sort_by(|a, b| a.1.centroid().x.total_cmp(&b.1.centroid().x));
        find_split_index_and_cost(&aabbs)
    };
    let y_index_and_cost = {
        aabbs.sort_by(|a, b| a.1.centroid().y.total_cmp(&b.1.centroid().y));
        find_split_index_and_cost(&aabbs)
    };
    let z_index_and_cost = {
        aabbs.sort_by(|a, b| a.1.centroid().z.total_cmp(&b.1.centroid().z));
        find_split_index_and_cost(&aabbs)
    };

    let (left, right) =
        if x_index_and_cost.1 < y_index_and_cost.1 && x_index_and_cost.1 < z_index_and_cost.1 {
            aabbs.sort_by(|a, b| a.1.centroid().x.total_cmp(&b.1.centroid().x));
            aabbs.split_at_mut(x_index_and_cost.0)
        } else if y_index_and_cost.1 < z_index_and_cost.1 {
            aabbs.sort_by(|a, b| a.1.centroid().y.total_cmp(&b.1.centroid().y));
            aabbs.split_at_mut(y_index_and_cost.0)
        } else {
            aabbs.split_at_mut(z_index_and_cost.0)
        };

    let left_node = split_node(left);
    let right_node = split_node(right);

    BvhNode {
        aabb: merge_aabbs(aabbs),
        kind: BvhNodeKind::Branch(Box::new(left_node), Box::new(right_node)),
    }
}

fn find_split_index_and_cost(aabbs: &[(Entity, Aabb)]) -> (usize, f32) {
    assert!(aabbs.len() > 1);
    let mut min = (1, f32::INFINITY);

    for i in 1..aabbs.len() {
        let current_cost = cost(aabbs, i);
        if current_cost < min.1 {
            min = (i, current_cost);
        }
    }

    min
}

fn cost(aabbs: &[(Entity, Aabb)], index: usize) -> f32 {
    let (left, right) = aabbs.split_at(index);

    merge_aabbs(left).total_surface_area() * (index as f32)
        + merge_aabbs(right).total_surface_area() * (aabbs.len() - index) as f32
}

fn merge_aabbs(aabbs: &[(Entity, Aabb)]) -> Aabb {
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(-f32::INFINITY, -f32::INFINITY, -f32::INFINITY);

    for aabb in aabbs {
        min.x = min.x.min(aabb.1.min.x.min(aabb.1.max.x));
        min.y = min.y.min(aabb.1.min.y.min(aabb.1.max.y));
        min.z = min.z.min(aabb.1.min.z.min(aabb.1.max.z));
        max.x = max.x.max(aabb.1.min.x.max(aabb.1.max.x));
        max.y = max.y.max(aabb.1.min.y.max(aabb.1.max.y));
        max.z = max.z.max(aabb.1.min.z.max(aabb.1.max.z));
    }

    assert_ne!(min.length(), f32::INFINITY);
    assert_ne!(max.length(), f32::INFINITY);

    return Aabb { min, max };
}

impl From<&LocalBoundingBox> for Aabb {
    fn from(local_bb: &LocalBoundingBox) -> Self {
        Aabb {
            min: local_bb.min,
            max: local_bb.max,
        }
    }
}

impl std::ops::Sub<Vec3> for &Aabb {
    type Output = Aabb;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Aabb {
            min: self.min - rhs,
            max: self.max - rhs,
        }
    }
}

impl std::ops::Add<Vec3> for &Aabb {
    type Output = Aabb;

    fn add(self, rhs: Vec3) -> Self::Output {
        Aabb {
            min: self.min + rhs,
            max: self.max + rhs,
        }
    }
}
