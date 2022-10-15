struct BakeSettings {
    voxel_size: vec3<f32>,
    heightmap_size: f32,
    erosion_inertia: f32,
    erosion_capacity: f32,
    erosion_deposition: f32,
    erosion_erosion: f32,
    erosion_gravity: f32,
    erosion_evaporation: f32,
    erosion_radius: i32,
}

@group(0) @binding(0)
var voxel_data: texture_storage_3d<r32float, write>;
@group(0) @binding(1)
var heightmap_texture: texture_2d<f32>;
@group(0) @binding(2)
var heightmap_sampler: sampler;
@group(0) @binding(3)
var<uniform> settings: BakeSettings;

fn ground_at(pos: vec2<f32>) -> f32 {
    return textureSampleLevel(heightmap_texture, heightmap_sampler, pos, 0.).r;
}
@compute @workgroup_size(8, 8, 8)
fn bake_sdf_voxel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // sample X*X points around the voxel cell to find out the distance to closest point
    let world_pos = vec3<f32>(global_id);

    let steps = 32.;
    let sample_radius = 16.;
    var sdf: f32 = sample_radius;

    for (var y = 0.; y < steps; y+= 1.) {
        let sY = world_pos.y + ((y / (steps - .1)) * 2. - 1.) * sample_radius;
        if (sY > 0. && sY < settings.voxel_size.y) {
            for (var x = 0.; x < steps; x+=1.) {
                let sX = world_pos.x + ((x / (steps - .1)) * 2. - 1.) * sample_radius;
                if (sX > 0. && sX < settings.voxel_size.y) {
                    let uv = vec2<f32>(sX, sY) / (settings.voxel_size.x - 1.);

                    let ground_level = ground_at(uv);
                    let sample_in_world = vec3<f32>(uv.x * (settings.voxel_size.x - 1.), uv.y * (settings.voxel_size.y - 1.), ground_level);

                    var d = distance(world_pos, sample_in_world);


                    sdf = min(sdf, d);
                }
            }
        }
    }
    let ground_level = ground_at(world_pos.xy / (settings.voxel_size.x - 1.));
    if (world_pos.z - ground_level < 0.) {
        sdf *= -1.;
    }

    textureStore(voxel_data, vec3<i32>(global_id), vec4<f32>(sdf, 0., 0., 0.));
}