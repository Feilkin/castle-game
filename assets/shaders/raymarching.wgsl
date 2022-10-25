let PI = 3.141592653589793238462643383279502884197;
let E = 2.718281828459045235360287471352662497757;

struct View {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    world_position: vec3<f32>,
    width: f32,
    height: f32,
};

struct SkySettings {
    planet_center: vec3<f32>,
    planet_radius: f32,
    atmosphere_radius: f32,
    sun_distance: f32,
    sun_axis: vec3<f32>,
    density_falloff: f32,
    wavelengths: f32,
    scatter_strength: f32,
    scatter_coefficients: vec3<f32>,
}

var<private> time: f32 = 960.;
var<private> inscattering_points: i32 = 10;

let seconds_in_day = 1440.;

@group(0) @binding(0)
var<uniform> view: View;

let MAX_RT_STEPS = 180;

struct RaymarchCamera {
    fov: f32,
    view_matrix: mat4x4<f32>,
    view_matrix_inverse: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    projection_matrix_inverse: mat4x4<f32>,
}

let play_area_size = vec3<f32>(511., 511., 64.);
let play_area_size_h = vec3<f32>(255., 255., 64.);
let heightmap_size = vec2<f32>(1023., 1023.);
let voxel_size = vec3<f32>(511., 511., 64.);

struct World {
    time: f32,
}

struct Globals {
    camera: RaymarchCamera,
    world: World,
    sky_settings: SkySettings,
}

struct EntityData {
    position: vec3<f32>,
    z_rotation: f32,
    kind: i32,
    data: vec3<f32>,
    data2: vec4<f32>,
}

struct BvhNode {
    /// Minimum of the AABB
    min: vec3<f32>,
    /// Maximum of the AABB
    max: vec3<f32>,
    /// Left child index, or -1 if leaf node
    left: i32,
    /// Right child index, or entity index if leaf node
    right: i32,
}

struct BvhTree {
    tree: array<BvhNode>
}

@group(1) @binding(0)
var<uniform> globals: Globals;
@group(1) @binding(1)
var<storage> entity_data: array<EntityData>;
@group(1) @binding(2)
var<storage> bvh: BvhTree;
@group(1) @binding(3)
var texture: texture_2d<f32>;
@group(1) @binding(4)
var our_sampler: sampler;
@group(1) @binding(5)
var voxel_data: texture_3d<f32>;
@group(1) @binding(6)
var voxel_sampler: sampler;

struct FragmentInput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct HitEntities {
    count: u32,
    entities: array<EntityData, 10>,
}
var<private> hit_entities: HitEntities;

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1., 2. / 3., 1. / 3., 3.);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

fn modulo_vec3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    var r = a % b;
    r.x = select(r.x + abs(b.x), r.x, r.x >= 0.);
    r.y = select(r.y + abs(b.y), r.y, r.y >= 0.);
    r.z = select(r.z + abs(b.z), r.z, r.z >= 0.);
    return r;
}

fn modulo_vec2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    var r = a % b;
    r.x = select(r.x + abs(b.x), r.x, r.x >= 0.);
    r.y = select(r.y + abs(b.y), r.y, r.y >= 0.);
    return r;
}

// stolen from https://www.shadertoy.com/view/4tcGDr
// ray direction in view space (Y up)
fn ray_direction(fieldOfView: f32, size: vec2<f32>, fragCoord: vec2<f32>) -> vec3<f32> {
    let xy = fragCoord - size / 2.0;
    let z = size.y / tan(fieldOfView * 1.52 / 2.0);
    return normalize(vec3<f32>(xy.x, -xy.y, -z));
}

fn opSmoothUnion(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5*(d2-d1)/k, 0.0, 1.0);
    return mix(d2, d1, h) - k*h*(1.0-h);
}

fn opRepeat(p: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let q = modulo_vec3((p + 0.5*c), c) - 0.5*c;
    return q;
}

fn opRepeatFinite(p: vec3<f32>, c: f32, l: vec3<f32>) -> vec3<f32> {
    let q = p - c*clamp(round(p/c), -l, l);
    return q;
}

fn opRepeatFiniteH(p: vec3<f32>, c: f32, l: vec3<f32>) -> vec3<f32> {
    let q = p - c*clamp(round(p/c), vec3<f32>(0.), l);
    return q;
}

fn opSubstract(d1: f32, d2: f32) -> f32 {
    return max(-d1, d2);
}


fn rotateZ(p: vec3<f32>, r: f32) -> vec3<f32> {
    var n = p;
    n.x = p.x * cos(r) - p.y * sin(r);
    n.y = p.x * sin(r) + p.y * cos(r);
    return n;
}


fn rotateY(p: vec3<f32>, r: f32) -> vec3<f32> {
    var n = p;
    n.x = p.x * cos(r) + p.z * sin(r);
    n.z = -p.x * sin(r) + p.z * cos(r);
    return n;
}

fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q,vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0);
}

fn sdSphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdCylinder(p: vec3<f32>, radius: f32, height: f32) -> f32 {
    let d = abs(vec2<f32>(length(p.xy), p.z)) - vec2<f32>(height, radius);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdCone(p: vec3<f32>, c: vec2<f32>, height: f32) -> f32 {
    let q = length(p.xy);
    return max(dot(c.xy, vec2<f32>(q, p.z)), -height-p.z);
}

fn disBricks(p: vec3<f32>, d: f32) -> f32 {
    var d = d;
    let row = floor(p.z * 1.7) % 2.0;
    let column = floor(p.x * 0.57) % 2.0;


    d -= 0.055*pow(0.5 + 0.5*sin(p.x * 3.0 + row * 1.), 0.01);
    d -= 0.055*pow(0.5 + 0.5*sin(p.y * 5.0 + row * 1. + 0.13), 0.01);


    return d;
}

fn sdWall(p: vec3<f32>, b: vec3<f32>) -> f32 {
    var d = sdBox(p, b) - 0.04;
    let q = opRepeatFinite(p - vec3<f32>(0., b.y, b.z + 0.2), 1.2, vec3<f32>(floor(b.x / 1.2), 0., 0.));
    d = min(d, sdBox(q, vec3<f32>(0.3, 0.1, 0.4)) - 0.04);
    let q = opRepeatFinite(p - vec3<f32>(0., -b.y, b.z + 0.2), 1.2, vec3<f32>(floor(b.x / 1.2), 0., 0.));
    d = min(d, sdBox(q, vec3<f32>(0.3, 0.1, 0.4)) - 0.04);

    return d;
}

fn sdTower(p: vec3<f32>, height: f32, current_d: f32) -> f32 {
//    var d=sdCylinder(p, height, 2.);
    var d=sdBox(p, vec3<f32>(2., 2., height)) - 0.04;
    if (d <= current_d) {
        d = max(d, -sdBox(p - vec3<f32>(0., 0., height - 0.1), vec3<f32>(3., 0.8, 0.4)));
        d = max(d, -sdBox(p - vec3<f32>(0., 0., height - 0.1), vec3<f32>(0.8, 3., 0.4)));
    //////    d = max(d, -sdCylinder(p - vec3<f32>(0., 0., height - 0.1), 0.4, 1.8));
        d = max(d, -sdBox(p - vec3<f32>(0., 0., height - 0.1), vec3<f32>(1.8, 1.8, 0.4)));
        let q = opRepeatFinite(p - vec3<f32>(0., 0., 2.1), 2.3, vec3<f32>(0., 0., floor(height / 4.)));
        d = max(d, -sdBox(q, vec3<f32>(4., 0.31, 0.37)));
        d = max(d, -sdBox(q, vec3<f32>(0.31, 4., 0.37)));
    }
    return d;
}

fn sdBadGrass(p: vec3<f32>, ray_distance: f32) -> f32 {
    var d = p.z - 0.4;
    let grass_start = 25.;
    let grass_fade = 75.;

    if (ray_distance < grass_start) {
        let distance_factor = 1.;
        var uv = abs(p.xy * 0.3 * distance_factor) % 1.0 ;
        var r = textureSampleLevel(texture, our_sampler, uv, 0.).r;
        r = clamp(r, 0.05, 0.7);
        d = p.z - r;
    } else if (ray_distance < grass_fade) {
        let distance_factor = (ray_distance - grass_start)/(grass_fade - grass_start);
        var uv = abs(p.xy * 0.3) % 1.0;
        var r = textureSampleLevel(texture, our_sampler, uv, 0.).r;
        r = clamp(r, 0.05, 0.7);
        r = mix(r, 0.4, distance_factor);
        d = p.z - r;
    }

    return d;
}

fn sdGateway(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = sdBox(p, b);

    return d;
}

fn sdEntity(entity_index: u32, ray_pos: vec3<f32>, current_distance: f32) -> vec2<f32> {
    let entity = hit_entities.entities[entity_index];
    var ret = vec2(current_distance, -1.);

    switch (entity.kind) {
        case 1: {
            let tower_d = sdTower(ray_pos - entity.position - vec3(0., 0., 3.), entity.data.x, ret.x);
            if (tower_d < ret.x) {
                ret.x = tower_d;
                ret.y = 2.;
            }
        }
        default : {}
    }

    return ret;
}

fn sdWater(ray_pos: vec3<f32>) -> vec2<f32> {
    var waves = array(
        vec4(0.1, 0.4, 1., 1.),
        vec4(0.08, 0.3, -0.93, 3.),
        vec4(0.1, 0.47, 1., 7.),
        vec4(0.04, 0.8, 2., 1.5),
        vec4(0.051, 0.81, 1.97, 3.),
        vec4(0.061, 0.93, -1.81, -13.),
        vec4(0.003, 2.1, 2., 1.041),
        vec4(0.0021, 3.7, 1.97, 3.512),
        vec4(0.0011, 5.1, -1.81, -6.12),
    );
    var wave_acc = 0.;

    for (var i = 0; i < 9; i++) {
        let wave = waves[i];
        wave_acc += sin(ray_pos.x * wave.y + wave.w + globals.world.time * wave.z) * wave.x;
    }

    return vec2(ray_pos.z - 1. + wave_acc, -2.);
}

fn ground_at(p: vec2<f32>) -> f32 {
    let p = p + play_area_size_h.xy;
//    var uv = modulo_vec2(p - vec2<f32>(-128., -128.), vec2<f32>(512., 512.)) / vec2<f32>(511., 511.);
    var uv = p / play_area_size.xy;
    var r = textureSampleLevel(texture, our_sampler, uv, 0.).r;
    return r;
}

fn sdGround(p: vec3<f32>) -> f32 {
    let p = p + vec3<f32>(play_area_size_h.xy, 0.);
//    if (p.x < 0. || p.x > 1023. || p.y < 0. || p.y > 1023.) { return p.z; }
//    var uv = modulo_vec3(p  - vec3<f32>(-128., -128., 0.), vec3<f32>(512., 512., 300.)) / vec3<f32>(511., 511., 63.);
    var uv = p / play_area_size;
    return textureSampleLevel(voxel_data, voxel_sampler, uv, 0.).r;
}

fn calculate_ground_normal(pos: vec2<f32>) -> vec3<f32> {
    let e = vec2<f32>(1.0,0.0)*0.47734231*0.7;
    return normalize(vec3<f32>(
        2.*(ground_at(pos + e.xy) - ground_at(pos - e.xy))/e.x,
        2.*(ground_at(pos + e.yx) - ground_at(pos - e.yx))/e.x,
        4.
    ));
}


fn map(ray_pos: vec3<f32>, ray_dir: vec3<f32>, ray_distance: f32, shadow_pass: bool) -> vec2<f32> {
    let wall_height = 1.5;
    var ret = vec2<f32>(1000., 2.);

    // ground
    var r = sdGround(ray_pos);
    ret.x = min(r, ret.x);
    ret.y = 1.;

    // water
    let w = sdWater(ray_pos);
    if (w.x <= ret.x) {
        ret = w;
    }

    for (var i = 0u; i < hit_entities.count; i++) {
        let entity_ret = sdEntity(i, ray_pos, ret.x);
        if (entity_ret.x < ret.x) {
            ret.x = entity_ret.x;
            ret.y = entity_ret.y;
        }
    }


//    for (var i = 0u; i < walls.count; i++) {
//        let wall = walls.walls[i];
//        let wall_len = wall.w;
//        let wall_rot = wall.z;
//        let wall_pos = vec3<f32>(wall.xy, 0.5);
//        let ground_level_w = ground_at(ray_pos.xy);
//        let local_pos = rotateZ(ray_pos - wall_pos, -wall_rot) - vec3<f32>(-wall_len / 2.0, 0., 0.9 + ground_level_w);
//
//        let bb_box = sdBox(local_pos, vec3<f32>(wall_len / 2. + 0.5, 1.2, wall_height + 1.5));
//
//        if (bb_box < ret.x) {
//            let wall_d = sdWall(local_pos, vec3<f32>(wall_len / 2., 1., wall_height));
//
//            if (wall_d < ret.x) {
//                ret.y = 2.;
//            }
////            ret.x = opSmoothUnion(ret.x, wall_d, 0.1);
//            ret.x = min(ret.x, wall_d);
//            ret.x = max(-sdGateway(local_pos - vec3<f32>(0., 0., 2.38), vec3<f32>(wall_len / 2.0 + 0.4, 0.7, 0.9)), ret.x);
//        }
//    }

    // main gate
    ret.x = max(-sdGateway(ray_pos - vec3<f32>(0., 0., 1.), vec3<f32>(1.5, 2., 1.4)), ret.x);

    let drawbridge = sdBox(ray_pos - vec3<f32>(0., 0., .1), vec3<f32>(1.49, 4., 0.1));
    ret.y = select(ret.y, 4., drawbridge < ret.x);
    ret.x = min(ret.x, drawbridge);

    return ret;
}

fn ray_intersects_aabb(ray_pos: vec3<f32>, ray_dir: vec3<f32>, bb_min: vec3<f32>, bb_max: vec3<f32>) -> bool {
    let dirfrac = 1. / ray_dir;
    let t135 = (bb_min - ray_pos) * dirfrac;
    let t246 = (bb_max - ray_pos) * dirfrac;

    let tmin = max(max(min(t135.x, t246.x), min(t135.y, t246.y)), min(t135.z, t246.z));
    let tmax = min(min(max(t135.x, t246.x), max(t135.y, t246.y)), max(t135.z, t246.z));

    // AABB behind ray
    if (tmax < 0.) {
        return false;
    }

    // ray doesn't intersect
    if (tmin > tmax) {
        return false;
    }

    return true;
}

fn aabb_intersects_aabb(a_min: vec3<f32>, a_max: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> bool {
    return (a_min.x <= b_max.x && a_max.x >= b_min.x) &&
           (a_min.y <= b_max.y && a_max.y >= b_min.y) &&
           (a_min.z <= b_max.z && a_max.z >= b_min.z);
}

fn bvh_lookup_ray(ray_pos: vec3<f32>, ray_dir: vec3<f32>) {
    var queue: array<u32, 128>;
    var sp = 0;

    // reset hit_entities
    hit_entities.count = 0u;

    // load root to queue
    queue[0] = 0u;

    loop {
        if (sp < 0) { break; }
        // pop node from stack
        let node_id = queue[sp];
        sp--;
        let node = bvh.tree[node_id];

        let ray_hit = ray_intersects_aabb(ray_pos, ray_dir, node.min, node.max);
        if (ray_hit) {
            if (node.left == -1) {
                // leaf node, right is entity data index
                hit_entities.entities[hit_entities.count] = entity_data[node.right];
                hit_entities.count++;
            } else {
                // branch node, left and right are indices for the child nodes
                // push the child nodes to queue
                queue[sp + 1] = u32(node.left);
                queue[sp + 2] = u32(node.right);
                sp += 2;
            }
        }
    }
}

fn bvh_lookup_aabb(target_min: vec3<f32>, target_max: vec3<f32>) {
    var queue: array<u32, 128>;
    var sp = 0;

    // reset hit_entities
    hit_entities.count = 0u;

    // load root to queue
    queue[0] = 0u;

    loop {
        if (sp < 0) { break; }
        // pop node from stack
        let node_id = queue[sp];
        sp--;
        let node = bvh.tree[node_id];

        let ray_hit = aabb_intersects_aabb(target_min, target_max, node.min, node.max);
        if (ray_hit) {
            if (node.left == -1) {
                // leaf node, right is entity data index
                hit_entities.entities[hit_entities.count] = entity_data[node.right];
                hit_entities.count++;
            } else {
                // branch node, left and right are indices for the child nodes
                // push the child nodes to queue
                queue[sp + 1] = u32(node.left);
                queue[sp + 2] = u32(node.right);
                sp += 2;
            }
        }
    }
}

fn raymarch(ray_pos: vec3<f32>, ray_dir: vec3<f32>) -> vec3<f32> {
    bvh_lookup_ray(ray_pos, ray_dir);
    let max_steps = MAX_RT_STEPS;
    let epsilon = 0.01;

    var res = vec3<f32>(-1., -1., 0.);
    var tmin = 0.4;
    var tmax = 600.0;
    // raycast floor plane
//    let tp1 = (-1.0-ray_pos.z) / ray_dir.z;
//    if (tp1 > 0.0) {
//        tmax = min(tmax, tp1);
//        res.x = tp1;
//        res.y = 1.0;
//    }

    var step = 0;
    var t = tmin;
    loop {
        step += 1;
        if (step >= max_steps) { res.x = t; break; }
        let pos = ray_pos + ray_dir * t;
        if (pos.x > play_area_size_h.x && ray_dir.x > 0.) { break; }
        if (pos.y > play_area_size_h.y && ray_dir.y > 0.) { break; }
        if (pos.x < -play_area_size_h.x && ray_dir.x < 0.) { break; }
        if (pos.y < -play_area_size_h.y && ray_dir.y < 0.) { break; }

//        let map_bb = sdBox(pos, vec3<f32>(45., 25., 25.));
//
//        if ((map_bb <= res.x) || res.x == -1.) {
            let d = map(pos, ray_dir, t, false);

            t += d.x;

            if (d.x <=epsilon) {
                res.x = t;
                res.y = d.y;
                break;
            }
//        } else {
//            t += map_bb;
//        }
        if (t >= tmax) { break; }
    }

    res.z = f32(step);

    return res;
}

fn ray_sphere_intersection(sphere_pos: vec3<f32>, radius: f32, ray_pos: vec3<f32>, ray_dir: vec3<f32>) -> f32 {
    let L = sphere_pos - ray_pos;
    let tc = dot(L, ray_dir);

    let d2: f32 = (tc*tc) - (length(L) * length(L));
    let radius2 = radius * radius;

    let t1c = sqrt(radius2 - d2);
    let t1 = tc - t1c;
    let t2 = tc + t1c;

    return max(t1, t2);
}

fn sky_dir_to_sun(p: vec3<f32>) -> vec3<f32> {
    let sun_distance = 1500000.;
    let sun_pos = rotateY(vec3<f32>(sun_distance, 0., 0.), 3.7);

    return normalize(sun_pos);
}

fn sky_density_at(p: vec3<f32>) -> f32 {
    let planet_center = vec3<f32>(0., 0., 0.);
    let planet_radius = globals.sky_settings.planet_radius;
    let atmosphere_radius = planet_radius + globals.sky_settings.atmosphere_radius;
    let density_falloff = globals.sky_settings.density_falloff;
    let height_above_surface = length(p - planet_center) - planet_radius;
    let height01 = height_above_surface / (atmosphere_radius - planet_radius);
    let local_density = exp(-height01 * density_falloff);

    return local_density;
}

fn sky_optical_depth(ray_origin: vec3<f32>, ray_dir: vec3<f32>, ray_length: f32) -> f32 {
    var sample_point = ray_origin;
    let step_size = ray_length / f32(inscattering_points - 1);
    var optical_depth = 0.;

    for (var i = 0; i < inscattering_points; i++) {
        let local_density = sky_density_at(sample_point);
        optical_depth += local_density * step_size;
        sample_point += ray_dir * step_size;
    }

    return optical_depth;
}

fn sky_color(ray_dir: vec3<f32>) -> vec3<f32> {
    let planet_center = vec3<f32>(0., 0., 0.);
    let atmosphere_radius = 1.1;
    let earth_radius = .997;
    var inscatter_point = vec3<f32>(0., 0., earth_radius);
    let distance_to_sky = ray_sphere_intersection(planet_center, atmosphere_radius, inscatter_point, ray_dir);

    let step_size = distance_to_sky / f32(inscattering_points - 1);
    var inscattered_light = vec3(0.);

    for (var i = 0; i < inscattering_points; i++) {
        let dir_to_sun = sky_dir_to_sun(inscatter_point);
        let sunray_length = ray_sphere_intersection(planet_center, atmosphere_radius, inscatter_point, dir_to_sun);
        let sunray_optical_depth = sky_optical_depth(inscatter_point, dir_to_sun, sunray_length);
        let viewray_optical_depth = sky_optical_depth(inscatter_point, -ray_dir, step_size * f32(i));
        let transmittance = exp(-(sunray_optical_depth + viewray_optical_depth) * globals.sky_settings.scatter_coefficients);
        let local_density = sky_density_at(inscatter_point);

        inscattered_light += local_density * transmittance * globals.sky_settings.scatter_coefficients * step_size;
        inscatter_point += ray_dir * step_size;
    }

    return inscattered_light;
}

fn map_color(color: f32, ray_pos: vec3<f32>, normal: vec3<f32>, ray_dir: vec3<f32>) -> vec3<f32> {
    if (color == 1.0) {
        var ground_color = vec3<f32>(0.43, 0.43, 0.13);

        if (ray_pos.z > 26. + sin(ray_pos.y * .2 + ray_pos.x * 0.03)) {
            ground_color = vec3<f32>(0.71, 0.91, 0.75);
        }

        if (abs(normal.z) > 0.95) {
            ground_color = vec3<f32>(0.31, 0.59, 0.14);
            if (ray_pos.z > 21. + sin(ray_pos.x * .1)) {
                ground_color = vec3<f32>(0.77, 0.97, 0.81);
            }
        }

        if (ray_pos.z < 1.7) {
            ground_color = vec3<f32>(0.71, 0.69, 0.14);
        }

        let a = floor(modulo_vec2(ray_pos.xy * 8., vec2<f32>(16., 16.)));
        if (a.x == 0. || a.y == 0.) {
            return ground_color;
        } else {
            return ground_color * 0.81;
        }
    }
    if (color == 2.0) { return vec3<f32>(0.1, 0.12, 0.13); }
    if (color == 3.0) { return vec3<f32>(0.52, 0.7, 0.13); }
    if (color == 4.0) { return vec3<f32>(0.32, 0.11, 0.03); }
    if (color == 5.0) { return vec3<f32>(0.81, 0.11, 0.13); }
    if (color == 6.0) { return vec3<f32>(0.71, 0.31, 0.23); }
    if (color == 7.0) { return vec3<f32>(0.37, 0.71, 0.13); }

    if (color >= 100.) {
        return hsv2rgb(vec3<f32>(color - 100., 0.8, 0.5));
    }

    if (color == -1.0) {
        // sky
        return sky_color(ray_dir);
    }
    if (color == -2.0) {
        // water
//        let bounce = reflect(ray_dir, vec3<f32>(0., 0., 1.));
//        return sky_color(bounce);
        return vec3(1., 0., 1.);
    }

    return vec3<f32>(1., 0., 1.);
}

fn calculate_soft_shadow(ray_origin: vec3<f32>, ray_dir: vec3<f32>, distance_from_eye: f32) -> f32 {
    if (distance_from_eye > 500.) { return 1.; }
    bvh_lookup_ray(ray_origin, ray_dir);
    var res = 1.0;
    var t = 0.2;

    let max_steps = 24;
    var step = 0;
    loop {
        step += 1;
        if (step >= max_steps) { break; }

        let h = map(ray_origin + ray_dir * t, ray_dir, 1.0, true).x;
        let s = clamp(8.0*h/t, 0., 1.);
        res = min(res, s);
        t += clamp(h, 0.06, 0.4);
        if (res < 0.01) { break; };
    }

    return res;
}

fn calculate_normal(pos: vec3<f32>, distance_from_eye: f32) -> vec3<f32> {
    let e = vec2<f32>(1.0,-1.0)*0.57734231*0.03;
    bvh_lookup_aabb(pos + e.yyy * 4., pos + e.xxx * 4.);
    return normalize( e.xyy * (map( pos + e.xyy, vec3(1., 0., 0.), distance_from_eye, false).x) +
                      e.yyx * (map( pos + e.yyx, vec3(0., 0., 1.), distance_from_eye, false).x) +
                      e.yxy * (map( pos + e.yxy, vec3(0., 1., 0.), distance_from_eye, false).x) +
                      e.xxx * (map( pos + e.xxx, vec3(0., 0., 0.), distance_from_eye, false).x) );
}

fn calculate_ao(pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    var occ = 0.;
    var sca = 1.0;
    for (var i = 0; i < 5; i += 1) {
        let h = 0.01 + 0.12 * f32(i)/4.0;
        let d = map(pos + h*normal, vec3(0., 0., 0.), 0.01, true).x;
        occ += (h-d) * sca;
        sca *= 0.95;
        if (occ > 0.35) { break; }
    }
    return clamp(1.0 - 3.0*occ, 0.0, 1.0) * (0.5 + 0.5*normal.z);
}

fn calculate_light(ray_pos: vec3<f32>, ray_dir: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // TODO: materials
    let shininess = 200.;
    let diffuse_coeff = 1.;
    let specular_coeff = 1.0;
    // TODO: occlusion
    var light_acc = vec3(0.);
    // sun light
    {
        // TODO: sun color
        let sun_color = vec3(1.3, 1.1, 0.7);
        let sun_dir = sky_dir_to_sun(ray_pos);
        let sun_shadow = calculate_soft_shadow(ray_pos, sun_dir, 1.);

        // phong
        let half_way = normalize(sun_dir - ray_dir);
        let diffuse = clamp(dot(normal, sun_dir), 0., 1.) * sun_shadow * sun_color;
        let specular = clamp(pow(dot(reflect(-sun_dir, normal), -ray_dir), shininess), 0., 1.);

        light_acc += diffuse_coeff * diffuse + specular_coeff * specular;
//        light_acc = vec3(sun_shadow);
    }

    return light_acc;
}

struct FragmentOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

@fragment
fn fragment(in: FragmentInput) -> FragmentOutput {
    hit_entities.count = 2u;
    hit_entities.entities[0] = entity_data[1];
    hit_entities.entities[1] = entity_data[2];

    let AA = 2;
    var total = vec3<f32>(0.);
    var total_distance = 0.;
    for (var m=0; m<AA; m++) {
    for (var n=0; n<AA; n++) {
        let o = vec2<f32>(f32(m), f32(n)) / 4. - 0.5;
        let p = in.position.xy + o;

        let ray_in_view_space = ray_direction(globals.camera.fov, vec2<f32>(view.width, view.height), p);

//        let ray_ndc = normalize(vec4((in.uv.x * 2. - 1.), (1. - in.uv.y * 2.), 1., 0.));
//        let ray_dir = (raymarch_camera.projection_matrix_inverse * ray_ndc).xyz;
//        let ray_dir = ray_in_view_space * vec3(1., 1., 1.);
//        let ray_dir = ray_dir * vec3(1., 1., 1.);
        let ray_dir_unnorm = globals.camera.view_matrix_inverse * vec4<f32>(ray_in_view_space.xyz, 0.0);
        var ray_dir = normalize(ray_dir_unnorm.xyz);
//        let ray_dir = ray_ndc.xyz;

        let ray_pos: vec3<f32> = (globals.camera.view_matrix_inverse * vec4<f32>(0., 0., 0., 1.)).xyz;
        let ray_pos = ray_pos;

        let res = raymarch(ray_pos, ray_dir);


        if (res.z == f32(MAX_RT_STEPS)) {
            let fog_color = vec3<f32>(0.8, 0.8, 0.9);
            total += mix(vec3(0.0, 0.0, 0.07), fog_color, 1.0-exp(-0.00001*pow(res.x, 2.1)));
//            total += vec3(0.0, 0.0, 0.07);
        } else {
            let t = res.x;
            var m = res.y;
            var smoothness = 0.5;
            var pos = ray_pos + ray_dir * t;
            let distance_from_eye = distance(ray_pos, pos);
            var normal = vec3<f32>(0., 0., 1.);
            if (m > 0.5 && m < 1.5) {
                normal = calculate_ground_normal(pos.xy);
            }

            if (m > 1.5) {
                normal = calculate_normal(pos, distance_from_eye);
            }
        //    let normal = select(vec3<f32>(0., 0., 1.), calculate_normal(pos), m > 1.5);
            var color = map_color(m, pos, normal, ray_dir);
            if (m == -2.) {
                let water_pos = pos;
                let water_ray_dir = ray_dir;
                let water_normal = calculate_normal(water_pos, distance_from_eye);
                for (var water_bounce = 0; water_bounce < 50; water_bounce++) {
                    let bounce = reflect(ray_dir, normal);
                    let bounce_start = pos + bounce * 0.3;
                    let reflection = raymarch(bounce_start, bounce);
                    let reflection_pos = bounce_start + bounce * reflection.x;
                    var reflection_normal = vec3<f32>(0., 0., 1.);
                    if (reflection.y > 0.5 && reflection.y < 1.5) {
                        normal = calculate_ground_normal(reflection_pos.xy);
                    } else if (reflection.y > 1.5) {
                        normal = calculate_normal(reflection_pos, distance_from_eye);
                    }
                    color = map_color(reflection.y, reflection_pos, normal, bounce);
                    // mix in with shore color if shallow
                    let shore_color = map_color(1., pos, normal, ray_dir);
                    let water_depth = pos.z - ground_at(pos.xy);
                    let water_factor = 1. - clamp(pow(water_depth * .617, 1.), 0., 1.);
                    color = mix(color, shore_color, water_factor);
                    m = reflection.y;
                    pos = reflection_pos;
                    ray_dir = bounce;
                    if (m != -2.) { break; }
                }

                // hacky water specular
                let shininess = 200.;
                let sun_color = vec3(1.3, 1.1, 0.7);
                let sun_dir = sky_dir_to_sun(water_pos);

                let specular = clamp(pow(dot(reflect(sun_dir, water_normal), water_ray_dir), shininess), 0., 1.);

                color += 1.0 * specular * sun_color;
            }

            if (m > 0.) {
                var light_acc = calculate_light(pos, ray_dir, normal);
                color = light_acc * color;
                // fog
                let fog_color = vec3<f32>(0.8, 0.8, 0.9);
                color = mix(color, fog_color, 1.0-exp(-0.00001*pow(t, 2.1)));
            }

//                color = vec3<f32>(t / 1000., res.z / f32(MAX_RT_STEPS), 0.);
//                color = vec3<f32>(t / 10., 0., 0.);
//                color = vec3<f32>(0., 0., select(0., 1., normal.z > 0.93));
//                color = normal;
//                color = vec3<f32>(occ, 0., 0.);
//                color = vec3<f32>(max(ground_at(pos.xy), 0.) / 100.,-min(ground_at(pos.xy), 0.) / 100., 0.);
//                color = vec3<f32>(m / 5., 0., 0.);
//                color = ray_dir;
            total += color;
            total_distance += res.x;
        }

    }
    }

    return FragmentOutput(total_distance / f32(AA*AA), vec4<f32>(total / f32(AA*AA), 1.0));
}