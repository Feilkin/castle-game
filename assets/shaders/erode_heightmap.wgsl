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

struct Droplet {
    pos: vec2<f32>,
    dir: vec2<f32>,
    vel: f32,
    water: f32,
    sediment: f32,
    dead: f32,
}

@group(0) @binding(0)
var<storage, read_write> heightmap: array<f32, 1048576>;
@group(0) @binding(1)
var<uniform> settings: BakeSettings;
@group(0) @binding(2)
var<storage, read_write> droplets: array<Droplet, 4000000>;

fn gradient_and_height(pos: vec2<f32>) -> vec3<f32> {
    let map_size = i32(settings.heightmap_size);
    let ipos = vec2<i32>(pos);
    let fpos = fract(pos);

    let node_nw = ipos.y * map_size + ipos.x;
    let height_nw = heightmap[node_nw];
    let height_ne = heightmap[node_nw + 1];
    let height_sw = heightmap[node_nw + map_size];
    let height_se = heightmap[node_nw + map_size + 1];

    let gradient = vec2<f32>(
        (height_ne - height_nw) * (1. - fpos.y) + (height_se - height_sw) * fpos.y,
        (height_sw - height_nw) * (1. - fpos.x) + (height_se - height_ne) * fpos.y,
    );

    let height = height_nw * (1. - fpos.x) * (1. - fpos.y)
        + height_ne * fpos.x * (1. - fpos.y)
        + height_sw * (1. - fpos.x) * fpos.y
        + height_se * fpos.x * fpos.y;

    return vec3<f32>(gradient, height);
}


fn erode(pos: vec2<f32>, val: f32) {
    let half_radius = settings.erosion_radius / 2;
    let half_v = vec2<i32>(half_radius, half_radius);
    let cell_factor = 1. / f32(settings.erosion_radius); // how much is given to each cell
    let xy = vec2<i32>(pos);

    for (var y = -half_v.y; y < half_v.y; y++) {
        if (y + xy.y >= 0 && y + xy.y <= i32(settings.heightmap_size)) {
            for (var x = -half_v.x; x < half_v.x; x++) {
                if (x + xy.x >= 0 && x + xy.x <= i32(settings.heightmap_size)) {
                    let c_xy = xy + vec2<i32>(x, y);
                    let i = c_xy.y * i32(settings.heightmap_size) + c_xy.x;

                    let d = max(1. - length(vec2<f32>(f32(x), f32(y)) / f32(half_radius)), 0.);
                    heightmap[i] -= d * val * cell_factor;
                }
            }
        }
    }
}

fn deposit(pos: vec2<f32>, val: f32) {
    let map_size = i32(settings.heightmap_size);
    let ipos = vec2<i32>(pos);
    let fpos = fract(pos);
    let start = ipos.y * map_size + ipos.x;

    heightmap[start] += val * (1. - fpos.x) * (1. - fpos.y);
    heightmap[start + 1] += val * fpos.x * (1. - fpos.y);
    heightmap[start + map_size] += val * (1. - fpos.x) * fpos.y;
    heightmap[start + map_size + 1] += val * fpos.x * fpos.y;
}

@compute @workgroup_size(512, 1, 1)
fn erode_heightmap(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= 4000000u) { return; }
    var droplet = droplets[global_id.x];
    if (droplet.dead > 0.) { return; }

    if (droplet.water <= 0.) { droplet.dead = 1.; return; }
    if (droplet.pos.x < 0. || droplet.pos.x >= settings.heightmap_size
        || droplet.pos.y < 0. || droplet.pos.y >= settings.heightmap_size) {
        droplet.dead = 1.;
        return;
    }

    let gh = gradient_and_height(droplet.pos);
    let new_dir = droplet.dir * settings.erosion_inertia - gh.xy * (1. - settings.erosion_inertia);

    // break out if no direction (droplet on flat surface)
    if (length(new_dir) >= 0.01) {
        droplet.dir = normalize(new_dir);
    } else {
        droplet.dead = 1.0;
        return;
    }


    let new_pos = droplet.pos + droplet.dir;


    let delta = gradient_and_height(new_pos).z - gh.z;
    let c = -delta * droplet.vel * droplet.water * settings.erosion_capacity;
    if (droplet.sediment > c || delta > 0.) {
        let depo = select((droplet.sediment - c) * settings.erosion_deposition, min(delta, droplet.sediment), delta > 0.);
        deposit(droplet.pos, depo);
        droplet.sediment -= depo;
    } else {
        let ero = min((c - droplet.sediment) * settings.erosion_erosion, -delta);
        erode(droplet.pos, ero);
        droplet.sediment += ero;
    }

    droplet.vel = sqrt(max(droplet.vel*droplet.vel + delta * settings.erosion_gravity, 0.));
    droplet.water = droplet.water * (1. - settings.erosion_evaporation);

    droplet.pos = new_pos;
//    droplet.dead = 1.0;
    droplets[global_id.x] = droplet;
//    heightmap[i32(droplet.pos.y) * 1024 + i32(droplet.pos.x)] = delta;
}