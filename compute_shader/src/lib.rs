#![cfg_attr(target_arch = "spirv", no_std)]

#[allow(unused_imports)] // Spir-v compiler will complain if we don't
use spirv_std::num_traits::Float;
use spirv_std::{
    glam::{IVec2, UVec2, UVec3, UVec4, Vec2, Vec3, Vec3Swizzles},
    spirv,
};
//use compute_shader_shared::RasterParameters;
pub const GRID_CELL_SIZE_U32: u32 = 8;
pub const GRID_CELL_SIZE: usize = 8;

use zerocopy::Immutable;
use zerocopy::IntoBytes;

#[repr(C)]
#[derive(IntoBytes, Immutable, Debug)]
pub struct RasterParameters {
    pub raster_dim_size: u32,
    pub height_min: f32,
    pub height_max: f32,
    pub vertex_count: u32,
    pub triangle_count: u32,
}

impl RasterParameters {
    pub fn new(
        raster_dim_size: u32,
        height_min: f32,
        height_max: f32,
        vertex_count: u32,
        triangle_count: u32,
    ) -> Self {
        Self {
            raster_dim_size,
            height_min,
            height_max,
            vertex_count,
            triangle_count,
        }
    }
}

// use compute_shader_shared::{
//    VertexArrays, cell_pixel_x_y_to_index, cell_to_index, AABBValues, RasterParameters, AABB, GRID_CELL_SIZE
// };

// The spir-v compiler is very picky about how we extend/wrap
// types in external crates. traits on a type alias is the only way
// I've found to make this work
pub type AABB = UVec4;

pub trait AABBValues {
    fn min_x(&self) -> u32;
    fn min_y(&self) -> u32;
    fn max_x(&self) -> u32;
    fn max_y(&self) -> u32;
}

impl AABBValues for UVec4 {
    fn min_x(&self) -> u32 {
        self.x
    }

    fn min_y(&self) -> u32 {
        self.y
    }

    fn max_x(&self) -> u32 {
        self.z
    }

    fn max_y(&self) -> u32 {
        self.w
    }
}

// pixel xy, in cell to raster x,y
pub fn cell_pixel_x_y_to_raster_xy(cell_pixel_x: u32, cell_pixel_y: u32, cell: UVec2) -> UVec2 {
    let pixel_x = cell.x * GRID_CELL_SIZE_U32 + cell_pixel_x;
    let pixel_y = cell.y * GRID_CELL_SIZE_U32 + cell_pixel_y;
    UVec2::new(pixel_x, pixel_y)
}

// pixel x,y to raster index
pub fn raster_x_y_to_raster_index(cell_x: u32, cell_y: u32, params: &RasterParameters) -> usize {
    ((cell_y * params.raster_dim_size) + cell_x) as usize
}

// pixel x,y in celll to raster index
pub fn cell_pixel_x_y_to_raster_index(
    cell_pixel_x: u32,
    cell_pixel_y: u32,
    cell: UVec2,
    params: &RasterParameters,
) -> usize {
    let pixel = cell_pixel_x_y_to_raster_xy(cell_pixel_x, cell_pixel_y, cell);
    raster_x_y_to_raster_index(pixel.x, pixel.y, params)
}

// cell index to raster index (top left corner)
// just a special case of pixel_x_y_to_index
// where pixel index is 0,0
pub fn cell_to_raster_index(cell: UVec2, params: &RasterParameters) -> usize {
    let pixel_x = cell.x * GRID_CELL_SIZE_U32;
    let pixel_y = cell.y * GRID_CELL_SIZE_U32;
    raster_x_y_to_raster_index(pixel_x, pixel_y, params)
}
/// end shared

// u8 was not working
//const CW_FLAG: u32 = 0b1; // Bit flag for clockwise winding order
//const DEGENERATE_FLAG: u32 = 0b10; // Bit flag for degenerate triangles

#[spirv(compute(threads(1, 1, 1)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &RasterParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] u_buffer: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] v_buffer: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] h_buffer: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] indices: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] bounding_boxes: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] storage: &mut [f32],
) {
    // rasterise the cell for this thread
    //gen_raster_debug_dump_inputs(
    rasterise_cell(
        params,
        u_buffer,
        v_buffer,
        h_buffer,
        indices,
        bounding_boxes,
        storage,
        global_id,
    );
}

pub fn rasterise_cell(
    params: &RasterParameters,
    u_buffer: &[u32],
    v_buffer: &[u32],
    h_buffer: &[u32],
    i_buffer: &[u32],
    bounding_boxes: &[UVec4],
    storage: &mut [f32],
    global_id: UVec3,
) {
    let cell = global_id.xy();
    // Iterate over each pixel in the cell
    for y in (0..GRID_CELL_SIZE_U32).rev() {
        for x in 0..GRID_CELL_SIZE_U32 {
            // Iterate over the bounding boxes
            for i in 0..bounding_boxes.len() {
                let aabb = &bounding_boxes[i];
                if intersects_cell(aabb, cell) {
                    let triangle_indices =
                        [i_buffer[i * 3], i_buffer[i * 3 + 1], i_buffer[i * 3 + 2]];
                    let v = [
                        UVec3::new(
                            u_buffer[triangle_indices[0] as usize],
                            v_buffer[triangle_indices[0] as usize],
                            h_buffer[triangle_indices[0] as usize],
                        ),
                        UVec3::new(
                            u_buffer[triangle_indices[1] as usize],
                            v_buffer[triangle_indices[1] as usize],
                            h_buffer[triangle_indices[1] as usize],
                        ),
                        UVec3::new(
                            u_buffer[triangle_indices[2] as usize],
                            v_buffer[triangle_indices[2] as usize],
                            h_buffer[triangle_indices[2] as usize],
                        ),
                    ];
                    let v_xy_i: [IVec2; 3] = [
                        v[0].xy().as_ivec2(),
                        v[1].xy().as_ivec2(),
                        v[2].xy().as_ivec2(),
                    ];
                    // // Compute the full area of the triangle
                    let area_full = edge_function(v_xy_i);

                    // // Skip degenerate triangles
                    if area_full == 0 {
                        continue;
                    }

                    // // negative area implies clockwise winding order
                    let is_cw = area_full < 0;

                    // // is the point in the triangle?
                    let p_raster_xy = cell_pixel_x_y_to_raster_xy(x, y, cell);
                    let w = calculate_edge_weights(v_xy_i, p_raster_xy.as_ivec2(), is_cw);
                    if w[0] >= 0 && w[1] >= 0 && w[2] >= 0 {
                        let v_xyz_f: [Vec3; 3] = [v[0].as_vec3(), v[1].as_vec3(), v[2].as_vec3()];
                        let p_raster_index =
                            raster_x_y_to_raster_index(p_raster_xy.x, p_raster_xy.y, params);
                        let height =
                            interpolate_barycentric(v_xyz_f, p_raster_xy.as_vec2(), params)
                                .unwrap();

                        storage[p_raster_index] = height;
                    }
                }
            }
        }
    }
}

// single thread at 0,0,0 -dump inputs into output buffer
// fn gen_raster_debug_dump_inputs(
//     params: &RasterParameters,
//     u_buffer: &[u32],
//     v_buffer: &[u32],
//     h_buffer: &[u32],
//     i_buffer: &[u32],
//     bounding_boxes: &[UVec4],
//     storage: &mut [f32],
//     global_id: UVec3,
// ) {
//     // workgroup size is 1x1x1,

//     if global_id.x == 0 && global_id.y == 0 && global_id.z == 0 {
//         let mut raster_idx: u32 = 0;
//         storage[raster_idx as usize] = params.raster_dim_size as f32;
//         raster_idx += 1;
//         storage[raster_idx as usize] = params.vertex_count as f32;
//         raster_idx += 1;

//         for index in 0..params.vertex_count {
//             storage[raster_idx as usize] = u_buffer[index as usize] as f32;
//             raster_idx += 1;
//         }
//         storage[raster_idx as usize] = params.vertex_count as f32;
//         raster_idx += 1;

//         for index in 0..params.vertex_count {
//             storage[raster_idx as usize] = v_buffer[index as usize] as f32;
//             raster_idx += 1;
//         }
//         storage[raster_idx as usize] = params.vertex_count as f32;
//         raster_idx += 1;

//         for index in 0..params.vertex_count {
//             storage[raster_idx as usize] = h_buffer[index as usize] as f32;
//             raster_idx += 1;
//         }
//         storage[raster_idx as usize] = params.triangle_count as f32;
//         raster_idx += 1;

//         for index in 0..params.triangle_count * 3 {
//             storage[raster_idx as usize] = i_buffer[index as usize] as f32;
//             raster_idx += 1;
//         }
//         storage[raster_idx as usize] = params.triangle_count as f32;
//         raster_idx += 1;
//         for index in 0..params.triangle_count {
//             storage[raster_idx as usize] = bounding_boxes[index as usize].x as f32;
//             raster_idx += 1;
//             storage[raster_idx as usize] = bounding_boxes[index as usize].y as f32;
//             raster_idx += 1;
//             storage[raster_idx as usize] = bounding_boxes[index as usize].z as f32;
//             raster_idx += 1;
//             storage[raster_idx as usize] = bounding_boxes[index as usize].w as f32;
//             raster_idx += 1;

//             raster_idx += 1;
//         }
//         storage[raster_idx as usize] = raster_idx as f32;
//     }
// }

fn intersects_cell(cell_aabb: &AABB, cell: UVec2) -> bool {
    cell_aabb.min_x() <= cell.x + GRID_CELL_SIZE_U32
        && cell_aabb.max_x() >= cell.x
        && cell_aabb.min_y() <= cell.y + GRID_CELL_SIZE_U32
        && cell_aabb.max_y() >= cell.y
}

// Only needed on host
// #[cfg(not(target_arch = "spirv"))]
// pub fn rasterise_triangle(
//     params: &RasterParameters,
//     u_buffer: &[u32],
//     v_buffer: &[u32],
//     h_buffer: &[u32],
//     indices: &[u32],
//     storage: &mut [f32],
//     index: usize,
// ) {
//     use compute_shader_shared::calculate_triangle_aabb;

//     let triangle_indices: [usize; 3] = [indices[index * 3] as usize, indices[index * 3 + 1] as usize,  indices[index * 3 + 2] as usize];

//         let vertices = [
//                                     UVec3::new(
//                                         u_buffer[triangle_indices[0] as usize],
//                                         v_buffer[triangle_indices[0] as usize],
//                                         h_buffer[triangle_indices[0] as usize],
//                                     ),
//                                     UVec3::new(
//                                         u_buffer[triangle_indices[1] as usize],
//                                         v_buffer[triangle_indices[1] as usize],
//                                         h_buffer[triangle_indices[1] as usize],
//                                     ),
//                                     UVec3::new(
//                                         u_buffer[triangle_indices[2] as usize],
//                                         v_buffer[triangle_indices[2] as usize],
//                                         h_buffer[triangle_indices[2] as usize],
//                                     ),
//                                 ];

//     let aabb: AABB = calculate_triangle_aabb(&vertices, &triangle_indices);

//     // invert the y axis line order
//     for y in (aabb.min_y()..=aabb.max_y()).rev() {
//         for x in aabb.min_x()..=aabb.max_x() {
//             // index in the flat raster
//             let raster_idx = ((y * params.raster_dim_size) + x) as usize;

//             // Check if the raster cell is empty by reading the atomic value without locking
//             if let Some(value) =
//                 triangle_face_height_interpolator(UVec2::new(x, y), [vertices[0], vertices[1], vertices[2]], params)
//             {
//                 // this invites a race condition as multiple threads can write to the same cell though in
//                 // theory they should be writing the same value
//                 storage[raster_idx] = value;
//             }
//         }
//     }
// }

// Is v2 inside the edge formed by v0 and v1
pub fn edge_function(v: [IVec2; 3]) -> i32 {
    (v[2].x - v[0].x) * (v[1].y - v[0].y) - (v[2].y - v[0].y) * (v[1].x - v[0].x)
}

// Calculate the edge weights for a point p
pub fn calculate_edge_weights(v: [IVec2; 3], p: IVec2, is_cw: bool) -> [i32; 3] {
    if is_cw {
        [
            edge_function([v[1], v[0], p]),
            edge_function([v[2], v[1], p]),
            edge_function([v[1], v[0], p]),
        ]
    } else {
        [
            edge_function([v[0], v[1], p]),
            edge_function([v[1], v[2], p]),
            edge_function([v[2], v[0], p]),
        ]
    }
}

// Is the point p inside the triangle formed by vertices v
// Uses orientation of 3 edges to determine if the point is inside the triangle
pub fn point_in_triangle(v: [UVec3; 3], p: UVec2) -> bool {
    // Get the xy components of the vertices
    //let v_xy = v.map(|v| v.xy().as_ivec2());
    //RJ - cannot cast between pointer types -- spirv
    let v_xy: [IVec2; 3] = [
        v[0].xy().as_ivec2(),
        v[1].xy().as_ivec2(),
        v[2].xy().as_ivec2(),
    ];

    // Calculate the full area of the triangle
    let area_full = edge_function(v_xy);

    // If the area is zero, the triangle is degenerate
    if area_full == 0 {
        return false;
    }

    // Determine winding order
    let is_cw = area_full < 0;

    // Calculate edge weights
    let w = calculate_edge_weights(v_xy, p.as_ivec2(), is_cw);

    w[0] >= 0 && w[1] >= 0 && w[2] >= 0
}

// Calculate the barycentric weights for a point p
pub fn calculate_barycentric_weights(v: [Vec2; 3], p: Vec2) -> [f32; 3] {
    //area of the sub-triangles formed by the vertices and point p
    let area_abc = f32::abs((v[1] - v[0]).perp_dot(v[2] - v[0]));
    let area_pbc = f32::abs((v[1] - p).perp_dot(v[2] - p));
    let area_pca = f32::abs((v[2] - p).perp_dot(v[0] - p));
    let area_pab = f32::abs((v[0] - p).perp_dot(v[1] - p));
    [
        area_pbc / area_abc,
        area_pca / area_abc,
        area_pab / area_abc,
    ]
}

// Interpolate the height of a point p inside the triangle formed by vertices v
pub fn interpolate_barycentric(v: [Vec3; 3], p: Vec2, params: &RasterParameters) -> Option<f32> {
    // RJ - error casting pointers spirv
    //let wb = calculate_barycentric_weights(v.map(|v| v.xy()), p);
    let v_xy: [Vec2; 3] = [v[0].xy(), v[1].xy(), v[2].xy()];
    let wb = calculate_barycentric_weights(v_xy, p);

    // Not checking weights as we have already decided that the point is inside the triangle
    let numerator = wb[0] * v[0].z + wb[1] * v[1].z + wb[2] * v[2].z; // 131068

    // Normalize and map the height. Unmapped range is 0-32767
    let normalized_height = numerator / 32767.;
    let mapped_height =
        params.height_min + normalized_height * (params.height_max - params.height_min);
    Some(mapped_height)
}

// Check if the point is inside the triangle (all weights must be non-negative)
// using integer edge function weights. This avoids floating point precision issues.
// Interpolate the z value using barycentric coordinates only if the point
// is determined to be inside the triangle
pub fn triangle_face_height_interpolator(
    p: UVec2,
    v: [UVec3; 3],
    params: &RasterParameters,
) -> Option<f32> {
    if point_in_triangle(v, p) {
        // Interpolate the z value using barycentric coordinates
        // Ideally work in double precision and reduce for output
        // As of now, f64 seems to be broken in rust-gpu

        // -- Error casting pointers in spirv compiler
        interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), params)
    } else {
        None
    }
}

#[cfg(test)]

mod tests {

    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_edge_function_colinear() {
        // Point on the edge
        let v0 = IVec2::new(0, 0);
        let v1 = IVec2::new(4, 4);
        let v2 = IVec2::new(2, 2);
        let result = edge_function([v0, v1, v2]);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_edge_function_left_of_edge() {
        // Point to the left of the edge
        let v0 = IVec2::new(0, 0);
        let v1 = IVec2::new(4, 4);
        let v2 = IVec2::new(1, 3);
        let result = edge_function([v0, v1, v2]);
        assert_eq!(result, -8);
    }

    #[test]
    fn test_edge_function_right_of_edge() {
        // Point to the right of the edge
        let v0 = IVec2::new(0, 0);
        let v1 = IVec2::new(4, 4);
        let v2 = IVec2::new(3, 1);
        let result = edge_function([v0, v1, v2]);
        assert_eq!(result, 8);
    }

    #[test]
    fn test_calculate_weights_inside_triangle() {
        // Note the CCW winding order
        let v0 = IVec2::new(0, 0);
        let v1 = IVec2::new(0, 3);
        let v2 = IVec2::new(3, 0);
        let p = IVec2::new(1, 1);

        let w = calculate_edge_weights([v0, v1, v2], p, false);
        assert_eq!(w[0], 3);
        assert_eq!(w[1], 3);
        assert_eq!(w[2], 3);
    }

    #[test]
    fn test_calculate_weights_inside_triangle_cw_winding() {
        // Note the CCW winding order
        let v0 = IVec2::new(0, 0);
        let v1 = IVec2::new(3, 0);
        let v2 = IVec2::new(0, 3);
        let p = IVec2::new(1, 1);

        let w = calculate_edge_weights([v0, v1, v2], p, true);
        assert_eq!(w[0], 3);
        assert_eq!(w[1], 3);
        assert_eq!(w[2], 3);
    }

    #[test]
    fn test_calculate_weights_outside_triangle() {
        let v0 = IVec2::new(0, 0);
        let v1 = IVec2::new(0, 2);
        let v2 = IVec2::new(2, 0);
        let p = IVec2::new(3, 3);

        let w = calculate_edge_weights([v0, v1, v2], p, false);
        assert_eq!(w[0], 6);
        assert_eq!(w[1], -8);
        assert_eq!(w[2], 6);
    }

    #[test]
    fn test_calculate_weights_outside_triangle_cw_winding() {
        let v0 = IVec2::new(0, 0);
        let v2 = IVec2::new(0, 2);
        let v1 = IVec2::new(2, 0);
        let p = IVec2::new(3, 3);

        let w = calculate_edge_weights([v0, v1, v2], p, true);
        assert_eq!(w[0], 6);
        assert_eq!(w[1], -8);
        assert_eq!(w[2], 6);
    }

    #[test]
    fn test_calculate_weights_on_edge() {
        let v0 = IVec2::new(0, 0);
        let v1 = IVec2::new(0, 2);
        let v2 = IVec2::new(2, 0);
        let p = IVec2::new(1, 1);

        let w = calculate_edge_weights([v0, v1, v2], p, false);
        assert_eq!(w[0], 2);
        assert_eq!(w[1], 0);
        assert_eq!(w[2], 2);
    }

    #[test]
    fn test_calculate_weights_at_vertex() {
        let v0 = IVec2::new(0, 0);
        let v1 = IVec2::new(0, 2);
        let v2 = IVec2::new(2, 0);
        let p = IVec2::new(0, 0);

        let w = calculate_edge_weights([v0, v1, v2], p, false);
        assert_eq!(w[0], 0);
        assert_eq!(w[1], 4);
        assert_eq!(w[2], 0);
    }

    #[test]
    fn test_calculate_weights_epsilon_outside() {
        //this point is outside the triangle but is determined to be inside due to floating point precision
        //edge method correctly determines it to be outside
        let v0 = IVec2::new(0, 0);
        let v1 = IVec2::new(0, 32767);
        let v2 = IVec2::new(1, 16384);
        let p = IVec2::new(1, 16383);

        let w = calculate_edge_weights([v0, v1, v2], p, false);
        assert_eq!(w[0], 32767);
        assert_eq!(w[1], 1);
        assert_eq!(w[2], -1);
    }

    #[test]
    fn test_barycentric_weights_inside_triangle() {
        let v0 = Vec2::new(0., 0.);
        let v1 = Vec2::new(0., 3.);
        let v2 = Vec2::new(3., 0.);
        let p = Vec2::new(1., 1.);

        let w = calculate_barycentric_weights([v0, v1, v2], p);
        assert_eq!(w[0], 1. / 3.);
        assert_eq!(w[1], 1. / 3.);
        assert_eq!(w[2], 1. / 3.);
    }

    #[test]
    fn test_barycentric_weights_on_edge() {
        let v0 = Vec2::new(0., 0.);
        let v1 = Vec2::new(0., 3.);
        let v2 = Vec2::new(3., 0.);
        let p = Vec2::new(1., 0.);
        let w = calculate_barycentric_weights([v0, v1, v2], p);
        assert_eq!(w[0], 2. / 3.);
        assert_eq!(w[1], 0.);
        assert_eq!(w[2], 1. / 3.);
    }

    #[test]
    fn test_barycentric_weights_epsilon_outside() {
        //this point is outside the triangle but is determined to be inside due to floating point precision
        let v0 = Vec2::new(0., 0.);
        let v1 = Vec2::new(0., 32767.);
        let v2 = Vec2::new(1., 16384.);
        let p = Vec2::new(1., 16383.);

        let w = calculate_barycentric_weights([v0, v1, v2], p);
        assert_abs_diff_eq!(w[0], 0., epsilon = 1e-4);
        assert_abs_diff_eq!(w[1], 0., epsilon = 1e-4);
        assert_abs_diff_eq!(w[2], 1., epsilon = 1e-5);
    }

    #[test]
    fn test_barycentric_weights_at_vertex() {
        let v0 = Vec2::new(0., 0.);
        let v1 = Vec2::new(2., 0.);
        let v2 = Vec2::new(0., 2.);
        let p = Vec2::new(0., 0.);

        let w = calculate_barycentric_weights([v0, v1, v2], p);
        assert_eq!(w[0], 1.);
        assert_eq!(w[1], 0.);
        assert_eq!(w[2], 0.);
    }

    #[test]
    fn test_barycentric_weights_outside_triangle() {
        let v0 = Vec2::new(0., 0.);
        let v1 = Vec2::new(2., 0.);
        let v2 = Vec2::new(0., 2.);
        let p = Vec2::new(5., 5.);

        let w = calculate_barycentric_weights([v0, v1, v2], p);
        assert_eq!(w[0], 4.);
        assert_eq!(w[1], 2.5);
        assert_eq!(w[2], 2.5);
    }

    #[test]
    fn test_shader_edge_interpolator_inside_triangle_zero_plane() {
        let p = UVec2::new(1, 1);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(0, 3, 0),
            UVec3::new(3, 0, 0),
        ];
        let params: RasterParameters = RasterParameters::new(100, 0., 1., 0, 0);

        let result = triangle_face_height_interpolator(p, v, &params);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 0.);
    }

    #[test]
    fn test_shader_edge_interpolator_inside_triangle_different_z() {
        let params = RasterParameters::new(100, 0., 1., 0, 0);

        let p = UVec2::new(1, 1);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(0, 3, 32767),
            UVec3::new(3, 0, 16383),
        ];

        let result = triangle_face_height_interpolator(p, v, &params);
        assert!(result.is_some());
        assert_abs_diff_eq!(result.unwrap(), 0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_shader_edge_interpolator_on_edge_different_z() {
        let params = RasterParameters::new(100, 0., 1., 0, 0);

        let p = UVec2::new(1, 1);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(0, 2, 32767),
            UVec3::new(2, 0, 16383),
        ];

        let result = triangle_face_height_interpolator(p, v, &params);
        assert!(result.is_some());
        assert_abs_diff_eq!(result.unwrap(), 0.75, epsilon = 1e-5);
    }

    #[test]
    fn test_shader_edge_interpolator_outside_triangle() {
        let params = RasterParameters::new(100, 0., 1., 0, 0);

        let p = UVec2::new(3, 3);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(2, 0, 32767),
            UVec3::new(0, 2, 32767),
        ];

        let result = triangle_face_height_interpolator(p, v, &params);
        assert!(result.is_none());
    }

    #[test]
    fn test_shader_edge_interpolator_at_vertex_different_z() {
        let params = RasterParameters::new(100, 0., 1., 0, 0);

        let p = UVec2::new(0, 0);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(0, 2, 32767),
            UVec3::new(2, 0, 16383),
        ];

        let result = triangle_face_height_interpolator(p, v, &params);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 0.);
    }

    #[test]
    fn test_shader_edge_interpolator_on_edge() {
        let params = RasterParameters::new(100, 0., 1., 0, 0);

        let p = UVec2::new(1, 0);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(2, 0, 32767),
            UVec3::new(0, 2, 32767),
        ];

        let result = triangle_face_height_interpolator(p, v, &params);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 0.5);
    }

    #[test]
    fn test_shader_edge_interpolator_at_vertex() {
        let params = RasterParameters::new(100, 0., 1., 0, 0);

        let p = UVec2::new(0, 0);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(2, 0, 32767),
            UVec3::new(0, 2, 32767),
        ];

        let result = triangle_face_height_interpolator(p, v, &params);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 0.);
    }

    #[test]
    fn test_shader_edge_interpolator_degenerate_triangle() {
        let params = RasterParameters::new(100, 0., 1., 0, 0);

        let p = UVec2::new(1, 1);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(0, 0, 32767),
            UVec3::new(0, 0, 32767),
        ];

        let result = triangle_face_height_interpolator(p, v, &params);
        assert!(result.is_none());
    }
}
