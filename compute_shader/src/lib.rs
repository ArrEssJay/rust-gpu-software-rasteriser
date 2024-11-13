#![cfg_attr(target_arch = "spirv", no_std)]

use bytemuck::{Pod, Zeroable};
#[allow(unused_imports)] // Spir-v compiler will complain if we don't
use spirv_std::num_traits::Float;
use spirv_std::{
    arch,
    glam::{IVec2, UVec2, UVec3, UVec4, Vec2, Vec3, Vec3Swizzles, Vec4Swizzles},
    spirv,
};

// Importing from external crates is problematic due
// to the spir-v compiler. Hence we define some of the
// constants and structures here, despite this not
// necessarily being the best practice
pub const GRID_CELL_SIZE_U32: u32 = 8;
pub const GRID_CELL_SIZE: usize = 8;

// Shared memory for per-cell vertices
pub const MAX_CELL_TRIANGLES: usize = 8;
pub type CellData = [[UVec3; 3]; MAX_CELL_TRIANGLES];

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
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

// The spir-v compiler is very picky about how we extend/wrap
// types in external crates. traits on a type alias is the only way
// I've found to make this work
pub type AABB = UVec4;

pub trait AABBValues {
    fn min_x(&self) -> u32;
    fn min_y(&self) -> u32;
    fn max_x(&self) -> u32;
    fn max_y(&self) -> u32;

    fn min(&self) -> UVec2;
    fn max(&self) -> UVec2;
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

    fn min(&self) -> UVec2 {
        self.xy()
    }

    fn max(&self) -> UVec2 {
        self.zw()
    }
}

// For a given bounding box and cell, check if the cell intersects the bounding box
fn intersects_cell(cell_aabb: &AABB, cell: UVec2) -> bool {
    cell_aabb.min_x() <= cell.x + GRID_CELL_SIZE_U32
        && cell_aabb.max_x() >= cell.x
        && cell_aabb.min_y() <= cell.y + GRID_CELL_SIZE_U32
        && cell_aabb.max_y() >= cell.y
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
    pixel_x: u32,
    pixel_y: u32,
    cell: UVec2,
    params: &RasterParameters,
) -> usize {
    let pixel = cell_pixel_x_y_to_raster_xy(pixel_x, pixel_y, cell);
    raster_x_y_to_raster_index(pixel.x, pixel.y, params)
}

pub fn cell_pixel_to_raster_index(pixel: UVec2, cell: UVec2, params: &RasterParameters) -> usize {
    cell_pixel_x_y_to_raster_index(pixel.x, pixel.y, cell, params)
}

// cell index to raster index (top left corner)
// just a special case of pixel_x_y_to_index
// where pixel index is 0,0
pub fn cell_to_raster_index(cell: UVec2, params: &RasterParameters) -> usize {
    let pixel_x = cell.x * GRID_CELL_SIZE_U32;
    let pixel_y = cell.y * GRID_CELL_SIZE_U32;
    raster_x_y_to_raster_index(pixel_x, pixel_y, params)
}

// Spir-v entry point for aabb calculation
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(1)))]
pub fn spirv_compute_aabb(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] _params: &RasterParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] u_buffer: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] v_buffer: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] _h_buffer: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] indices: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] aabb: &mut [AABB],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] _storage: &mut [f32],
    #[spirv(workgroup)] _cell_data: &mut CellData,
) {
    //n-triangles x 1x1x1 workgroups will be dispatched
    aabb[global_id.x as usize] = compute_aabb(global_id.x as usize, u_buffer, v_buffer, indices);
}

// Compute the axis aligned bounding box for a triangle
pub fn compute_aabb(
    triangle_index: usize,
    u_buffer: &[u32],
    v_buffer: &[u32],
    indices: &[u32],
) -> UVec4 {
    // 3 vertices per triangle
    let i0 = indices[triangle_index * 3] as usize;
    let i1 = indices[triangle_index * 3 + 1] as usize;
    let i2 = indices[triangle_index * 3 + 2] as usize;

    // Extract x and y components of each point
    let x0 = u_buffer[i0];
    let y0 = v_buffer[i0];
    let x1 = u_buffer[i1];
    let y1 = v_buffer[i1];
    let x2 = u_buffer[i2];
    let y2 = v_buffer[i2];

    // Compute min_x and max_x manually
    // TODO: There is an issue with the glam uvec min/max function in spir-v
    // requiring i8 support - needs investigation
    let min_x = if x0 < x1 { if x0 < x2 { x0 } else { x2 } } else if x1 < x2 { x1 } else { x2 };
    let max_x = if x0 > x1 { if x0 > x2 { x0 } else { x2 } } else if x1 > x2 { x1 } else { x2 };

    let min_y = if y0 < y1 { if y0 < y2 { y0 } else { y2 } } else if y1 < y2 { y1 } else { y2 };
    let max_y = if y0 > y1 { if y0 > y2 { y0 } else { y2 } } else if y1 > y2 { y1 } else { y2 };

    UVec4::new(min_x, min_y, max_x, max_y)
}

#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(8, 8, 1)))]
pub fn spirv_rasterise(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &RasterParameters,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] u_buffer: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] v_buffer: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] h_buffer: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] indices: &[u32],
    // aabb is mapped RW below simply to avoid the need to create a separate read only binding - it is not written to
    // TODO: Either disable this validation or map R/O
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] aabb: &mut [AABB], 
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] storage: &mut [f32],
    #[spirv(workgroup)] cell_data: &mut CellData,
) {
    // Designate thread (0,0,0) to load data into shared memory
    if local_id.x == 0 && local_id.y == 0 && local_id.z == 0 {
        // Load the vertices for this cell into shared memory
        load_cell_triangles(
            u_buffer,
            v_buffer,
            h_buffer,
            indices,
            aabb,
            workgroup_id.xy(),
            cell_data,
        );
    }

    // Wait here until the vertices are loaded into shared memory
    unsafe {
        arch::workgroup_memory_barrier_with_group_sync();
    }

    // rasterise the pixel and write to raster
    // the raster write is done here to allow the software rasteriser to use the same
    // code but handle parallelisation and safe access of shared memory
    let raster_index =
        cell_pixel_x_y_to_raster_index(local_id.x, local_id.y, workgroup_id.xy(), params);

    // Will return an invalid value if the pixel is outside the cell
    let val = rasterise_pixel(params, cell_data, workgroup_id.xy(), local_id.xy());
    if val > params.height_min {
        storage[raster_index] = val;
    }
}

pub fn load_cell_triangles(
    u_buffer: &[u32],
    v_buffer: &[u32],
    h_buffer: &[u32],
    indices: &[u32],
    bounding_boxes: &[AABB],
    cell: UVec2,
    cell_data: &mut CellData,
) {
    for i in 0..bounding_boxes.len() {
        if i >= MAX_CELL_TRIANGLES {
            // We can't panic so just return with nothing rendered for this cell
            // This lacks finesse but for this implementation it will suffice
            return;
        }
        let aabb = &bounding_boxes[i];
        if intersects_cell(aabb, cell) {
            let triangle_indices = [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]];
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
            cell_data[i] = v;
            //cell_data.triangle_count += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rasterise_pixel(
    raster_parameters: &RasterParameters,
    cell_data: &CellData,
    cell: UVec2,
    pixel: UVec2,
) -> f32 {
    // Iterate over the shared memory to rasterise the pixel
    // There are likely less than MAX_CELL_TRIANGLES triangles in the shared memory
    // and we will break out of the loop once we find the triangle that contains the pixel

    #[allow(clippy::needless_range_loop)]
    for i in 0..MAX_CELL_TRIANGLES {
        let v = cell_data[i];

        let v_xy_i: [IVec2; 3] = [
            v[0].xy().as_ivec2(),
            v[1].xy().as_ivec2(),
            v[2].xy().as_ivec2(),
        ];
        // Compute the full area of the triangle
        let area_full = edge_function(v_xy_i);

        // Skip degenerate triangles
        if area_full == 0 {
            continue;
        }

        // Negative area implies clockwise winding order
        let is_cw = area_full < 0;

        // Check if the point is in the triangle
        let p_raster_xy = cell_pixel_x_y_to_raster_xy(pixel.x, pixel.y, cell);
        let w = calculate_edge_weights(v_xy_i, p_raster_xy.as_ivec2(), is_cw);

        if w[0] >= 0 && w[1] >= 0 && w[2] >= 0 {
            let v_xyz_f: [Vec3; 3] = [v[0].as_vec3(), v[1].as_vec3(), v[2].as_vec3()];
            let height =
                interpolate_barycentric(v_xyz_f, p_raster_xy.as_vec2(), raster_parameters).unwrap();
            return height;
        }
    }
    // If no triangle contains the pixel, return NAN
    raster_parameters.height_min - 1.0
}

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
    // let v_xy = v.map(|v| v.xy().as_ivec2());
    // RJ - cannot cast between pointer types -- spirv
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
    // Negative area implies clockwise winding order
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

// These tests will be built and run for the host target only
#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_compute_aabb_basic() {
        let u_buffer = vec![63, 0, 0];
        let v_buffer = vec![0, 63, 0];
        let indices = vec![0, 1, 2];

        let result = compute_aabb(0, &u_buffer, &v_buffer, &indices);
        let expected = UVec4::new(0, 0, 63, 63);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_aabb_basic_b() {
        let u_buffer = vec![0, 0, 63];
        let v_buffer = vec![0, 63, 0];
        let indices = vec![0, 1, 2];

        let result = compute_aabb(0, &u_buffer, &v_buffer, &indices);
        let expected = UVec4::new(0, 0, 63, 63);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_aabb_second_triangle() {
        let u_buffer = vec![63, 0, 0, 63];
        let v_buffer = vec![0, 63, 0, 63];
        let indices = vec![0, 1, 2, 1, 2, 3];

        let result1 = compute_aabb(0, &u_buffer, &v_buffer, &indices);
        let expected = UVec4::new(0, 0, 63, 63);
        assert_eq!(result1, expected);

        let result2 = compute_aabb(1, &u_buffer, &v_buffer, &indices);
        assert_eq!(result2, expected);
    }

    #[test]
    fn test_compute_aabb_single_point() {
        let u_buffer = vec![10, 10, 10];
        let v_buffer = vec![10, 10, 10];
        let indices = vec![0, 1, 2];

        let result = compute_aabb(0, &u_buffer, &v_buffer, &indices);
        let expected = UVec4::new(10, 10, 10, 10);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_aabb_two_points() {
        let u_buffer = vec![10, 10, 20];
        let v_buffer = vec![10, 10, 20];
        let indices = vec![0, 1, 2];

        let result = compute_aabb(0, &u_buffer, &v_buffer, &indices);
        let expected = UVec4::new(10, 10, 20, 20);

        assert_eq!(result, expected);
    }

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

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 0.);
    }

    #[test]
    fn test_shader_edge_interpolator_inside_triangle_one_plane() {
        let p = UVec2::new(1, 1);
        let v = [
            UVec3::new(0, 0, 32767),
            UVec3::new(0, 127, 32767),
            UVec3::new(127, 0, 32767),
        ];
        let params: RasterParameters = RasterParameters::new(128, 0., 1., 0, 0);

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert!(result.is_some());
        assert_abs_diff_eq!(result.unwrap(), 1.0, epsilon = 1e-5);
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

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
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

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
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

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
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

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
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

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
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

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
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

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert!(result.is_none());
    }
}
