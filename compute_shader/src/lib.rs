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
pub const GRID_CELL_THREADS: usize = GRID_CELL_SIZE * GRID_CELL_SIZE;

// Shared memory for per-cell vertices
pub const MAX_CELL_TRIANGLES: usize = 32;
pub type CellData = [[UVec3; 3]; MAX_CELL_TRIANGLES];

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RasterParameters {
    scaled_raster_size: u32,
    pub raster_scale_factor: u32,
    pub raster_native_max: u32,
    pub attribute_f_min: f32,
    pub attribute_f_max: f32,
    pub attribute_u_max: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
}

impl RasterParameters {
    pub fn new(
        raster_scale_factor: u32,
        raster_native_max: u32,
        attribute_f_min: f32,
        attribute_f_max: f32,
        attribute_u_max: u32,
        vertex_count: u32,
        triangle_count: u32,
    ) -> Self {
        let scaled_raster_size = Self::_scaled_raster_size(raster_scale_factor, raster_native_max);
        if (scaled_raster_size % GRID_CELL_SIZE_U32) != 0 {
            panic!("Scaled raster size: {} % {} = {} (!=0)", scaled_raster_size, GRID_CELL_SIZE_U32, scaled_raster_size % GRID_CELL_SIZE_U32);
        }

        Self {
            scaled_raster_size,
            raster_scale_factor,
            raster_native_max,
            attribute_f_min,
            attribute_f_max,
            attribute_u_max,
            vertex_count,
            triangle_count,
        }
    }
    fn _scaled_raster_size(raster_scale_factor: u32, raster_native_max: u32) -> u32 {
        (raster_native_max + 1) >> raster_scale_factor
    }
    pub fn scaled_raster_size(&self) -> u32 {
        self.scaled_raster_size
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

    fn new_aabb(min_x: u32, min_y: u32, max_x: u32, max_y: u32) -> AABB;
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
    fn new_aabb(min_x: u32, min_y: u32, max_x: u32, max_y: u32) -> Self {
        UVec4::new(min_x, min_y, max_x, max_y)
    }
}

// For a given bounding box (raster space) and cell position (cell space)
// check if the bounding box intersects the cell
// All bounding box edges are considered to be inside the cell as any vertices
// that lie on this edge could form triangles in the cell
fn intersects_cell(aabb: &AABB, cell: UVec2) -> bool {
    // Min and max (x,y) of the cell in raster space
    let cell_ul = cell_to_raster_pixel(cell);
    let cell_lr = cell_ul + GRID_CELL_SIZE_U32;

    // Min and max (x,y) of the bounding box in raster space
    let aabb_ul = aabb.min();
    let aabb_lr = aabb.max();

    // Check for any overlap between the cell's bounding box and aabb, including edges
    cell_ul.x <= aabb_lr.x
        && cell_lr.x >= aabb_ul.x
        && cell_ul.y <= aabb_lr.y
        && cell_lr.y >= aabb_ul.y
}

// Various methods to convert between cell and raster pixel space
// No bounds checking is performed

pub fn cell_pixel_to_raster_pixel(cell_pixel: UVec2, cell: UVec2) -> UVec2 {
    let cell_ul_raster_pixel = cell * GRID_CELL_SIZE_U32;
    cell_ul_raster_pixel + cell_pixel
}

pub fn cell_to_raster_pixel(cell: UVec2) -> UVec2 {
    cell * GRID_CELL_SIZE_U32
}

pub fn raster_pixel_to_raster_index(pixel: UVec2, params: &RasterParameters) -> usize {
    // reverse scan line order in raster space
    let adjusted_y = params.scaled_raster_size() - 1 - pixel.y;
    ((adjusted_y * params.scaled_raster_size()) + pixel.x) as usize
}

pub fn cell_pixel_to_raster_index(
    cell_pixel: UVec2,
    cell: UVec2,
    params: &RasterParameters,
) -> usize {
    raster_pixel_to_raster_index(cell_pixel_to_raster_pixel(cell_pixel, cell), params)
}

pub fn cell_to_raster_index(cell: UVec2, params: &RasterParameters) -> usize {
    raster_pixel_to_raster_index(cell_to_raster_pixel(cell), params)
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
    let min_x = if x0 < x1 {
        if x0 < x2 {
            x0
        } else {
            x2
        }
    } else if x1 < x2 {
        x1
    } else {
        x2
    };
    let max_x = if x0 > x1 {
        if x0 > x2 {
            x0
        } else {
            x2
        }
    } else if x1 > x2 {
        x1
    } else {
        x2
    };

    let min_y = if y0 < y1 {
        if y0 < y2 {
            y0
        } else {
            y2
        }
    } else if y1 < y2 {
        y1
    } else {
        y2
    };
    let max_y = if y0 > y1 {
        if y0 > y2 {
            y0
        } else {
            y2
        }
    } else if y1 > y2 {
        y1
    } else {
        y2
    };

    UVec4::new(min_x, min_y, max_x, max_y)
}

#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(8, 8, 1)))]
pub fn spirv_rasterise(
    #[spirv(workgroup_id)] cell: UVec3,
    #[spirv(local_invocation_index)] cell_thread: u32,
    #[spirv(local_invocation_id)] cell_pixel: UVec3,
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
    // Discard the z component of the 3D workgroup & workgroup thread
    let cell_pixel: UVec2 = cell_pixel.xy();
    let cell: UVec2 = cell.xy();


    if cell_thread == 0 {
        load_cell_triangles(
            u_buffer,
            v_buffer,
            h_buffer,
            indices,
            aabb,
            cell,
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
    let raster_index = cell_pixel_to_raster_index(cell_pixel, cell, params);

    // Will return an invalid value if the pixel is outside the cell
    let val = rasterise_cell_pixel(params, cell_data, cell, cell_pixel);
    if val.is_some() {
        storage[raster_index] = val.unwrap();
    }
}

#[allow(clippy::too_many_arguments)]
pub fn load_cell_triangles(
    u_buffer: &[u32],
    v_buffer: &[u32],
    h_buffer: &[u32],
    indices: &[u32],
    aabb: &[AABB],
    cell: UVec2,
    cell_data: &mut CellData,
) {

    // Pre-initialize cell_data with u32::MAX
    for i in 0..MAX_CELL_TRIANGLES {
        cell_data[i] = [
            UVec3::new(u32::MAX, u32::MAX, u32::MAX),
            UVec3::new(u32::MAX, u32::MAX, u32::MAX),
            UVec3::new(u32::MAX, u32::MAX, u32::MAX),
        ];
    }

    let mut cell_triangle_idx: usize = 0;

    for i in 0..aabb.len() {
        // If we have already loaded the maximum number of triangles for the cell, break
        if cell_triangle_idx == MAX_CELL_TRIANGLES {
            break;
        }
        if intersects_cell( &aabb[i], cell) {
            let idx = i * 3;
            let triangle_indices = [indices[idx], indices[idx + 1], indices[idx + 2]];
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

            cell_data[cell_triangle_idx] = v;
            cell_triangle_idx += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rasterise_cell_pixel(
    raster_parameters: &RasterParameters,
    cell_data: &CellData,
    cell: UVec2,
    cell_pixel: UVec2,
) -> Option<f32> {
    // Iterate over the shared memory to rasterise the pixel
    // There are likely less than MAX_CELL_TRIANGLES triangles in the shared memory
    // and we will break out of the loop once we find the triangle that contains the pixel

    #[allow(clippy::needless_range_loop)]
    for i in 0..MAX_CELL_TRIANGLES {
        let v = cell_data[i];

        // cell_data is initialised with u32::MAX, so we can break as soon
        // as we encounter it 
        if v[0].x == u32::MAX {
            return None;
        }

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

        // Calculate position of the pixel in raster space
        let raster_pixel = cell_pixel_to_raster_pixel(cell_pixel, cell);

        // Calculate edge weights for the pixel. Winding order is used to ensure
        // that a positive weight indicates the point is on the in side of the edge regardless
        // of the sign of the triangle face normal
        let w = calculate_edge_weights(v_xy_i, raster_pixel.as_ivec2(), is_cw);

        // If all weights are positive, the point is inside the triangle
        if w[0] >= 0 && w[1] >= 0 && w[2] >= 0 {
            // interpolation is performed in single-precision floating point
            let v_xyz_f: [Vec3; 3] = [v[0].as_vec3(), v[1].as_vec3(), v[2].as_vec3()];                
            return Some(interpolate_barycentric(v_xyz_f, raster_pixel.as_vec2(), raster_parameters))
        }
    }
    // If no triangle contains the pixel, return NAN
    None
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
            edge_function([v[0], v[2], p]),
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

// Interpolate the attribute of a point p inside the triangle formed by vertices v
pub fn interpolate_barycentric(v: [Vec3; 3], p: Vec2, params: &RasterParameters) -> f32 {
    // RJ - error casting pointers spirv
    //let wb = calculate_barycentric_weights(v.map(|v| v.xy()), p);
    let v_xy: [Vec2; 3] = [v[0].xy(), v[1].xy(), v[2].xy()];
    let wb = calculate_barycentric_weights(v_xy, p);

    // Not checking weights as we have already decided that the point is inside the triangle
    let numerator = wb[0] * v[0].z + wb[1] * v[1].z + wb[2] * v[2].z; // 131068

    // Normalize and map the attribute. Unmapped range is 0..attribute_u_max
    // Mapped range is attribute_f_min..attribute_f_max
    let normalized_attribute = numerator / params.attribute_u_max as f32;
    let mapped_attribute = params.attribute_f_min
        + normalized_attribute * (params.attribute_f_max - params.attribute_f_min);
    mapped_attribute
}

// These tests will be built and run for the host target only
#[cfg(test)]
mod tests {

    use core::u32;

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
        let params: RasterParameters = RasterParameters::new(8, 32767, 0., 1., 32767, 0, 0);

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert_eq!(result, 0.);
    }

    #[test]
    fn test_shader_edge_interpolator_inside_triangle_one_plane() {
        let p = UVec2::new(1, 1);
        let v = [
            UVec3::new(0, 0, 32767),
            UVec3::new(0, 127, 32767),
            UVec3::new(127, 0, 32767),
        ];
        let params: RasterParameters = RasterParameters::new(8,32767, 0., 1., 32767, 0, 0);

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_shader_edge_interpolator_inside_triangle_different_z() {
        let params = RasterParameters::new(8, 32767, 0., 1., 32767, 0, 0);

        let p = UVec2::new(1, 1);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(0, 3, 32767),
            UVec3::new(3, 0, 16383),
        ];

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_shader_edge_interpolator_on_edge_different_z() {
        let params = RasterParameters::new(8,32767, 0., 1., 32767, 0, 0);

        let p = UVec2::new(1, 1);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(0, 2, 32767),
            UVec3::new(2, 0, 16383),
        ];

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert_abs_diff_eq!(result, 0.75, epsilon = 1e-5);
    }

    #[test]
    fn test_shader_edge_interpolator_outside_triangle() {
        let params = RasterParameters::new(8,32767, 0., 1., 32767, 0, 0);
        // this point is outside the triangle
        // we actually calculate the interpolated value for the plane on
        // which the triangle lies, so the result is still valid
        let p = Vec2::new(3., 3.);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(2, 0, 32767),
            UVec3::new(0, 2, 32767),
        ]
        .map(|v| v.as_vec3());

        let result = interpolate_barycentric(v, p, &params);
        assert!(result == 3.);
    }

    #[test]
    fn test_shader_edge_interpolator_at_vertex_different_z() {
        let params = RasterParameters::new(8,32767, 0., 1., 32767, 0, 0);

        let p = UVec2::new(0, 0);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(0, 2, 32767),
            UVec3::new(2, 0, 16383),
        ];

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert_eq!(result, 0.);
    }

    #[test]
    fn test_shader_edge_interpolator_on_edge() {
        let params = RasterParameters::new(8,32767, 0., 1., 32767, 0, 0);

        let p = UVec2::new(1, 0);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(2, 0, 32767),
            UVec3::new(0, 2, 32767),
        ];

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_shader_edge_interpolator_at_vertex() {
        let params = RasterParameters::new(8,32767, 0., 1., 32767, 0, 0);

        let p = UVec2::new(0, 0);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(2, 0, 32767),
            UVec3::new(0, 2, 32767),
        ];

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert_eq!(result, 0.);
    }

    #[test]
    fn test_shader_edge_interpolator_degenerate_triangle() {
        let params = RasterParameters::new(8,32767, 0., 1., 32767, 0, 0);

        let p = UVec2::new(1, 1);
        let v = [
            UVec3::new(0, 0, 0),
            UVec3::new(0, 0, 32767),
            UVec3::new(0, 0, 32767),
        ];

        let result = interpolate_barycentric(v.map(|v| v.as_vec3()), p.as_vec2(), &params);
        assert!(result.is_nan());
    }

    #[test]
    fn test_rasterise_pixel_inside_triangle() {
        let params = RasterParameters::new(9, 32767, 0., 1000000., 32767, 3, 1);

        let mut cell_data: CellData = [[UVec3::new(0, 0, 0); 3]; MAX_CELL_TRIANGLES];
        cell_data[0] = [
            UVec3::new(0, 0, 0),
            UVec3::new(63, 0, 32767),
            UVec3::new(0, 63, 32767),
        ];

        // Top right corner of the raster, inside the triangle
        let cell = UVec2::new(7, 0);
        let pixel = UVec2::new(5, 1);

        let result = rasterise_cell_pixel(&params, &cell_data, cell, pixel);
        
        assert!(result.is_some());
        assert_abs_diff_eq!(result.unwrap(), 984127.0, epsilon = 1e-5);
    }

    #[test]
    fn test_rasterise_pixel_outside_triangle() {
        let params = RasterParameters::new(9,32767, 0., 1., 32767, 3, 1);

        let mut cell_data: CellData = [[UVec3::new(0, 0, 0); 3]; MAX_CELL_TRIANGLES];
        cell_data[0] = [
            UVec3::new(0, 0, 0),
            UVec3::new(63, 0, 32767),
            UVec3::new(0, 63, 32767),
        ];

        // bottom right corner of the raster, outside the triangle
        let cell = UVec2::new(7, 7);
        let pixel = UVec2::new(7, 7);

        let result = rasterise_cell_pixel(&params, &cell_data, cell, pixel);
        assert!(result.is_none());
    }

    #[test]
    fn test_rasterise_pixel_on_edge() {
        let params = RasterParameters::new(9,32767, 0., 1000000., 32767, 3, 1);

        let mut cell_data: CellData = [[UVec3::new(0, 0, 0); 3]; MAX_CELL_TRIANGLES];
        cell_data[0] = [
            UVec3::new(0, 0, 0),
            UVec3::new(63, 0, 32767),
            UVec3::new(0, 63, 32767),
        ];

        // Top right corner of the raster, corner of triangle
        let cell = UVec2::new(7, 0);
        let pixel = UVec2::new(7, 0);

        let result = rasterise_cell_pixel(&params, &cell_data, cell, pixel);
        assert!(result.is_some());
        let res_val = result.unwrap();
        assert!(res_val.is_finite());
        assert_abs_diff_eq!(res_val, 1000000.0, epsilon = 1e-5);
    }

    #[test]
    fn test_load_cell_triangles_with_triangles() {
        let u_buffer = vec![0, 2, 0];
        let v_buffer = vec![0, 0, 2];
        let h_buffer = vec![0, 32767, 32767];
        let indices = vec![0, 1, 2, 0, 1, 2]; //two copies of same triangle
        let bounding_boxes = vec![UVec4::new(0, 0, 2, 2), UVec4::new(0, 0, 2, 2)];
        let cell = UVec2::new(0, 0);
        let mut cell_data: CellData =
            [[UVec3::new(u32::MAX, u32::MAX, u32::MAX); 3]; MAX_CELL_TRIANGLES];

        assert_eq!(bounding_boxes.len(), 2);

        // two bounding boxes, one for thread one. one for thread two.
        // call twice, and cell_data should be filled with 2x the same triangle

        load_cell_triangles(
            &u_buffer,
            &v_buffer,
            &h_buffer,
            &indices,
            &bounding_boxes,
            cell,
            &mut cell_data,
        );

        let expected_cell_data = [
            UVec3::new(0, 0, 0),
            UVec3::new(2, 0, 32767),
            UVec3::new(0, 2, 32767),
        ];

        assert_eq!(cell_data[0], expected_cell_data);
        assert_eq!(cell_data[1], expected_cell_data);
    }

    #[test]
    fn test_load_cell_triangles_with_triangles_not_covering_cell() {
        let u_buffer = vec![0, 2, 0];
        let v_buffer = vec![0, 0, 2];
        let h_buffer = vec![0, 32767, 32767];
        let indices = vec![0, 1, 2];
        let bounding_boxes = vec![UVec4::new(0, 0, 2, 2)];
        let cell = UVec2::new(7, 7);
        let mut cell_data: CellData =
            [[UVec3::new(u32::MAX, u32::MAX, u32::MAX); 3]; MAX_CELL_TRIANGLES];

        load_cell_triangles(
            &u_buffer,
            &v_buffer,
            &h_buffer,
            &indices,
            &bounding_boxes,
            cell,
            &mut cell_data,
        );

        let expected_cell_data = [[
            UVec3::new(u32::MAX, u32::MAX, u32::MAX),
            UVec3::new(u32::MAX, u32::MAX, u32::MAX),
            UVec3::new(u32::MAX, u32::MAX, u32::MAX),
        ]];

        assert_eq!(cell_data[0], expected_cell_data[0]);
    }

    #[test]
    fn test_intersects_cell() {
        // 8,8 is the top left corner of the cell 1,1
        let aabb = AABB::new_aabb(8, 8, 16, 16);
        assert!(intersects_cell(&aabb, UVec2::new(1, 1)));

        // 15,1 is the bottom right corner of the cell 1,1
        let aabb = AABB::new_aabb(15, 15, 16, 16);
        assert!(intersects_cell(&aabb, UVec2::new(1, 1)));

        //For aabb 7,7:15,15
        // 0,0, 0,1, 1,0, 1,1 all intersect
        // 0,2, 2,0, 2,2 do not intersect
        let aabb = AABB::new_aabb(7, 7, 15, 15);
        assert!(intersects_cell(&aabb, UVec2::new(0, 0)));
        assert!(intersects_cell(&aabb, UVec2::new(0, 1)));
        assert!(intersects_cell(&aabb, UVec2::new(1, 0)));
        assert!(intersects_cell(&aabb, UVec2::new(1, 1)));

        assert!(!intersects_cell(&aabb, UVec2::new(0, 2)));
        assert!(!intersects_cell(&aabb, UVec2::new(2, 0)));
        assert!(!intersects_cell(&aabb, UVec2::new(2, 2)));

        // 1,1 is entirely inside
        let aabb = AABB::new_aabb(0, 0, 31, 31);
        assert!(intersects_cell(&aabb, UVec2::new(1, 1)));
    }

    #[test]
    fn test_cell_pixel_to_raster_pixel() {
        let cell = UVec2::new(2, 3);
        let cell_pixel = UVec2::new(4, 5);
        let expected = UVec2::new(2*GRID_CELL_SIZE_U32 +4, 3*GRID_CELL_SIZE_U32+5);
        let result = cell_pixel_to_raster_pixel(cell_pixel, cell);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cell_to_raster_pixel() {
        let cell = UVec2::new(2, 3);
        let expected = UVec2::new(2*GRID_CELL_SIZE_U32, 3*GRID_CELL_SIZE_U32);
        let result = cell_to_raster_pixel(cell);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_raster_pixel_to_raster_index() {
        let pixel = UVec2::new(1954, 2045);
        let params = RasterParameters::new(3, 32767, 0.0, 100.0, 32767, 0, 0);
        
        //reverse y axis
        let adjusted_y = params.scaled_raster_size() - 1 - pixel.y;

        let expected = (adjusted_y * params.scaled_raster_size() + 1954) as usize;
        let result = raster_pixel_to_raster_index(pixel, &params);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cell_to_raster_index() {
        let cell = UVec2::new(2, 3);
        let params = RasterParameters::new(3, 32767, 0.0, 100.0, 32767, 0, 0);
        let raster_pixel = cell_to_raster_pixel(cell);
        let expected = raster_pixel_to_raster_index(raster_pixel, &params);
        let result = cell_to_raster_index(cell, &params);
        assert_eq!(result, expected);
    }


}
