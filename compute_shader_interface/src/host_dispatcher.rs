// This module is responsible for dispatching the compute shader on a CPU host

use rayon::prelude::*;
use glam::{UVec2, UVec3};

use compute_shader::{
    cell_pixel_to_raster_index, compute_aabb, load_cell_triangles, rasterise_pixel, CellData, RasterParameters, AABB, GRID_CELL_SIZE_U32, MAX_CELL_TRIANGLES
};
use crate::VertexBuffers;
use std::sync::Mutex;

/// Executes the compute shader by parallelizing over 8x8 cells.
/// Within each cell, pixels are processed serially.
pub fn execute_compute_shader_host(
    vertex_arrays: VertexBuffers,
    params: &RasterParameters,
) -> Vec<f32> {

    let raster_size = (params.raster_dim_size * params.raster_dim_size) as usize;
    let raster = Mutex::new(vec![-1.0; raster_size]);
    let cell_count = params.raster_dim_size / GRID_CELL_SIZE_U32;

    // Calculate bounding-boxes for each triangle
    let aabb: Vec<AABB> = (0..params.triangle_count as usize)
    .map(|i| compute_aabb(i, vertex_arrays.u, vertex_arrays.v, vertex_arrays.indices))
    .collect();

    // Parallel iterate over cell rows
    (0..cell_count).into_par_iter().for_each(|cell_y| {
        for cell_x in 0..cell_count {

            let mut cell_data: CellData = [[UVec3::ZERO; 3]; MAX_CELL_TRIANGLES];
            
            let cell = UVec2::new(cell_x, cell_y);
            // Load cell vertices (populate shared_indices)
            load_cell_triangles(
                vertex_arrays.u,
                vertex_arrays.v,
                vertex_arrays.attribute,
                vertex_arrays.indices,
                aabb.as_slice(),
                cell,
                &mut cell_data,
            );
            

            // Iterate over each pixel within the 8x8 cell
            for yp in 0..GRID_CELL_SIZE_U32 {
                for xp in 0..GRID_CELL_SIZE_U32 {
                                        
                    // Compute local pixel position within the cell
                    let pixel = UVec2::new(xp, yp);

                    // Compute pixel value and store in raster
                    // The mutex is not ideal, but it is a simple way to share the raster
                    // between threads in this case, despite each thread only writing to a
                    // unique region of the raster.
                    let raster_index = cell_pixel_to_raster_index(cell, pixel, params);
                    let mut storage = raster.lock().unwrap();
                    
                    // will return an out of bounds value if pixel is outside cell
                    let val  = rasterise_pixel(params, &cell_data, cell, pixel);
                    if val > params.height_min {
                        storage[raster_index] = val;
                    }
                }
            }
        }
    });

    raster.into_inner().unwrap()
}