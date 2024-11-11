use rayon::prelude::*;
use glam::{UVec2, UVec3};

use compute_shader::{
    cell_pixel_to_raster_index, load_cell_triangles, rasterise_pixel, CellData, RasterParameters, AABB, GRID_CELL_SIZE_U32, MAX_CELL_TRIANGLES
};
use crate::VertexArrays;

/// Executes the compute shader by parallelizing over 8x8 cells.
/// Within each cell, pixels are processed serially.
pub fn execute_compute_shader_host(
    vertex_arrays: VertexArrays,
    params: &RasterParameters,
    bounding_boxes: &[AABB],
) -> Vec<f32> { 
    let raster_size = (params.raster_dim_size * params.raster_dim_size) as usize;
    use std::sync::Mutex;
    let storage = Mutex::new(vec![-1.0; raster_size]);

    let grid_size = params.raster_dim_size / GRID_CELL_SIZE_U32;

    // Parallel iterate over cell rows
    (0..grid_size).into_par_iter().for_each(|cell_y| {
        for cell_x in 0..grid_size {

            let mut cell_data: CellData = [[UVec3::ZERO; 3]; MAX_CELL_TRIANGLES];
            
            let cell = UVec2::new(cell_x, cell_y);
            // Load cell vertices (populate shared_indices)
            load_cell_triangles(
                vertex_arrays.u,
                vertex_arrays.v,
                vertex_arrays.h,
                vertex_arrays.i,
                bounding_boxes,
                cell,
                &mut cell_data,
            );

            // Iterate over pixels within the 8x8 cell
            for yp in 0..GRID_CELL_SIZE_U32 {
                for xp in 0..GRID_CELL_SIZE_U32 {
                                        
                    // Compute local pixel position within the cell
                    let pixel = UVec2::new(xp, yp);

                    let raster_index = cell_pixel_to_raster_index(cell, pixel, params);
                    let mut storage = storage.lock().unwrap();
                    
                    // will return an out of bounds value if pixel is outside cell
                    let val  = rasterise_pixel(params, &cell_data, cell, pixel);
                    if val > params.height_min {
                        storage[raster_index] = val;
                    }
                }
            }
        }
    });

    storage.into_inner().unwrap()
}