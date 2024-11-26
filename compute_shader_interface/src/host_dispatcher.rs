// This module is responsible for dispatching the compute shader on a CPU host
// Note the use of sync-unsafe cells for concurrent mutable access to the raster
// This safety is dependent on the fact that each thread writes to unique ranges within the raster

use glam::{UVec2, UVec3};
use rayon::prelude::*;

use crate::VertexBuffers;
use compute_shader::{
    cell_pixel_to_raster_index, compute_aabb, load_cell_triangles, rasterise_pixel, CellData,
    RasterParameters, AABB, GRID_CELL_SIZE_U32, MAX_CELL_TRIANGLES,
};
use std::cell::SyncUnsafeCell;

/// Executes the compute shader by parallelizing over 8x8 cells.
/// Within each cell, pixels are processed serially.
pub fn execute_compute_shader_host(
    vertex_buffers: VertexBuffers,
    params: &RasterParameters,
) -> Vec<f32> {
    let raster_size = (params.raster_dim_size * params.raster_dim_size) as usize;
    let unsafe_raster: SyncUnsafeCell<Vec<f32>> = vec![-1.0f32; raster_size].into();

    let cell_count = params.raster_dim_size / GRID_CELL_SIZE_U32;

    // Calculate bounding-boxes for each triangle
    let aabb: Vec<AABB> = (0..params.triangle_count as usize)
        .map(|i| {
            compute_aabb(
                i,
                vertex_buffers.u,
                vertex_buffers.v,
                vertex_buffers.indices,
            )
        })
        .collect();

    // Parallel iterate over cell rows
    (0..cell_count).into_par_iter().for_each(|cell_y| {
        for cell_x in 0..cell_count {
            let mut cell_data: CellData = [[UVec3::ZERO; 3]; MAX_CELL_TRIANGLES];

            let cell = UVec2::new(cell_x, cell_y);
            // Load cell vertices (populate shared_indices)
            load_cell_triangles(
                vertex_buffers.u,
                vertex_buffers.v,
                vertex_buffers.attribute,
                vertex_buffers.indices,
                aabb.as_slice(),
                cell,
                &mut cell_data,
                cell_y * GRID_CELL_SIZE_U32 + cell_x,
            );

            // Iterate over each pixel within the 8x8 cell
            for yp in 0..GRID_CELL_SIZE_U32 {
                for xp in 0..GRID_CELL_SIZE_U32 {
                    // Compute local pixel position within the cell
                    let pixel = UVec2::new(xp, yp);

                    // Compute pixel value and store in raster
                    let raster_index = cell_pixel_to_raster_index(cell, pixel, params);

                    // will return an out of bounds value if pixel is outside cell
                    let val = rasterise_pixel(params, &cell_data, cell, pixel);
                    if val > params.attribute_f_min {
                        // SAFETY: multiple concurrent references to the raster are safe
                        // only if no writes to the same address are made concurrently.
                        // The code as-written assigns unique albeit non-contiguous ranges 
                        // of the raster to each thread, guaranteeing safety
                        // 
                        // If this behaviour is changed, the safety of this code MUST be re-evaluated
                        unsafe {
                            let raster_ref = &mut *unsafe_raster.get();
                            raster_ref[raster_index] = val;
                        }
                    }
                }
            }
        }
    });

    unsafe_raster.into_inner()
}
