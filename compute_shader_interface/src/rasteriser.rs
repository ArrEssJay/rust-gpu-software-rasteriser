mod gpu_runner;

use compute_shader::RasterParameters;
use compute_shader::GRID_CELL_SIZE_U32;
use glam::UVec2;
use glam::UVec3;
use glam::UVec4;
use gpu_runner::run_compute_shader;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VertexArrays<'a> {
    pub u: &'a [u32],
    pub v: &'a [u32],
    pub h: &'a [u32],
    pub i: &'a [u32],
}

#[derive(PartialEq)]
pub enum Rasteriser {
    CPU,
    GPU,
}
pub fn rasterise(
    vertex_arrays: VertexArrays,
    params: &RasterParameters,
    rasteriser: Rasteriser,
) -> Vec<f32> {
    let bounding_boxes: Vec<UVec4> = generate_triangle_bounding_boxes(vertex_arrays, &params);

    let result = if rasteriser == Rasteriser::CPU {
        run_compute_shader_cells_cpu(vertex_arrays, &params, bounding_boxes.as_slice())
    } else {
        async_std::task::block_on(async {
            run_compute_shader(vertex_arrays, &params, &bounding_boxes).await
        })
    };
    result
}

pub fn generate_triangle_bounding_boxes(
    vertex_arrays: VertexArrays,
    params: &RasterParameters,
) -> Vec<UVec4> {
    let mut bounding_boxes: Vec<UVec4> = Vec::new();

    for i in 0..params.triangle_count as usize {
        let v0 = i * 3;
        let vertex_indices: [usize; 3] = [
            vertex_arrays.i[v0] as usize,
            vertex_arrays.i[v0 + 1] as usize,
            vertex_arrays.i[v0 + 2] as usize,
        ];

        let vertices = [
            UVec2::new(
                vertex_arrays.u[vertex_indices[0] as usize],
                vertex_arrays.v[vertex_indices[0] as usize],
            ),
            UVec2::new(
                vertex_arrays.u[vertex_indices[1] as usize],
                vertex_arrays.v[vertex_indices[1] as usize],
            ),
            UVec2::new(
                vertex_arrays.u[vertex_indices[2] as usize],
                vertex_arrays.v[vertex_indices[2] as usize],
            ),
        ];

        let aabb: UVec4 = calculate_triangle_aabb(&vertices);
        bounding_boxes.push(aabb);
    }

    bounding_boxes
}

// takes 3 x,y vertices and returns the axis aligned bounding box
pub fn calculate_triangle_aabb(v: &[UVec2]) -> UVec4 {
    let min = v[0].min(v[1]).min(v[2]);
    let max = v[0].max(v[1]).max(v[2]);

    UVec4::new(min.x, min.y, max.x, max.y)
}

pub fn run_compute_shader_cells_cpu(
    vertex_arrays: VertexArrays,
    params: &RasterParameters,
    bounding_boxes: &[UVec4],
) -> Vec<f32> {
    use compute_shader::rasterise_cell;

    let mut storage: Vec<f32> =
        vec![-1.; (params.raster_dim_size * params.raster_dim_size) as usize];

    let grid_size = params.raster_dim_size / GRID_CELL_SIZE_U32;
    for y in 0..grid_size {
        for x in 0..grid_size {
            rasterise_cell(
                params,
                vertex_arrays.u,
                vertex_arrays.v,
                vertex_arrays.h,
                vertex_arrays.i,
                &bounding_boxes,
                storage.as_mut_slice(),
                UVec3::new(x, y, 0),
            );
        }
    }
    storage
}

#[cfg(test)]

mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use paste::paste;

    macro_rules! print_raster {
        ($vec:expr, $items_per_row:expr, $label:expr) => {
            println!("{}", $label);
            let mut count = 0;
            for val in $vec {
                print!("{:<5} ", val);
                count += 1;
                if count % $items_per_row == 0 {
                    println!();
                }
            }
            if count % $items_per_row != 0 {
                println!();
            }
        };
    }

    // This is intended to be used with the commented out code above that outputs address/vertex data in the raster
    // for validation purposes

    #[test]
    fn test_vertices() {
        let indices: Vec<u32> = (0..3).flat_map(|i| std::iter::repeat(i).take(32)).collect();

        let u: Vec<u32> = (0..32)
            .flat_map(|i| std::iter::repeat(i).take(32))
            .collect();

        let v: Vec<u32> = (0..32)
            .flat_map(|i| std::iter::repeat(i).take(32))
            .collect();
        let h: Vec<u32> = (0..(32 * 32)).collect();

        let vertex_arrays = VertexArrays {
            u: &u,
            v: &v,
            h: &h,
            i: &indices,
        };

        let params = RasterParameters {
            raster_dim_size: 32,
            height_min: 0.0,
            height_max: 100.0,
            vertex_count: u.len() as u32,
            triangle_count: (indices.len() / 3) as u32,
        };

        let result = rasterise(vertex_arrays, &params, Rasteriser::GPU);

        print_raster!(&result, params.raster_dim_size as usize, "output");

        assert_eq!(
            result.len(),
            (params.raster_dim_size * params.raster_dim_size) as usize
        );
    }

    // For the plane tests below this is simply a linear gradient from 0 to 100
    fn generate_expected_output_row(dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|x| 100.0 * (x as f32 / (dim - 1) as f32))
            .collect()
    }

    macro_rules! generate_rasteriser_test {
        ($raster_fn:ident, $fn_prefix:ident, $dim:expr) => {
            paste! {
                #[test]
                fn [<$fn_prefix _ $dim>]() {


                    let vertices = vec![
                        UVec3::new(0, 0, 0),
                        UVec3::new(0, $dim as u32 - 1, 0),
                        UVec3::new($dim as u32 - 1, $dim as u32 - 1, 32767),
                        UVec3::new($dim as u32 - 1, 0, 32767),
                    ];

                    let u: Vec<u32> = vertices.iter().map(|v| v.x).collect();
                    let v: Vec<u32> = vertices.iter().map(|v| v.y).collect();
                    let height: Vec<u32> = vertices.iter().map(|v| v.z).collect();



                    let indices = vec![0, 1, 2,0, 2, 3];

                    let params = RasterParameters {
                        raster_dim_size: $dim,
                        height_min: 0.,
                        height_max: 100.,
                        vertex_count: vertices.len() as u32,
                        triangle_count: indices.len() as u32/3,

                    };

                    let vertex_arrays = VertexArrays {
                        u: &u,
                        v: &v,
                        h: &height,
                        i: &indices,
                    };
                    let storage = $raster_fn(vertex_arrays, &params, Rasteriser::GPU);

                    let expected_output_row = generate_expected_output_row($dim);

                    for row in 0..params.raster_dim_size {
                        let start = (row * params.raster_dim_size) as usize;
                        let end = start + params.raster_dim_size as usize;
                        let actual_row = &storage[start..end];
                        print_raster!(actual_row, $dim, "actual");
                        assert_abs_diff_eq!(
                            actual_row,
                            expected_output_row.as_slice(),
                            epsilon = 1e-4
                        );
                    }
                }
            }
        };
    }

    // Generate the test functions for given dimension and each rasteriser
    macro_rules! generate_cell_rasteriser_tests_cpu {
        ($($dim:expr),*) => {
            $(
                generate_rasteriser_test!(rasterise, rasteriser_test, $dim);
            )*
        };
    }

    // generate_plane_triangle_rasteriser_tests!(8, 16, 32, 64, 128, 256, 512, 1024);
    //generate_plane_triangle_rasteriser_tests!(2048, 4096, 8192, 16384, 32768); // large
    generate_cell_rasteriser_tests_cpu!(8, 16, 32, 64, 128, 256, 512, 1024);
    //generate_plane_raster_block_scanline_tests!(2048, 4096, 8192, 16384, 32768); //large

    #[test]
    fn test_generate_expected_output_row() {
        let dim = 64;
        let expected_output = vec![
            0.0, 1.5873017, 3.1746035, 4.7619047, 6.349207, 7.936508, 9.523809, 11.111112,
            12.698414, 14.285715, 15.873016, 17.460318, 19.047619, 20.63492, 22.222223, 23.809525,
            25.396828, 26.984129, 28.57143, 30.158731, 31.746033, 33.333336, 34.920635, 36.50794,
            38.095238, 39.68254, 41.26984, 42.857143, 44.444447, 46.031746, 47.61905, 49.20635,
            50.793655, 52.380955, 53.968258, 55.555557, 57.14286, 58.73016, 60.317463, 61.904762,
            63.492065, 65.07937, 66.66667, 68.25397, 69.84127, 71.42857, 73.01588, 74.60318,
            76.190475, 77.77778, 79.36508, 80.952385, 82.53968, 84.12698, 85.71429, 87.30159,
            88.88889, 90.47619, 92.06349, 93.650795, 95.2381, 96.82539, 98.4127, 100.0,
        ];
        let result = generate_expected_output_row(dim);
        assert_eq!(result, expected_output);
    }

    /// Helper function to flatten triangle indices from Vec<[u32; 3]> to Vec<u32>
    fn flatten_triangle_indices(triangles: &Vec<[u32; 3]>) -> Vec<u32> {
        triangles
            .iter()
            .flat_map(|triangle| triangle.iter())
            .copied()
            .collect()
    }    
}
