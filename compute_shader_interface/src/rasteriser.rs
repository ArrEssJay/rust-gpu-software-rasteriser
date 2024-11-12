pub mod host_dispatcher;
pub mod wgpu_dispatcher;

use compute_shader::RasterParameters;
use glam::UVec2;
use glam::UVec4;
use host_dispatcher::execute_compute_shader_host;
use wgpu_dispatcher::WgpuDispatcher;

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
    if rasteriser == Rasteriser::CPU {
        execute_compute_shader_host(vertex_arrays, params)
    } else {
        async_std::task::block_on(async {
            let mut dispatcher =
                WgpuDispatcher::setup_compute_shader_wgpu(vertex_arrays, params).await;
            dispatcher.execute_compute_shader_wgpu().await
        })
    }
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
                vertex_arrays.u[vertex_indices[0]],
                vertex_arrays.v[vertex_indices[0]],
            ),
            UVec2::new(
                vertex_arrays.u[vertex_indices[1]],
                vertex_arrays.v[vertex_indices[1]],
            ),
            UVec2::new(
                vertex_arrays.u[vertex_indices[2]],
                vertex_arrays.v[vertex_indices[2]],
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const F32_EPSILON: f32 = 1e-4;

    #[allow(unused_macros)]
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

    // For the plane tests below this is simply a linear gradient
    fn generate_expected_gradient(dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|x| 100.0 * (x as f32 / (dim - 1) as f32))
            .collect()
    }

    // pre-computed gradient for a 64x64 plane
    #[test]
    fn test_generate_expected_gradient() {
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
        let result = generate_expected_gradient(dim);
        assert_eq!(result, expected_output);
    }

    fn test_plane(
        dim_size: u32,
        height: f32,
        rasteriser: Rasteriser,
        epsilon: f32,
        gradient: bool,
    ) {
        // Simple pair of triangles forming a plane
        // Either flat at the max height or sloping from 0 along the x-axis
        let max = dim_size - 1;
        let indices: Vec<u32> = vec![0, 1, 2, 1, 2, 3];

        let u: Vec<u32> = vec![0, max, 0, max];
        let v: Vec<u32> = vec![0, 0, max, max];

        let h: Vec<u32> = if gradient {
            vec![0, 32767, 0, 32767]
        } else {
            vec![32767, 32767, 32767, 32767]
        };

        let vertex_arrays = VertexArrays {
            u: &u,
            v: &v,
            h: &h,
            i: &indices,
        };

        let params = RasterParameters {
            raster_dim_size: dim_size,
            height_min: 0.0,
            height_max: height,
            vertex_count: u.len() as u32,
            triangle_count: (indices.len() / 3) as u32,
        };

        let result = rasterise(vertex_arrays, &params, rasteriser);

        // Compare to the reference value(s)
        if gradient {
            let expected_output_row = generate_expected_gradient(dim_size as usize);
            for row in 0..params.raster_dim_size {
                let start = (row * params.raster_dim_size) as usize;
                let end = start + params.raster_dim_size as usize;
                let actual_row = &result[start..end];
                assert_abs_diff_eq!(
                    actual_row,
                    expected_output_row.as_slice(),
                    epsilon = epsilon
                );
            }
        } else {
            for &pixel in &result {
                assert_abs_diff_eq!(pixel, height, epsilon = epsilon);
            }
        }
    }

    #[test]
    fn test_plane_flat_64_cpu() {
        test_plane(64, 100.0, Rasteriser::CPU, F32_EPSILON, false);
    }

    #[test]
    fn test_plane_flat_64_gpu() {
        test_plane(64, 100.0, Rasteriser::GPU, F32_EPSILON, false);
    }

    #[test]
    fn test_plane_gradient_64_cpu() {
        test_plane(64, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }
    #[test]
    fn test_plane_gradient_64_gpu() {
        test_plane(64, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_1024_cpu() {
        test_plane(1024, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_1024_gpu() {
        test_plane(1024, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_4096_cpu() {
        test_plane(4096, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_4096_gpu() {
        test_plane(4096, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_8192_cpu() {
        test_plane(8192, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_8192_gpu() {
        test_plane(8192, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_16384_cpu() {
        test_plane(16384, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_16384_gpu() {
        test_plane(16384, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_32768_cpu() {
        test_plane(32768, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_32768_gpu() {
        test_plane(32768, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }
}
