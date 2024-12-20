#![feature(sync_unsafe_cell)]


pub mod host_dispatcher;
pub mod wgpu_dispatcher;

use compute_shader::RasterParameters;
use glam::UVec2;
use glam::UVec4;
use host_dispatcher::execute_compute_shader_host;
use wgpu_dispatcher::WgpuDispatcher;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VertexBuffers<'a> {
    pub u: &'a [u32],
    pub v: &'a [u32],
    pub attribute: &'a [u32],
    pub indices: &'a [u32],
}

#[derive(PartialEq)]
pub enum Rasteriser {
    CPU,
    GPU,
}
pub fn rasterise(
    vertex_buffers: VertexBuffers,
    params: &RasterParameters,
    rasteriser: Rasteriser,
) -> Vec<f32> {

    {
        let scaled_u_vec: Vec<u32> = vertex_buffers
            .u
            .iter()
            .map(|&value| value >> params.raster_scale_factor)
            .collect();
        let scaled_v_vec: Vec<u32> = vertex_buffers
            .v
            .iter()
            .map(|&value| value >> params.raster_scale_factor)
            .collect();
        let scaled_vertex_buffers = VertexBuffers {
            u: &scaled_u_vec,
            v: &scaled_v_vec,
            attribute: vertex_buffers.attribute,
            indices: vertex_buffers.indices,
        };
    
        for i in 0..params.triangle_count as usize{
            println!("====={:?} i: {:?} {:?} {:?}", i, vertex_buffers.indices[i*3], vertex_buffers.indices[i*3+1], vertex_buffers.indices[i*3+2]);
            
            
            println!("{:?} u: {:?} v: {:?} h: {:?}", vertex_buffers.indices[i*3], scaled_u_vec[vertex_buffers.indices[i*3] as usize], scaled_v_vec[vertex_buffers.indices[i*3] as usize], vertex_buffers.attribute[vertex_buffers.indices[i*3] as usize]);
            println!("{:?} u: {:?} v: {:?} h: {:?}", vertex_buffers.indices[i*3+1], scaled_u_vec[vertex_buffers.indices[i*3+1] as usize], scaled_v_vec[vertex_buffers.indices[i*3+1] as usize], vertex_buffers.attribute[vertex_buffers.indices[i*3+1] as usize]);
            println!("{:?} u: {:?} v: {:?} h: {:?}", vertex_buffers.indices[i*3+2], scaled_u_vec[vertex_buffers.indices[i*3+2] as usize], scaled_v_vec[vertex_buffers.indices[i*3+2] as usize], vertex_buffers.attribute[vertex_buffers.indices[i*3+2] as usize]);

        
        }
        if rasteriser == Rasteriser::CPU {
            execute_compute_shader_host(scaled_vertex_buffers, params)
        } else {
            async_std::task::block_on(async {
                let mut dispatcher =
                    WgpuDispatcher::setup_compute_shader_wgpu(scaled_vertex_buffers, params).await;
                dispatcher.execute_compute_shader_wgpu().await
            })
        }
    }
    
   
}

// pub fn generate_triangle_bounding_boxes(
//     vertex_buffers: VertexBuffers,
//     params: &RasterParameters,
// ) -> Vec<UVec4> {
//     let mut bounding_boxes: Vec<UVec4> = Vec::new();

//     for i in 0..params.triangle_count as usize {
//         let v0 = i * 3;
//         let vertex_indices: [usize; 3] = [
//             vertex_buffers.indices[v0] as usize,
//             vertex_buffers.indices[v0 + 1] as usize,
//             vertex_buffers.indices[v0 + 2] as usize,
//         ];

//         let vertices = [
//             UVec2::new(
//                 vertex_buffers.u[vertex_indices[0]],
//                 vertex_buffers.v[vertex_indices[0]],
//             ),
//             UVec2::new(
//                 vertex_buffers.u[vertex_indices[1]],
//                 vertex_buffers.v[vertex_indices[1]],
//             ),
//             UVec2::new(
//                 vertex_buffers.u[vertex_indices[2]],
//                 vertex_buffers.v[vertex_indices[2]],
//             ),
//         ];

//         let aabb: UVec4 = calculate_triangle_aabb(&vertices);
//         bounding_boxes.push(aabb);
//     }

//     bounding_boxes
// }

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
        raster_scale_factor: u32,
        height: f32,
        rasteriser: Rasteriser,
        epsilon: f32,
        gradient: bool,
    ) {
        // Simple pair of triangles forming a plane
        // Either flat at the max height or sloping from 0 along the x-axis
        let raster_dim_max = 32767;
        let attr_max = 32767;

        

        let indices: Vec<u32> = vec![0, 1, 2, 1, 2, 3];

        let u: Vec<u32> = vec![0, raster_dim_max, 0, raster_dim_max];
        let v: Vec<u32> = vec![0, 0, raster_dim_max, raster_dim_max];

        let attribute: Vec<u32> = if gradient {
            vec![0, attr_max, 0, attr_max]
        } else {
            vec![attr_max, attr_max, attr_max, attr_max]
        };

        let vertex_buffers = VertexBuffers {
            u: &u,
            v: &v,
            attribute: &attribute,
            indices: &indices,
        };
        
        let params = RasterParameters::new(
            raster_scale_factor,
            raster_dim_max,
            0.0,
            height,
            attr_max,
            u.len() as u32,
            (indices.len() / 3) as u32,
        );

        

        let result = rasterise(vertex_buffers, &params, rasteriser);
        if gradient {
            let expected_output_row = generate_expected_gradient(params.scaled_raster_size() as usize);
            for row in 0..params.scaled_raster_size() {
                let start = (row * params.scaled_raster_size()) as usize;
                let end = start + params.scaled_raster_size() as usize;
                let actual_row = &result[start..end];
                println!("== Row: {} Start: {} End: {}", row, start, end);
                println!("actual_row: {:?}", actual_row);

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
        test_plane(9, 100.0, Rasteriser::CPU, F32_EPSILON, false);
    }

    #[test]
    fn test_plane_flat_64_gpu() {
        test_plane(9, 100.0, Rasteriser::GPU, F32_EPSILON, false);
    }

    #[test]
    fn test_plane_gradient_64_cpu() {
        test_plane(9, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }
    #[test]
    fn test_plane_gradient_64_gpu() {
        test_plane(9, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }
    #[test]
    fn test_plane_gradient_128_cpu() {
        test_plane(8, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_256_cpu() {
        test_plane(7, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_512_cpu() {
        test_plane(6, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_1024_cpu() {
        test_plane(5, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_1024_gpu() {
        test_plane(5, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_4096_cpu() {
        test_plane(3, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_4096_gpu() {
        test_plane(3, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_8192_cpu() {
        test_plane(81292, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_8192_gpu() {
        test_plane(2, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_16384_cpu() {
        test_plane(1, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_16384_gpu() {
        test_plane(1, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_32768_cpu() {
        test_plane(0, 100.0, Rasteriser::CPU, F32_EPSILON, true);
    }

    #[test]
    fn test_plane_gradient_32768_gpu() {
        test_plane(0, 100.0, Rasteriser::GPU, F32_EPSILON, true);
    }
}
