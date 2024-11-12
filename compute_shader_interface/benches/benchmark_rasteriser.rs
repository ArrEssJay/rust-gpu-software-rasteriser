use compute_shader::RasterParameters;
use compute_shader_interface::{
    generate_triangle_bounding_boxes, rasterise, wgpu_dispatcher, Rasteriser, VertexArrays,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use glam::UVec4;
use wgpu_dispatcher::WgpuDispatcher;
use async_std::task::block_on;

async fn benchmark_rasteriser(c: &mut Criterion) {
    let dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096];
    let mut group = c.benchmark_group("Shader Rasteriser");

    for &dim_size in &dim_sizes {
        // Pair of triangles forming a linear slope
        let max = dim_size - 1;
        let indices: Vec<u32> = vec![0, 1, 2, 1, 2, 3];
        let u: Vec<u32> = vec![0, max, 0, max];
        let v: Vec<u32> = vec![0, 0, max, max];
        let h: Vec<u32> = vec![0, 32767, 0, 32767];

        let vertex_arrays = VertexArrays {
            u: &u,
            v: &v,
            h: &h,
            i: &indices,
        };

        let params = RasterParameters {
            raster_dim_size: dim_size,
            height_min: 0.0,
            height_max: 100.0,
            vertex_count: u.len() as u32,
            triangle_count: (indices.len() / 3) as u32,
        };

        let data_size = dim_size * dim_size;

        // Benchmark CPU rasterization
        // there is no need to benchmark the setup of the CPU rasteriser
        group.bench_with_input(
            BenchmarkId::new("CPU", data_size),
            &dim_size,
            |b, _dim_size| {
                b.iter(|| {
                    let _result = rasterise(vertex_arrays, &params, Rasteriser::CPU);
                })
            },
        );

        // Setup GPU dispatcher outside of the execution benchmark
        let mut wgpu_dispatcher =
            WgpuDispatcher::setup_compute_shader_wgpu(vertex_arrays, &params)
                .await;

        // Benchmark GPU execution
        group.bench_with_input(
            BenchmarkId::new("GPU", data_size),
            &dim_size,
            |b, _dim_size| {
                //b.to_async(&executor).iter(|| async {
                //    wgpu_dispatcher.execute_compute_shader_wgpu().await;
                //})
                b.iter(|| {
                    let _result =wgpu_dispatcher.execute_compute_shader_wgpu();
                })
            },
        );
    }

    group.finish();
}

fn async_benchmark_rasteriser(c: &mut Criterion) {
    block_on(benchmark_rasteriser(c));
}

criterion_group!(benches, async_benchmark_rasteriser);
criterion_main!(benches);
