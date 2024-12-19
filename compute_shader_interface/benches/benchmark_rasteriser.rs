use std::sync::Arc;

use async_std::task;
use async_std::{sync::Mutex, task::block_on};
use compute_shader::RasterParameters;
use compute_shader_interface::{rasterise, wgpu_dispatcher, Rasteriser, VertexBuffers};
use wgpu_dispatcher::WgpuDispatcher;

use criterion::{
    async_executor::AsyncStdExecutor, criterion_group, criterion_main, BenchmarkId, Criterion,
};

async fn benchmark_rasteriser(c: &mut Criterion) {
    // Define raster_scale_factor values corresponding to raster dimensions
    let raster_scale_factors = [9, 8, 7, 6, 5, 4, 3]; // For raster_dim_size from 64 to 4096
    let mut group = c.benchmark_group("Shader Rasteriser");
    group.sample_size(10); // Set the sample size to 10

    for &raster_scale_factor in &raster_scale_factors {
        let raster_dim_size = 32768_u32 >> raster_scale_factor;
        let raster_dim_size_u32 = raster_dim_size as u32;
        let max_u_v = 32767_u32;

        // Pair of triangles forming a linear slope
        let indices: Vec<u32> = vec![0, 1, 2, 1, 2, 3];
        let u: Vec<u32> = vec![0, max_u_v, 0, max_u_v];
        let v: Vec<u32> = vec![0, 0, max_u_v, max_u_v];
        let attribute: Vec<u32> = vec![0, 32767, 0, 32767];

        let vertex_buffers = VertexBuffers {
            u: &u,
            v: &v,
            attribute: &attribute,
            indices: &indices,
        };

        let params = RasterParameters::new(
            raster_scale_factor,
            32767,
            0.0,
            100.0,
            32767,
            u.len() as u32,
            (indices.len() / 3) as u32,
        );

        let data_size = raster_dim_size * raster_dim_size;

        // Benchmark CPU rasterization
        group.bench_with_input(
            BenchmarkId::new("CPU", data_size),
            &raster_dim_size_u32,
            |b, _dim_size| {
                b.iter(|| {
                    let _result = rasterise(vertex_buffers, &params, Rasteriser::CPU);
                })
            },
        );

        // Benchmark GPU rasterization
        let wgpu_dispatcher = task::block_on(WgpuDispatcher::setup_compute_shader_wgpu(
            vertex_buffers,
            &params,
        ));
        let wgpu_dispatcher = Arc::new(Mutex::new(wgpu_dispatcher));

        let dispatcher_clone = Arc::clone(&wgpu_dispatcher);
        group.bench_with_input(
            BenchmarkId::new("GPU", data_size),
            &raster_dim_size_u32,
            |b, &_dim_size| {
                b.to_async(AsyncStdExecutor).iter(|| {
                    let dispatcher = Arc::clone(&dispatcher_clone);
                    async move {
                        let mut dispatcher = dispatcher.lock().await;
                        dispatcher.execute_compute_shader_wgpu().await;
                    }
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
