use glam::UVec4;
use std::fs::File;
use std::io::Read;
use wgpu::util::DeviceExt;
use wgpu::{Adapter, Features};
use zerocopy::{Immutable, IntoBytes};

use crate::VertexArrays;
use compute_shader::RasterParameters;
pub const GRID_CELL_SIZE_U32: u32 = 8;

// This replicates the glam UVec4 type
// This is simply for simplicty and clarity
// of serializing/deserializing
#[repr(C)]
#[derive(IntoBytes, Immutable)]
pub struct BufferUVec4 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
}

impl BufferUVec4 {
    // Custom method to convert from glam::UVec4
    pub fn from_uvec4(vec: UVec4) -> Self {
        BufferUVec4 {
            x: vec.x,
            y: vec.y,
            z: vec.z,
            w: vec.w,
        }
    }
}

fn print_gpu_capabilities(adapter: &Adapter) {
    // Print adapter properties
    let adapter_info = adapter.get_info();
    println!("Adapter Info:");
    println!("  Name: {}", adapter_info.name);
    println!("  Vendor: {}", adapter_info.vendor);
    println!("  Device: {}", adapter_info.device);
    println!("  Type: {:?}", adapter_info.device_type);
    println!("  Backend: {:?}", adapter_info.backend);

    // Print supported features
    let features = adapter.features();
    println!("Supported Features:");
    for feature in wgpu::Features::all().iter() {
        if features.contains(feature) {
            println!("  {:?}", feature);
        }
    }

    // Print supported limits
    let limits = adapter.limits();
    println!("Supported Limits:");
    println!(
        "  Max Texture Dimension 1D: {}",
        limits.max_texture_dimension_1d
    );
    println!(
        "  Max Texture Dimension 2D: {}",
        limits.max_texture_dimension_2d
    );
    println!(
        "  Max Texture Dimension 3D: {}",
        limits.max_texture_dimension_3d
    );
    println!(
        "  Max Texture Array Layers: {}",
        limits.max_texture_array_layers
    );
    println!("  Max Bind Groups: {}", limits.max_bind_groups);
    println!(
        "  Max Dynamic Uniform Buffers Per Pipeline Layout: {}",
        limits.max_dynamic_uniform_buffers_per_pipeline_layout
    );
    println!(
        "  Max Dynamic Storage Buffers Per Pipeline Layout: {}",
        limits.max_dynamic_storage_buffers_per_pipeline_layout
    );
    println!(
        "  Max Sampled Textures Per Shader Stage: {}",
        limits.max_sampled_textures_per_shader_stage
    );
    println!(
        "  Max Samplers Per Shader Stage: {}",
        limits.max_samplers_per_shader_stage
    );
    println!(
        "  Max Storage Buffers Per Shader Stage: {}",
        limits.max_storage_buffers_per_shader_stage
    );
    println!(
        "  Max Storage Textures Per Shader Stage: {}",
        limits.max_storage_textures_per_shader_stage
    );
    println!(
        "  Max Uniform Buffers Per Shader Stage: {}",
        limits.max_uniform_buffers_per_shader_stage
    );
    println!(
        "  Max Uniform Buffer Binding Size: {}",
        limits.max_uniform_buffer_binding_size
    );
    println!(
        "  Max Storage Buffer Binding Size: {}",
        limits.max_storage_buffer_binding_size
    );
    println!("  Max Vertex Buffers: {}", limits.max_vertex_buffers);
    println!("  Max Vertex Attributes: {}", limits.max_vertex_attributes);
    println!(
        "  Max Vertex Buffer Array Stride: {}",
        limits.max_vertex_buffer_array_stride
    );
    println!(
        "  Max Inter-Stage Shader Components: {}",
        limits.max_inter_stage_shader_components
    );
    println!(
        "  Max Compute Workgroup Storage Size: {}",
        limits.max_compute_workgroup_storage_size
    );
    println!(
        "  Max Compute Invocations Per Workgroup: {}",
        limits.max_compute_invocations_per_workgroup
    );
    println!(
        "  Max Compute Workgroup Size X: {}",
        limits.max_compute_workgroup_size_x
    );
    println!(
        "  Max Compute Workgroup Size Y: {}",
        limits.max_compute_workgroup_size_y
    );
    println!(
        "  Max Compute Workgroup Size Z: {}",
        limits.max_compute_workgroup_size_z
    );
    println!(
        "  Max Compute Workgroups Per Dimension: {}",
        limits.max_compute_workgroups_per_dimension
    );
}

pub async fn run_compute_shader(
    v: VertexArrays<'_>,
    params: &RasterParameters,
    bounding_boxes: &Vec<UVec4>,
) -> Vec<f32> {
    // device
    let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
        ..Default::default()
    });
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
        .await
        .expect("Failed to find an appropriate adapter");

    print_gpu_capabilities(&adapter);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("Failed to create device");
    drop(instance);
    drop(adapter);

    // Raster buffer
    let output_raster_size_bytes = (params.raster_dim_size as u64 * params.raster_dim_size as u64)
        * std::mem::size_of::<f32>() as u64;

    // bounding boxes for triangles
    // cast to an identical struct that implements IntoBytes
    let aabb_serialisable = bounding_boxes
        .iter()
        .map(|&a| BufferUVec4::from_uvec4(a))
        .collect::<Vec<_>>();

    // Input Data Buffers
    let params_bytes = params.as_bytes();
    let u_bytes = v.u.as_bytes();
    let v_bytes = v.v.as_bytes();
    let h_bytes = v.h.as_bytes();
    let indices_bytes = v.i.as_bytes();
    let aabb_bytes = aabb_serialisable.as_bytes();

    // Create buffers
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: params_bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let u_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("U Buffer"),
        contents: u_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let v_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("V Buffer"),
        contents: v_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let h_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("H Buffer"),
        contents: h_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Indices Buffer"),
        contents: indices_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let aabb_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("AABB Buffer"),
        contents: aabb_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Storage Buffer"),
        size: output_raster_size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // not bound to any bind group
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: output_raster_size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Read the shader file at runtime
    let entry_point = "main_cs";
    let path = "/Users/rowan/Projects-Code/qmlib/target/spirv-builder/spirv-unknown-spv1.3/release/deps/qmlib_compute_shader.spv";

    let mut file = File::open(path).expect("Failed to open SPIR-V file");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .expect("Failed to read SPIR-V file");

    // Convert bytes to Vec<u32>
    let spirv: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();

    let label = "Shader Module";
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::SpirV(spirv.into()),
    });

    // bind group
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: h_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: aabb_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: storage_buffer.as_entire_binding(),
            },
        ],
        label: Some("Compute Bind Group"),
    });

    // pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        compilation_options: Default::default(),
        cache: None,
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: &entry_point,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    // Calculate the number of workgroups needed - symmetric about x and y
    let num_workgroups_x_y = params.raster_dim_size / GRID_CELL_SIZE_U32;

    println!("num_workgroups_x_y: {}", num_workgroups_x_y,);

    // Set up the compute pass
    // Scope to ensure compute pass is dropped before the buffer is mapped
    {
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(num_workgroups_x_y, num_workgroups_x_y, 1);
    }
    // Copy data from the storage buffer to the readback buffer
    encoder.copy_buffer_to_buffer(
        &storage_buffer,
        0,
        &readback_buffer,
        0,
        output_raster_size_bytes,
    );

    queue.submit(Some(encoder.finish()));

    // init data vec here
    let buffer_slice = readback_buffer.slice(..);

    buffer_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());

    //change this to add code to the
    //buffer_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    // NOTE(eddyb) `poll` should return only after the above callbacks fire
    // (see also https://github.com/gfx-rs/wgpu/pull/2698 for more details).
    device.poll(wgpu::Maintain::Wait);

    let data = buffer_slice.get_mapped_range();
    let result = data
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    drop(data);
    readback_buffer.unmap();

    result
}
