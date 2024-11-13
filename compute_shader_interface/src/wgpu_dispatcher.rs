use bytemuck::{Pod, Zeroable};
use glam::UVec4;
use std::fs::File;
use std::io::Read;
use wgpu::util::DeviceExt;
use wgpu::{Adapter, Features,Limits};

use crate::VertexArrays;
use compute_shader::RasterParameters;
use compute_shader::GRID_CELL_SIZE_U32;

// Local defition of the glam UVec4 struct
// Glam does not implement the IntoBytes trait
// This negates manual implementation of the
// serialisation logic for the glam UVec4 struct
// This is a bit of a hack. The newtype approach
// etc. would be more idiomatic but did not
// work with the rust-gpu tooling at the time
// of writing
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
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
}

#[derive(Debug)]
pub struct WgpuDispatcher<'a> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    rasterise_compute_pipeline: wgpu::ComputePipeline,
    aabb_compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    storage_buffer: wgpu::Buffer,
    aabb_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    output_raster_size_bytes: u64,
    num_workgroups_x_y: u32,
    raster_parameters: &'a RasterParameters,
}

impl<'a> WgpuDispatcher<'a> {
    pub async fn setup_compute_shader_wgpu(v: VertexArrays<'_>,
        raster_parameters: &'a RasterParameters) -> Self {
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
 
     // Check default limits and increase where needed
     let default_limits = Limits::downlevel_defaults();
     let adapter_limits = adapter.limits();
 
     // Storage buffer determines the maximum size of the output raster
     // If we want to rasterise a large area, we need to either increase
     // this limit or use multiple buffers
     
     let max_storage_buffer_size = adapter_limits.max_storage_buffer_binding_size;
     let max_storage_buffers_per_shader_stage = adapter_limits.max_storage_buffers_per_shader_stage;
     let max_buffer_size = adapter_limits.max_buffer_size;
 
     let custom_limits = Limits {
         max_storage_buffer_binding_size: max_storage_buffer_size,
         max_storage_buffers_per_shader_stage,
         max_buffer_size,
         ..default_limits
     };
 
     println!("\nDefault limits:");
     println!("{:?}", default_limits);
     println!("\nSetting custom limits:");
     println!("max_storage_buffer_binding_size = {}", max_storage_buffer_size);
     println!("max_storage_buffers_per_shader_stage = {}", max_storage_buffers_per_shader_stage);
     println!("max_buffer_size = {}", max_buffer_size);
 
 
 
     let (device, queue) = adapter
         .request_device(
             &wgpu::DeviceDescriptor {
                 label: None,
                 required_features: Features::empty(),
                 required_limits: custom_limits.clone(),
                 memory_hints: wgpu::MemoryHints::Performance,
             },
             None,
         )
         .await
         .expect("Failed to create device");
     drop(instance);
     drop(adapter);
 
     // Raster buffer
     let output_raster_size_bytes = (raster_parameters.raster_dim_size as u64 * raster_parameters.raster_dim_size as u64)
         * std::mem::size_of::<f32>() as u64;
     
     println!("Required storage buffer size: {} bytes", output_raster_size_bytes);
     
     // Check that this doesn't fall outside buffer size limit
     
     if output_raster_size_bytes > max_storage_buffer_size as u64 {
         panic!("Output raster size exceeds device limits: {} > {}", output_raster_size_bytes, max_storage_buffer_size);
     }

      // Input Data Buffers
    let params_bytes =  bytemuck::bytes_of(raster_parameters);
    let u_bytes = bytemuck::cast_slice(v.u);
    let v_bytes = bytemuck::cast_slice(v.v);
    let h_bytes = bytemuck::cast_slice(v.h);
    let indices_bytes =bytemuck::cast_slice(v.i);

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

    let aabb_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("AABB Buffer"),
        size:  16u64 * raster_parameters.triangle_count as u64, // AABB=UVEC4=4xu32=16 bytes
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
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

    // Read the shader binary image
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spirv_target = env!("SPIRV_TARGET");
    let spirv_crate = env!("SPIRV_CRATE");
    let path = format!("{}/../target/spirv-builder/{}/release/deps/{}.spv", manifest_dir, spirv_target, spirv_crate);
    println!("Loading SPIR-V file: {}", path);
    let mut spirv_file = File::open(path).expect("Failed to open SPIR-V file");
    let mut spirv_bytes = Vec::new();
    spirv_file.read_to_end(&mut spirv_bytes)
        .expect("Failed to read SPIR-V file");

    let label = "Shader Module";
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::SpirV( bytemuck::cast_slice(&spirv_bytes).to_vec().into()),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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


    let aabb_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        compilation_options: Default::default(),
        cache: None,
        label: Some("spirv_compute_aabb"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "spirv_compute_aabb",
    });
    
    let rasterise_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        compilation_options: Default::default(),
        cache: None,
        label:  Some("spirv_rasterise"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "spirv_rasterise",
    });


    // Calculate the number of workgroups needed - symmetric about x and y
    let num_workgroups_x_y = raster_parameters.raster_dim_size / GRID_CELL_SIZE_U32;

    // Unlikely to be an issue but check the number of workgroups is within device limits
    if num_workgroups_x_y > custom_limits.max_compute_workgroups_per_dimension {
        panic!("Number of workgroups exceeds device limits: {} > {}", num_workgroups_x_y, custom_limits.max_compute_workgroups_per_dimension);
    }

    WgpuDispatcher {
        device,
        queue,
        rasterise_compute_pipeline,
        aabb_compute_pipeline,
        bind_group,
        storage_buffer,
        aabb_buffer,
        readback_buffer,
        output_raster_size_bytes,
        num_workgroups_x_y,
        raster_parameters
    }
    }

    pub async fn execute_compute_shader_wgpu(&mut self) -> Vec<f32> {
    
        let mut aabb_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("AABB Command Encoder"),
        });
        
        // first pass computes bounding boxes per-triangle
        {
            let mut aabb_pass = aabb_encoder.begin_compute_pass(&Default::default());
            aabb_pass.set_pipeline(&self.aabb_compute_pipeline);
            aabb_pass.set_bind_group(0, &self.bind_group, &[]);
    
            // dispatch 1 workgroup per triangle, 1 thread per triangle
            aabb_pass.dispatch_workgroups(self.raster_parameters.triangle_count,1,1);
        }
         // Copy data from the storage buffer to the readback buffer
         aabb_encoder.copy_buffer_to_buffer(
            &self.aabb_buffer,
            0,
            &self.readback_buffer,
            0,
            16u64 * self.raster_parameters.triangle_count as u64,
        );
        
        self.queue.submit(Some(aabb_encoder.finish()));



        // Second pass loads vertices per-cell and rasterises each pixel
        // Scope to ensure compute pass is dropped before the buffer is mapped
        let mut rasterise_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Rasterise Command Encoder"),
        });
        {
            let mut compute_pass = rasterise_encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&self.rasterise_compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
    
            compute_pass.dispatch_workgroups(self.num_workgroups_x_y, self.num_workgroups_x_y, 1);
        }
        
        // Copy data from the storage buffer to the readback buffer
        rasterise_encoder.copy_buffer_to_buffer(
            &self.storage_buffer,
            0,
            &self.readback_buffer,
            0,
            self.output_raster_size_bytes,
        );
    
        self.queue.submit(Some(rasterise_encoder.finish()));
    
        let buffer_slice = self.readback_buffer.slice(..);
    
        buffer_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    
        // NOTE(eddyb) `poll` should return only after the above callbacks fire
        // (see also https://github.com/gfx-rs/wgpu/pull/2698 for more details).
        self.device.poll(wgpu::Maintain::Wait);
    
        let data = buffer_slice.get_mapped_range();
        let result = data
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        drop(data);
        self.readback_buffer.unmap();

        result
    }
    
}


