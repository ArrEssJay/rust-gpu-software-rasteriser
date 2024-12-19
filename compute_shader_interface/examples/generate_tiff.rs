use std::error::Error;
use std::fs::File;

use std::env;
use tiff::encoder::colortype::Gray32Float;
use tiff::encoder::TiffEncoder;
use tiff::encoder::compression::Lzw;
use compute_shader::RasterParameters;
use compute_shader_interface::{rasterise, Rasteriser, VertexBuffers};

fn generate_raster(
    rasteriser: Rasteriser,
    raster_scale_factor: u32,
) -> Vec<f32> {
    // Simple pair of triangles forming a plane
    // Either flat at the max attribute or sloping from 0 along the x-axis
    let max = 32767_u32;
    let indices: Vec<u32> = vec![0, 1, 2, 1, 2, 3];

    let u: Vec<u32> = vec![0, max, 0, max];
    let v: Vec<u32> = vec![0, 0, max, max];

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

    rasterise(vertex_buffers, &params, rasteriser)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command-line arguments for rasteriser type and raster_scale_factor
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <cpu|gpu> <raster_scale_factor> [output_path]", args[0]);
        std::process::exit(1);
    }

    let rasteriser = match args[1].as_str() {
        "cpu" => Rasteriser::CPU,
        "gpu" => Rasteriser::GPU,
        _ => {
            eprintln!("Invalid rasteriser type. Use 'cpu' or 'gpu'.");
            std::process::exit(1);
        }
    };

    let raster_scale_factor: u32 = args[2].parse().expect("Invalid raster_scale_factor value");

    let dim_size = 32768_u32 >> raster_scale_factor;

    let output_path = if args.len() >= 4 {
        args[3].clone()
    } else {
        format!("raster_{}x{}.tiff", dim_size, dim_size)
    };

    println!("Generating raster of size {}x{}", dim_size, dim_size);

    let raster = generate_raster(rasteriser, raster_scale_factor);
    let file = File::create(output_path).expect("Failed to create output file");
    let mut tiff = TiffEncoder::new(file).expect("Failed to create TiffEncoder");
    let image = tiff
        .new_image_with_compression::<Gray32Float, Lzw>(dim_size as u32, dim_size as u32, Lzw)
        .unwrap();
    image.write_data(&raster)?;
    Ok(())
}