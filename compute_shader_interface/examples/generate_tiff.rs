use std::error::Error;
use std::fs::File;

use std::env;
use tiff::encoder::colortype::Gray32Float;
use tiff::encoder::TiffEncoder;
use tiff::encoder::compression::Lzw;
use compute_shader::RasterParameters;
use compute_shader_interface::{rasterise, Rasteriser, VertexArrays};

fn generate_raster(
    dim_size: u32,
    rasteriser: Rasteriser,
) -> Vec<f32>{
    // Simple pair of triangles forming a plane
    // Either flat at the max height or sloping from 0 along the x-axis
    let max = dim_size - 1;
    let indices: Vec<u32> = vec![0, 1, 2, 1, 2, 3];

    let u: Vec<u32> = vec![0, max, 0, max];
    let v: Vec<u32> = vec![0, 0, max, max];

    let h: Vec<u32> =vec![0, 32767, 0, 32767];
    

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

    rasterise(vertex_arrays, &params, rasteriser)
 
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command-line arguments for raster size and rasteriser type
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <dim_size> <cpu|gpu> [output_path]", args[0]);
        std::process::exit(1);
    }

    let dim_size: u32 = args[1].parse().expect("dim_size must be a positive integer");
    let rasteriser = match args[2].as_str() {
        "cpu" => Rasteriser::CPU,
        "gpu" => Rasteriser::GPU,
        _ => {
            eprintln!("Invalid rasteriser type. Use 'cpu' or 'gpu'.");
            std::process::exit(1);
        }
    };

    let output_path = if args.len() >= 4 {
        args[3].clone()
    } else {
        format!("raster_{}x{}.tiff", dim_size, dim_size)
    };

    println!(
        "Generating raster of size {}x{} u",
        dim_size, dim_size
    );


    let raster = generate_raster(dim_size, rasteriser);
    let file = File::create(output_path).expect("Failed to create output file");
    let mut tiff: TiffEncoder<File> = TiffEncoder::new(file).expect("Failed to create TiffEncoder");
    let image = tiff
        .new_image_with_compression::<Gray32Float,Lzw>(dim_size.into(), dim_size.into(), Lzw)
        .unwrap();
    image.write_data(&raster)?;
    Ok(())


}