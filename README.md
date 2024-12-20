# rust-gpu-software-rasterise

This is an experimental cell-based Compute Shader Rasteriser written in Rust using `rust-gpu` for SPIR-V targetting. Broadly, the implementation follows Juan Piñeda's method \[1\].

It is principally a proof-of-concept and learning exercise. It may not be directly suitable for use in your project. However, as a practical demonstration of the aforementioned, being an out-of-tree, non-trivial example of using `rust-gpu` I hope you find something useful here.

For certain applications, the GPU render pipeline is not appropriate, such as for analytical applications. Display-targeted graphics APIs like OpenGL, Vulkan, Metal, etc., do not guarantee pixel-exact consistency between implementations. By implementing a software rasteriser on the GPU as a compute shader, we can ensure that the output is consistent regardless of the target platform.

Using `rust-gpu`, the shader itself is written in `no-std` Rust, which can be compiled both to SPIR-V bytecode and for a CPU target.

- **`wgpu_dispatcher`** handles loading the SPIR-V bytecode and associated buffer plumbing via `wgpu`.
- **`host_dispatcher`** simulates the parallelism of the GPU environment to execute the shader code on the host, using `rayon` for parallel execution at the cell level. Each pixel is rendered sequentially.

## Usage

```rust
// Two triangles on a sloping plane, 1024x1024 grid

let u: Vec<u32> = vec![0, 1024, 0, 1024];
let v: Vec<u32> = vec![0, 0, 1024, 1024];
let attr: Vec<u32> = vec![0, 32767, 0, 32767];

// Triangles using vertices [0,1,2] and [1,2,3]
let indices: Vec<u32> = vec![0, 1, 2, 1, 2, 3];

let vertex_buffers = VertexBuffers {
    u: &u,
    v: &v,
    attribute: &attr,
    indices: &indices,
};

let params = RasterParameters {
    raster_dim_size: 1024,                      // Raster dimension size (max value of `u` and `v`)
    attribute_f_min: 0.0,                       // Minimum f32 scaling value for attribute
    attribute_f_max: 100.0,                     // Maximum f32 scaling value for attribute
    attribute_u_max: 32767,                     // Maximum u32 scaling value for attribute (min=0)
    vertex_count: u.len() as u32,               // Total number of vertices
    triangle_count: (indices.len() / 3) as u32, // Total number of triangles
};

let raster: Vec<f32> = rasterise(vertex_buffers, &params, Rasteriser::GPU);
```

Where:

- `u`, `v`: `u32` 2D vertex coordinates.
- `attribute`: `u32` attribute, e.g. height, interpolated over the plane defined by the 3 vertices of each triangle. Linearly interpolated `0:attribute_u_max -> attribute_f_min:attribute_f_max`
- `indices`: `u32` triangle vertex indices in `u`, `v`, `attribute` (stride of 3).

The rasteriser is specifically targeted at closed, triangulated meshes. Some optimisation has been performed; however, there is certainly scope for improvement.

## Assumptions and Restrictions

- **Closed Mesh**: The mesh is assumed to be closed and well-formed, without holes, overlapping, or degenerate triangles.
- **Integer Vertex Coordinates**: Vertices use integer `X`/`Y` coordinates. Integer edge inside/outside testing is used to avoid the need for robust floating-point methods in triangle bounds checking.
  - **Buffer Size Limitation**: Buffer size is the primary limitation to the size of the raster that can be generated. Tiled rasterisation and transfer are not yet supported.
- **Attribute Interpolation Precision**: Interpolation of the triangle face attribute (`attribute`) is limited to `f32`. Using `f64` may yield lower floating-point precision errors but has not been tested.
- **Raster Dimensions**: Vertices/raster dimensions must be equal in `x` and `y` and divisible by 8. This is not a hard limitation, but more flexibility could be added.
- **Vertex Attribute Range**: Vertex attributes must be supplied as `u32` in the range 0–32767. This could be made more flexible.
- **Face Value Function**: The function for calculating the triangle face value is currently hard-coded to perform linear interpolation over the plane formed by the three vertices. Enhancements could make this more versatile.

## Build
- Checkout `rust-gpu` in `$THIS_REPO/..`
- Ensure you have the nightly rust toolchain specified in `rust-toolchain.toml`
- `cargo build`

## Compute Pipeline Stages

- **Stage 1**:
  - Calculate axis-aligned bounding boxes (AABB) for each triangle.
  - Executed across multiple workgroups (one thread per workgroup per triangle).
  - Store the AABBs in global memory.

- **Stage 2**:
  - Subdivide the X×Y raster grid into 8×8 cells (workgroup invocations).
  - For each cell:
    - Parse bounding boxes and load vertices for triangles intersecting the cell from global into shared workgroup memory.
    - Dispatch 8×8 threads per workgroup, one per pixel:
      - Test each triangle sequentially for coverage of the pixel.
      - Use the edge orientation method to determine if the point is inside the triangle or on an edge (exactly on an edge is considered inside).
      - If the pixel is inside, calculate barycentric coordinates and interpolate the triangle's `attribute` value at the given point and linearly interpolate.

`wgpu` is used to provide an operating system and graphics API abstraction layer when executing on the GPU.

## Testing

Run `cargo bench` to obtain benchmark numbers. A 52x speedup (115mS vs 6S) compared to running multithreaded on the CPU for a 4096x4096 raster was achieved on a 2019 Macbook Pro (AMD Radeon Pro 560X).
![Benchmark](assets/bench_comparison.svg)

## References

\[1\] Juan Piñeda, "A Parallel Algorithm for Polygon Rasterization," ACM SIGGRAPH Computer Graphics, Volume 22, Number 4, August 1988.
