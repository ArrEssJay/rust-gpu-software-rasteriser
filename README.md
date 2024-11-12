# rust-gpu-software-rasterise

This is an experimental cell-based Compute Shader Rasteriser written in Rust using rust-gpu. Broadly, the implementation follows Juan Piñeda's method [1].

It is principally a Proof-Of-Concept/learning exercise. It is likely not *directly* suitable for use in your project. However, as a practical demonstration of a non-trivial GPU compute shader, it may be useful for reference.

For certain applications, the GPU render pipeline is not appropriate, such as for analytical applications. Display-targeted graphics APIs such as OpenGL, Vulkan, Metal, etc. offer no guarantee of pixel-exact consistency between implementations. By implementing a software rasteriser on the GPU as a compute shader, we can guarantee that the output will be the same regardless of the target on which the data is rendered.

Using rust-gpu, the shader itself is written in `no-std` Rust, which can be compiled both to SPIR-V bytecode and for a CPU target.

- `wgpu_dispatcher` handles loading the SPIR-V bytecode and associated buffer plumbing via `wgpu`.
- `host_dispatcher` mocks the parallelism of the GPU environment sufficiently to execute the shader code on the host, using `rayon` for parallel execution at the cell level. Each pixel is rendered sequentially.

The rasteriser itself is specifically targeted at closed, triangulated meshes. At present, it has been tested with quantized-mesh terrain data

Correspondingly, there are a number of assumptions and restrictions:
- Closed cell mesh: It is assumed that the mesh is closed and well-formed, absent of holes, overlapping, or degenerate triangles.
- Vertices with integer X/Y coordinates: Integer edge inside/outside testing is used to avoid the need for robust floating point methods in triangle bounds checking.
  - Buffer size is the primary limitation to the size of the raster that can be generated. Tiled rasterisation and transfer is not (yet) supported
- Interpolation of the triangle face attribute (height) is limited to `f32` as I do not have an environment supporting `f64`. It *may* work, yielding lower float precision error.

## Compute Pipeline Stages

- Stage 1:
  - Calculate axis-aligned bounding boxes for each triangle - stored in global memory - executed across no. triangles * workgroups @ 1 thread per workgroup.
  - Store in global memory.

- Stage 2:
  - X*Y Raster grid subdivided into 8x8 cells (workgroup invocations).
  - Parse bounding boxes and load vertices for triangles intersecting this cell from global to shared workgroup memory.
  - Dispatch 8x8 threads per workgroup, 1 per pixel:
    - Test triangles sequentially for coverage of this pixel.
    - Use edge orientation method to determine whether the point is in the triangle/on a line. Exactly on an edge is considered inside.
    - If the pixel is inside, calculate barycentric coordinates of the point and interpolate the triangle Z (height) value.


## References
[1] Juan Piñeda, "A Parallel Algorithm for Polygon Rasterization", ACM SIGGRAPH Computer Graphics, Volume 22, Number 4, August 1988.