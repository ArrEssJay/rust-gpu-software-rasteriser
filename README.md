# rust-gpu-software-rasterise

This is an experimental Compute Shader Rasteriser written in Rust using rust-gpu.
It is principally a Proof-Of-Concept/learning exercise. It is likely not directly suitable for use in your project.

For certain applications, the GPU render pipeline is not appropriate, such as for analytical applications. Whether DX, GL-ES, Vulkan, Metal...  there is no guarantee of pixel-perfect consistency between implementations.

The rasteriser itself is specifically targeted at closed, trianglulated meshes. At present, it has been tested with quantized-mesh terrain data.

Correspondingly there are a number of assumptions and restrictions:
- Vertices with integer X/Y coordinates: Integer edge inside/outside testing is used to avoid the need for robust floating point methods in triangle bounds checking
- Closed cell mesh: It is assumed that the mesh is closed and well-formed, absent of holes, overlapping or degenerate triangles.

Broadly, the implementation follows Juan Piñeda's method [1].

- Raster grid subdivided into 8x8 cells. One cell per workgroup
- Axis-aligned bounding boxes are pre-computed for each triangle at cell resolution
- Each workgroup (cell) processes triangles with a bounding box covering it
- In each cell, for each pixel, test triangles sequentially to determine which this pixel is inside
- Use edge orientation method to determine whether the point is in the triangle/on a line. Exactly on an edge is considered inside
- If the pixel is inside, calculate barycentric coordinates of the point and interpolate the triangle Z value relative to those of the 3 vertices.
- Each 8x8 pixel block may be rendered by 8x8 threads


## References
[1] Juan Piñeda, "A Parallel Algorithm for Polygon Rasterization", ACM SIGGRAPH Computer Graphics, Volume 22, Number 4, August 1988.