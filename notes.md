## Potential optimizations
- The Voronoi calculations form the main performance bottleneck while saving data, but for long equilibrations some of these optimizations may make sense
- [ ] Rearrange data in multiple cache-aligned arrays (one per coordinate)
- [ ] Store (pointers to) points in quadtree(ish) or grid to reduce the number of points that need to be considered for intersections, or
- [ ] Periodic boundary conditions:
  - [x] Do not need to be checked if the particle is not near a boundary
  - [ ] While making non-periodic pass, record which particles are near the opposite boundary, and only check those

## ToDo
- [x] Create new dataset for working in 2D
- [x] Implement a [Weeks-Chandler-Andersen reference system model](http://www.sklogwiki.org/SklogWiki/index.php/Weeks-Chandler-Andersen_reference_system_model)
