# Chapter 5 — Memory Architecture and Data Locality

This chapter focuses on the CUDA memory hierarchy, data locality, shared-memory tiling, synchronization, boundary handling, and the relationship between memory traffic and kernel performance.

## Key Topics

- CUDA memory types and their scope
- global memory, registers, and shared memory
- data reuse and locality
- tiled matrix multiplication
- cooperative loading
- `__syncthreads()` and inter-thread dependencies
- boundary checks for non-divisible matrix sizes
- arithmetic intensity
- memory-bound vs compute-bound kernels
- shared-memory usage and occupancy

## Notes

- [`notes/chapter05_notes.tex`](notes/chapter05_notes.tex) — LaTeX source
- [`notes/chapter05_notes.pdf`](notes/chapter05_notes.pdf) — compiled notes
- [`notes/fig1.png`](notes/fig1.png) — figure used by the LaTeX notes

## Code

### Tiled Matrix Multiplication

- [`code/tiled_matrix_multiplication.cu`](code/tiled_matrix_multiplication.cu)
- [`code/experiment_summary.md`](code/experiment_summary.md)

The implementation compares a naive matrix-multiplication kernel with a shared-memory tiled version.

The tiled kernel includes:

- cooperative loading of input tiles;
- shared-memory reuse;
- synchronization between loading and computation phases;
- boundary checks for matrix dimensions that are not multiples of the tile width;
- correctness verification against a CPU implementation;
- CUDA-event timing and GFLOP/s measurement.

The experiment also compares the naive and tiled kernels for multiple matrix sizes and discusses arithmetic intensity and global-memory traffic.

## Exercises

- [`exercises/chapter05_exercises.md`](exercises/chapter05_exercises.md)

The exercise solutions cover:

- shared-memory reuse;
- synchronization hazards;
- local vs shared variables;
- global-memory load reduction through tiling;
- arithmetic intensity;
- memory-bound vs compute-bound behavior;
- warp-level execution assumptions;
- shared-memory usage;
- occupancy and resource constraints.

## Main Takeaway

The main idea of this chapter is that GPU performance is often limited not only by arithmetic throughput, but by the cost of moving data.

Tiling improves performance by loading data from global memory into shared memory and reusing it across multiple threads:

```text
global memory
      ↓
shared-memory tile
      ↓
multiple reuses
      ↓
higher arithmetic intensity
```

The broader principle is:

> Use limited on-chip resources such as shared memory and registers to increase data reuse and reduce expensive global-memory traffic, while maintaining enough active warps and blocks to achieve good occupancy.