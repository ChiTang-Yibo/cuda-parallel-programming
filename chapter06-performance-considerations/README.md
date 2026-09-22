# Chapter 6 — Performance Considerations

This chapter focuses on practical CUDA performance optimization, especially how memory access patterns, latency hiding, and thread coarsening affect kernel efficiency.

The main topics are:

- global-memory coalescing;
- corner turning for column-major data;
- vectorized memory access;
- latency vs. bandwidth;
- occupancy and memory-level parallelism;
- DRAM banks and channels;
- thread coarsening;
- arithmetic intensity;
- shared-memory bank conflicts and padding;
- performance trade-offs and bottleneck analysis.

---

## Contents

```text
chapter06-performance-considerations/
├── README.md
├── code/
│   ├── coarse_matmul_benchmark.cu
│   └── README.md
├── exercises/
│   ├── corner_turning_benchmark.cu
│   └── CUDA_Chapter6_Exercises.md
└── notes/
    ├── chapter06_notes.pdf
    └── chapter06_notes.tex
```

---

## Notes

The chapter notes summarize the main concepts and performance reasoning developed while studying Chapter 6.

- [Chapter 6 Notes — PDF](notes/chapter06_notes.pdf)
- [Chapter 6 Notes — LaTeX Source](notes/chapter06_notes.tex)

Topics include memory coalescing, latency hiding, occupancy, memory-level parallelism, thread coarsening, arithmetic intensity, and optimization trade-offs.

---

## Exercises

- [Chapter 6 Exercises](exercises/CUDA_Chapter6_Exercises.md)
- [`corner_turning_benchmark.cu`](exercises/corner_turning_benchmark.cu)

The exercises include:

- column-major matrix multiplication and corner turning;
- coalesced vs. uncoalesced memory-access analysis;
- block-size reasoning for tiled matrix multiplication;
- floating-point operations per byte;
- a small benchmark comparing row-major, column-major, corner-turning, and padded shared-memory implementations.

---

## Thread-Coarsened GEMM Experiment

- [`coarse_matmul_benchmark.cu`](code/coarse_matmul_benchmark.cu)
- [Benchmark Notes](code/README.md)

The kernel uses:

```text
TILE_WIDTH    = 16
COARSE_FACTOR = 4
```

so each block covers 64 output columns.

The benchmark compares matrix sizes around execution boundaries such as:

```text
255 / 256 / 257
319 / 320 / 321
511 / 512 / 513
```

The experiment shows that GPU execution cost changes in discrete tile and phase units. Crossing a boundary can introduce additional padded work even when the mathematical problem size increases only slightly.

---

## Key Observations

### Memory layout and coalescing

A memory layout is not inherently fast or slow. Performance depends on how the thread mapping follows the contiguous memory direction.

For column-major matrix `B`, a conventional row-major-style thread mapping produces strided global-memory accesses. Corner turning reorganizes the load so neighboring threads access neighboring elements.

### Shared-memory padding

Corner turning can improve global-memory coalescing while introducing unfavorable shared-memory bank mapping.

Padding the shared-memory tile,

```cpp
float B_shared[TILE_WIDTH][TILE_WIDTH + 1];
```

changes the shared-memory stride and can substantially reduce bank conflicts.

### Thread coarsening

Thread coarsening increases data reuse and arithmetic intensity by allowing each thread to compute multiple output elements.

The main trade-off is:

```text
more reuse
    ↓
less global-memory traffic
    ↓
higher arithmetic intensity

but also

more per-thread work
    ↓
more registers / fewer blocks
    ↓
potentially lower occupancy and parallelism
```

### Performance analysis

Optimization should be driven by the actual bottleneck rather than by applying optimization rules mechanically.

A useful workflow is:

```text
understand the access pattern
        ↓
measure
        ↓
identify the bottleneck
        ↓
optimize
        ↓
measure again
```

---

## Build

Example compilation commands from the chapter directory:

```bash
nvcc -std=c++17 code/coarse_matmul_benchmark.cu -o coarse_matmul_benchmark
nvcc -std=c++17 exercises/corner_turning_benchmark.cu -o corner_turning_benchmark
```

On Windows, the generated executables will typically have the `.exe` extension.

---

## Main Takeaway

The central lesson of this chapter is that CUDA performance depends on the interaction between:

```text
data layout
    +
thread-to-data mapping
    +
global-memory access pattern
    +
shared-memory organization
    +
occupancy and available parallelism
```

The goal is not to maximize any single metric in isolation, but to reduce the dominant bottleneck while preserving enough parallelism to keep the GPU busy.
