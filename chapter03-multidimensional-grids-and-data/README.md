# Chapter 3 — Multidimensional Grids and Data

Chapter 3 extends CUDA thread organization from one-dimensional vectors to multidimensional data such as images and matrices.

The main focus is the mapping

```text
multidimensional data
        ↓
multidimensional CUDA grid
        ↓
2D / 3D thread coordinates
        ↓
linear memory addresses
```

## Key Topics

- Multidimensional CUDA grids and blocks
- `dim3`
- `blockIdx`, `threadIdx`, `blockDim`, and `gridDim`
- Two-dimensional thread-to-data mapping
- Row-major storage
- Flattening 2D indices into linear memory
- Boundary handling for non-divisible image dimensions
- Neighborhood operations for image blur
- One-thread-per-output-element matrix multiplication

## Core Relations

### Two-dimensional global thread coordinates

```math
col = \texttt{blockIdx.x}\times\texttt{blockDim.x}+\texttt{threadIdx.x}
```

```math
row = \texttt{blockIdx.y}\times\texttt{blockDim.y}+\texttt{threadIdx.y}
```

### Row-major indexing

For a two-dimensional array with `width` columns,

```math
idx = row\times width+col
```

### Matrix multiplication

```math
P_{row,col}
=
\sum_k M_{row,k}N_{k,col}
```

A basic CUDA matrix-multiplication kernel assigns one output element of \(P\) to one CUDA thread.

## Projects

### 1. Grayscale Conversion

[`code/grayscale/`](code/grayscale/)

A pixel-wise RGB-to-grayscale example using a two-dimensional CUDA launch.

The project includes:

- CUDA implementation
- NumPy CPU baseline
- timing comparison
- thread-to-pixel mapping
- boundary handling
- kernel-only, transfer-inclusive, and end-to-end timing analysis

### 2. Image Blur

[`code/image_blur/`](code/image_blur/)

A neighborhood-based image-processing example.

Each thread computes one output pixel by averaging valid neighboring input pixels. This project introduces:

- local neighborhood access
- image-edge boundary conditions
- different blur radii
- CUDA vs. NumPy CPU timing comparison

### 3. Matrix Multiplication

[`code/matrix_multiplication/`](code/matrix_multiplication/)

A basic matrix-multiplication implementation that maps one CUDA thread to one output matrix element.

This example connects:

```text
output coordinate (row, col)
        ↓
one row of M
+
one column of N
        ↓
dot product
        ↓
P[row, col]
```

The project includes a CUDA implementation, a CPU/reference implementation, and accompanying notes or benchmark discussion.

## Notes

- [`notes/chapter03_notes.tex`](notes/chapter03_notes.tex) — LaTeX source
- [`notes/chapter03_notes.pdf`](notes/chapter03_notes.pdf) — compiled chapter notes

## Exercises

[`exercises/`](exercises/)

This directory contains the Chapter 3 exercise answers and CUDA coding exercises.

## Third-party Dependencies

The image-processing examples use the single-header `stb` libraries:

- [`third_party/stb/stb_image.h`](third_party/stb/stb_image.h)
- [`third_party/stb/stb_image_write.h`](third_party/stb/stb_image_write.h)

These files are third-party dependencies and retain their original license notices.

## Chapter Focus

The central idea of this chapter is that CUDA thread organization should follow the structure of the data whenever possible:

```text
vector  -> 1D threads
image   -> 2D threads
matrix  -> 2D threads
volume  -> 3D threads
```

Once each thread can correctly determine its multidimensional coordinate and corresponding memory location, more advanced CUDA techniques such as shared-memory tiling and optimized matrix multiplication can be built on top of the same mapping.
