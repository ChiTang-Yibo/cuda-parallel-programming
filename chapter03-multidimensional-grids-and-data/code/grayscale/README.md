# CUDA vs. NumPy CPU Grayscale Benchmark

This experiment compares a CUDA implementation of RGB-to-grayscale conversion with a NumPy CPU baseline. The goal is not only to compare computation time, but also to distinguish kernel execution from host-device transfer and full end-to-end program overhead.

## Files

- `gray_cuda_timed.cu` — CUDA implementation and timing
- `gray_cpu_timed.py` — NumPy CPU baseline
- `../../third_party/stb/stb_image.h` — third-party image loader
- `../../third_party/stb/stb_image_write.h` — third-party image writer

The CUDA program uses the `stb` headers for image I/O. They are stored separately under `third_party/stb/` because they are third-party dependencies rather than original project code.

## Experiment Setup

The input RGB image has:

- Width: 2557 pixels
- Height: 1524 pixels
- Total pixels: 3,896,868
- Repeated runs: 100

The CUDA program uses a two-dimensional launch configuration:

- Block size: \(16 \times 16\)
- Grid size: \(160 \times 96\)
- Total CUDA threads: 3,932,160

Each CUDA thread processes one image pixel. Since the image dimensions are not exact multiples of 16, the launch contains extra threads. These threads are filtered by the boundary check:

```cpp
if (col < width && row < height)
```

The basic thread-to-pixel mapping is:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

## Grayscale Operation

Both implementations use the same RGB-to-grayscale relation:

```math
L = 0.21R + 0.71G + 0.07B
```

The CUDA version assigns one thread to one pixel, while the CPU baseline performs the same operation using NumPy array operations.

## Timing Results

| Item | CUDA | NumPy CPU |
|---|---:|---:|
| Image load time | 36.79 ms | 84.33 ms |
| Compute total time, 100 runs | 3.03 ms | 3986.16 ms |
| Average compute time per run | 0.0303 ms | 39.8616 ms |
| Image save time | 166.25 ms | 106.85 ms |
| Total program time | 304.84 ms | 4236.24 ms |

## Speedup Analysis

The measured speedup depends on which part of the workflow is compared.

### 1. Kernel-only comparison

Comparing the NumPy CPU computation with the CUDA kernel execution alone:

```math
\text{Speedup}_{\text{kernel}}
=
\frac{39.8616}{0.0303}
\approx 1316
```

The CUDA kernel is therefore about **1316× faster** than the NumPy CPU computation for the grayscale arithmetic itself.

This number should be interpreted carefully because it excludes CUDA memory allocation, host-device transfers, and image I/O.

### 2. CUDA processing path including memory transfer

Including host-to-device transfer, kernel execution, and device-to-host transfer:

```math
T_{\text{CUDA, processing}}
=
1.5126 + 0.0303 + 0.5999
=
2.1428\ \text{ms}
```

Compared with the NumPy CPU compute time:

```math
\text{Speedup}_{\text{processing}}
=
\frac{39.8616}{2.1428}
\approx 18.6
```

So the CUDA processing path is approximately **18.6× faster** when transfer overhead is included.

### 3. End-to-end program comparison

Comparing the complete measured program times:

```math
\text{Speedup}_{\text{end-to-end}}
=
\frac{4236.24}{304.84}
\approx 13.9
```

The full CUDA program is therefore approximately **13.9× faster** in this experiment.

These three speedups describe different scopes and should not be interpreted as the same performance metric.

## Why the CUDA Kernel Is Faster

The grayscale operation is naturally data-parallel: each output pixel depends only on the RGB values of the corresponding input pixel.

The CUDA implementation maps this directly to GPU execution:

```text
one pixel
   ↓
one CUDA thread
   ↓
one grayscale output value
```

With nearly 3.9 million pixels, the GPU can execute a large number of these independent operations concurrently.

The NumPy implementation is already vectorized and should therefore be viewed as a **NumPy CPU baseline**, rather than as a pure Python serial loop. The comparison demonstrates the advantage of GPU parallelism for a highly parallel pixel-wise operation, while also showing that non-kernel overhead can significantly reduce the practical end-to-end speedup.

## Overhead Analysis

The CUDA kernel itself is very short, but the full program also includes:

- image loading;
- CUDA memory allocation;
- host-to-device transfer;
- device-to-host transfer;
- image saving.

For this simple operation, these overheads are large relative to the kernel execution time. In particular, PNG I/O and CUDA memory management contribute much more time than the grayscale kernel itself.

This illustrates an important GPU-programming principle:

> A very fast kernel does not automatically imply the same speedup for the complete application.

CUDA becomes more advantageous when data remains on the GPU across multiple operations, because repeated allocation and host-device transfer can then be avoided.

## Build and Run

From the `code/grayscale/` directory, compile the CUDA implementation with:

```bash
nvcc gray_cuda_timed.cu -I../../third_party/stb -o gray_cuda_timed
```

Example CUDA run:

```bash
gray_cuda_timed.exe input.png output_cuda.png 100
```

The Python baseline requires NumPy and Pillow:

```bash
pip install numpy pillow
```

Example CPU run:

```bash
python gray_cpu_timed.py input.png output_cpu.png 100
```

## Key Observation

This experiment demonstrates both sides of GPU acceleration:

- the grayscale kernel maps extremely well to CUDA because each pixel can be processed independently;
- the kernel itself achieves a very large computational speedup;
- host-device transfer, memory allocation, and image I/O reduce the practical application-level speedup;
- keeping data on the GPU across multiple processing stages would make better use of the available parallelism.

For this reason, kernel-only timing, processing-path timing, and end-to-end timing should always be reported separately.
