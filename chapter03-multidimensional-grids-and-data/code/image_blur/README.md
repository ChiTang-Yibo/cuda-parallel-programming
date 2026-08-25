# CUDA vs. NumPy CPU Image Blur Benchmark

This project compares a CUDA implementation of RGB image blur with a NumPy CPU baseline. The main goal is to connect the Chapter 3 concepts of two-dimensional thread mapping, neighborhood operations, boundary handling, and GPU parallelism with an actual image-processing benchmark.

## Files

- `blur_cuda_timed.cu` — CUDA image-blur implementation with timing
- `blur_cpu_timed.py` — NumPy CPU baseline
- `../../third_party/stb/stb_image.h` — third-party image loader used by the CUDA program
- `../../third_party/stb/stb_image_write.h` — third-party image writer used by the CUDA program

The `stb` headers are stored under `third_party/stb/` because they are external dependencies rather than original project code.

## Blur Operation

Each output pixel is computed by averaging the valid pixels inside a square neighborhood centered on the corresponding input pixel.

For blur radius \(r\), the window width is

```math
W = 2r + 1
```

so the two tested configurations are:

| Blur Radius | Blur Window | Maximum Pixels per Output Pixel |
|---:|---:|---:|
| 3 | 7 × 7 | 49 |
| 5 | 11 × 11 | 121 |

For pixels near an image boundary, some neighborhood positions lie outside the image. Both implementations ignore these invalid positions and average only valid pixels.

The CUDA kernel therefore checks:

```cpp
if (curRow >= 0 && curRow < height &&
    curCol >= 0 && curCol < width)
```

before reading each neighboring pixel.

## CUDA Thread Mapping

The CUDA implementation uses one GPU thread for one output pixel.

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

The benchmark uses:

- block size: **16 × 16**
- input image size: **2557 × 1524**
- grid size: **160 × 96**
- total pixels: **3,896,868**
- total CUDA threads launched: **3,932,160**
- repeated kernel runs: **100**

Because the image dimensions are not exact multiples of 16, the launch contains extra threads. The outer boundary check prevents these threads from accessing invalid pixels:

```cpp
if (col < width && row < height)
```

## Experiment Results

### Radius = 3, Window = 7 × 7

| Stage | CUDA | NumPy CPU |
|---|---:|---:|
| CUDA initialization | 3000.56 ms | N/A |
| Image load | 37.6225 ms | 84.6518 ms |
| Device memory allocation | 1.1563 ms | N/A |
| Host-to-device copy | 1.9079 ms | N/A |
| Compute total, 100 runs | 45.0306 ms | 94657.6139 ms |
| Average compute per run | 0.450306 ms | 946.5761 ms |
| Device-to-host copy | 1.8718 ms | N/A |
| Image save | 348.8430 ms | 362.0126 ms |
| Measured workflow total | 3518.7300 ms | 95104.2783 ms |

### Radius = 5, Window = 11 × 11

| Stage | CUDA | NumPy CPU |
|---|---:|---:|
| CUDA initialization | 2963.86 ms | N/A |
| Image load | 35.5007 ms | 84.6518 ms |
| Device memory allocation | 0.5412 ms | N/A |
| Host-to-device copy | 1.5609 ms | N/A |
| Compute total, 100 runs | 106.2590 ms | 176014.9029 ms |
| Average compute per run | 1.06259 ms | 1760.1490 ms |
| Device-to-host copy | 1.4022 ms | N/A |
| Image save | 335.1810 ms | 344.0919 ms |
| Measured workflow total | 3449.5300 ms | 176443.6466 ms |

## Speedup Analysis

The performance result depends on which part of the workflow is being compared.

### 1. Compute-only speedup

For the 7 × 7 blur:

```math
\text{Speedup}_{7\times7}
=
\frac{946.5761}{0.450306}
\approx 2102
```

For the 11 × 11 blur:

```math
\text{Speedup}_{11\times11}
=
\frac{1760.1490}{1.06259}
\approx 1657
```

Therefore, the measured CUDA kernel is about **2102× faster** for the 7 × 7 blur and about **1657× faster** for the 11 × 11 blur when comparing only the repeated computation stage.

These numbers are kernel/compute comparisons and do not include CUDA initialization, image I/O, memory allocation, or host-device transfer.

### 2. Processing path including host-device transfer

For one processing pass, combining H2D transfer, one average kernel execution, and D2H transfer gives:

| Blur Window | CUDA Processing Time | NumPy CPU Compute | Approx. Speedup |
|---:|---:|---:|---:|
| 7 × 7 | 4.2300 ms | 946.5761 ms | 223.8× |
| 11 × 11 | 4.0257 ms | 1760.1490 ms | 437.2× |

This comparison shows how memory transfer reduces the apparent GPU advantage relative to kernel-only timing.

### 3. Repeated-workflow comparison

Using the measured workflow totals for the 100-run experiment:

```math
\frac{95104.2783}{3518.7300}
\approx 27.0
```

for radius 3, and

```math
\frac{176443.6466}{3449.5300}
\approx 51.2
```

for radius 5.

These totals correspond to the benchmark structure used here: image loading and saving occur once, while the blur computation is repeated 100 times.

## Why Blur Benefits from CUDA

Image blur contains much more work per output pixel than grayscale conversion.

For radius 3, an interior pixel can require up to 49 neighboring RGB pixels. For radius 5, it can require up to 121 neighboring RGB pixels.

The computation is still highly data-parallel:

```text
one output pixel
        ↓
one CUDA thread
        ↓
read local neighborhood
        ↓
accumulate valid RGB values
        ↓
divide by valid pixel count
        ↓
write one blurred pixel
```

Millions of output pixels can therefore be processed concurrently on the GPU.

The CPU implementation should be interpreted as a **NumPy CPU baseline**, not a pure Python pixel-by-pixel loop. It iterates over neighborhood offsets while using NumPy slice operations to accumulate image regions.

## Performance Interpretation

The benchmark demonstrates an important distinction between kernel performance and application performance.

The CUDA kernel itself executes in less than about 1.1 ms per run in these measurements, but the complete program also contains:

- CUDA runtime initialization;
- image loading;
- device memory allocation;
- host-to-device transfer;
- device-to-host transfer;
- image saving.

The measured CUDA initialization alone was about 3 seconds in these runs, while PNG saving also contributed hundreds of milliseconds.

Therefore:

> A large kernel-level speedup does not automatically imply the same end-to-end application speedup.

GPU acceleration becomes more useful when data can remain on the device across multiple processing stages and expensive initialization or transfer overhead can be amortized.

## Build and Run

From the `code/image_blur/` directory, compile the CUDA version with:

```bash
nvcc blur_cuda_timed.cu -I../../third_party/stb -o blur_cuda_timed
```

Example CUDA run for radius 3:

```bash
blur_cuda_timed.exe input.png output_r3.png 100 3
```

Example CUDA run for radius 5:

```bash
blur_cuda_timed.exe input.png output_r5.png 100 5
```

The CPU baseline requires NumPy and Pillow:

```bash
pip install numpy pillow
```

Run the CPU version with:

```bash
python blur_cpu_timed.py input.png 100
```

To reproduce both benchmark radii with the current CPU script, set:

```python
radius_list = [3, 5]
```

before running the benchmark.

## Key Observations

- Two-dimensional CUDA threads map naturally to image pixels.
- Image blur is a neighborhood operation rather than a one-input-to-one-output operation.
- Boundary pixels require explicit validity checks because they have incomplete neighborhoods.
- Increasing the blur radius increases the amount of computation per output pixel.
- Kernel-only timing shows a very large GPU advantage in this experiment.
- Host-device transfer, initialization, and image I/O significantly reduce application-level speedup.
- Kernel timing and end-to-end timing should therefore be reported separately.
