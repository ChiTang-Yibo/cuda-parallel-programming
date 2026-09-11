# CUDA Tiled Matrix Multiplication Experiment

## 1. Experiment Objective

This experiment compares a naive CUDA matrix multiplication kernel with a shared-memory tiled implementation.

The main goals are to:

1. verify the correctness of tiled matrix multiplication;
2. measure the performance benefit of shared-memory data reuse;
3. verify that the boundary-handling logic works when matrix dimensions are not divisible by `TILE_WIDTH`;
4. relate the measured performance to global-memory traffic and arithmetic intensity.

The tiled kernel uses:

```cpp
TILE_WIDTH = 16
```

so each thread block contains `16 x 16 = 256` threads.

---

## 2. Performance Experiment Setup

For the main performance comparison, three square matrix sizes were tested:

- `512 x 512`
- `1024 x 1024`
- `2048 x 2048`

Each kernel was executed 20 times and the average kernel execution time was recorded.

The reported timing measures **kernel execution only**. Host-device memory transfers and memory-allocation overhead are not included.

GFLOP/s is calculated using the conventional GEMM operation count

$$
\mathrm{FLOPs} \approx 2MNK.
$$

---

## 3. Main Performance Results

| Matrix Size | Naive Time (ms) | Tiled Time (ms) | Naive GFLOP/s | Tiled GFLOP/s | Speedup |
|---|---:|---:|---:|---:|---:|
| 512 x 512 | 0.2198 | 0.1585 | 1221.13 | 1694.09 | 1.387x |
| 1024 x 1024 | 1.7679 | 1.4147 | 1214.69 | 1517.97 | 1.250x |
| 2048 x 2048 | 15.2427 | 10.1650 | 1127.09 | 1690.10 | 1.500x |

All GPU results passed the CPU reference correctness check.

For these three matrix sizes, the tiled kernel achieved approximately **1.25x–1.50x speedup** over the naive kernel.

The tiled implementation reached roughly **1.5–1.7 TFLOP/s**, while the naive implementation achieved about **1.1–1.2 TFLOP/s**.

---

## 4. Boundary Correctness Tests

The tiled kernel also contains boundary checks so that matrix dimensions do not need to be exact multiples of `TILE_WIDTH`.

Three additional cases were tested.

### Case 1: Large Non-Divisible Rectangular Matrices

```text
A: 513 x 509
B: 509 x 517
C: 513 x 517
Block: (16, 16)
Grid:  (33, 33)
```

| Kernel | Average Time (ms) | GFLOP/s | Correctness |
|---|---:|---:|---|
| Naive | 0.2144 | 1259.20 | PASSED |
| Tiled | 0.1626 | 1660.08 | PASSED |

Speedup:

$$
\frac{T_{\mathrm{naive}}}{T_{\mathrm{tiled}}}
=
1.3184.
$$

This test is important because all three matrix dimensions are not divisible by 16:

$$
513 = 32\times16+1,
$$

$$
509 = 31\times16+13,
$$

$$
517 = 32\times16+5.
$$

Therefore, it simultaneously tests incomplete output tiles and an incomplete final phase along the shared dimension.

---

### Case 2: Small `17 x 17` Matrices

```text
A: 17 x 17
B: 17 x 17
C: 17 x 17
Block: (16, 16)
Grid:  (2, 2)
```

| Kernel | Average Time (ms) | GFLOP/s | Correctness |
|---|---:|---:|---|
| Naive | 0.0048 | 2.0444 | PASSED |
| Tiled | 0.0085 | 1.1570 | PASSED |

Speedup:

$$
0.5659\times.
$$

The tiled kernel is slower in this very small case. This is expected because the workload is too small to amortize the overhead of shared-memory stores, synchronization, and execution of partially filled tiles.

This result shows that **tiling is not automatically faster for every matrix size**.

---

### Case 3: Small Rectangular Non-Divisible Matrices

```text
A: 31 x 23
B: 23 x 37
C: 31 x 37
Block: (16, 16)
Grid:  (3, 2)
```

| Kernel | Average Time (ms) | GFLOP/s | Correctness |
|---|---:|---:|---|
| Naive | 0.0054 | 9.7103 | PASSED |
| Tiled | 0.0045 | 11.6606 | PASSED |

Speedup:

$$
1.2008\times.
$$

This test further confirms that the boundary-handling logic works for rectangular matrices whose `M`, `K`, and `N` dimensions are all different.

---

## 5. Why Tiling Reduces Global-Memory Load Demand

For a tile width $T$, one block contains

$$
T^2
$$

threads.

In a naive implementation, every thread independently loads the input elements required for its dot product.

For a shared dimension of length $K$, the approximate software-level load demand per block is

$$
2KT^2
$$

floating-point elements.

In the tiled implementation, each phase loads only two $T\times T$ tiles:

$$
2T^2
$$

elements.

The number of phases is approximately

$$
\frac{K}{T}.
$$

Therefore, the total load demand per block becomes

$$
2T^2\frac{K}{T}
=
2KT.
$$

The idealized reduction factor is therefore

$$
\frac{2KT^2}{2KT}
=
T.
$$

For

$$
T=16,
$$

the software access model predicts approximately a **16x reduction in repeated global-memory load demand**.

However, this does **not** mean that actual DRAM traffic is necessarily reduced by exactly 16x. The naive kernel can already benefit from L1/L2 caches and other memory-system reuse.

---

## 6. Arithmetic Intensity

Tiling does not reduce the number of arithmetic operations. Its main benefit is increasing the amount of computation performed per byte of global-memory traffic.

### Naive Kernel

For one output element, the kernel performs approximately

$$
2K
$$

FLOPs and reads approximately

$$
2K
$$

floating-point values.

Since each `float` occupies 4 bytes,

$$
\mathrm{AI}_{\mathrm{naive}}
\approx
\frac{2K}{2K\times4}
=
0.25\ \mathrm{FLOP/B}.
$$

This simplified model ignores the final output store and hardware caching.

### Tiled Kernel

For one phase, two $T\times T$ tiles are loaded:

$$
2T^2
$$

floats, corresponding to

$$
8T^2
$$

bytes.

The block performs

$$
T^2
$$

thread-level dot-product fragments, each containing $T$ multiply-add operations:

$$
2T^3
$$

FLOPs.

Therefore,

$$
\mathrm{AI}_{\mathrm{tiled}}
\approx
\frac{2T^3}{8T^2}
=
\frac{T}{4}\ \mathrm{FLOP/B}.
$$

For `TILE_WIDTH = 16`,

$$
\mathrm{AI}_{\mathrm{tiled}}
\approx
4\ \mathrm{FLOP/B}.
$$

Thus, under the idealized model,

$$
0.25
\rightarrow
4\ \mathrm{FLOP/B},
$$

which corresponds to a factor of 16 increase in arithmetic intensity.

---

## 7. Interpretation of the Measured Speedup

The observed speedup of approximately **1.25x–1.50x** for the main square-matrix tests is much smaller than the idealized factor of 16 in load reuse.

This is reasonable for several reasons:

- the naive kernel can benefit from hardware caches;
- the tiled kernel introduces shared-memory load/store instructions;
- each phase requires synchronization;
- tiled indexing requires additional address calculations;
- the simple tiled kernel still computes only one output element per thread;
- the kernel does not yet use more advanced GEMM techniques such as register tiling, vectorized memory operations, double buffering, or Tensor Cores.

Therefore, the factor of 16 should be interpreted as an **idealized data-reuse / arithmetic-intensity improvement**, not as an expected 16x execution-time speedup.

---

## 8. Boundary Handling

When a matrix dimension is not divisible by `TILE_WIDTH`, the last block or phase contains invalid thread positions.

The kernel handles these positions by loading zero into shared memory when the corresponding global-memory coordinate is outside the valid matrix range.

Conceptually:

```cpp
if (valid_A_coordinate)
    A_shared[ty][tx] = A[...];
else
    A_shared[ty][tx] = 0.0f;

if (valid_B_coordinate)
    B_shared[ty][tx] = B[...];
else
    B_shared[ty][tx] = 0.0f;
```

The final output write is also guarded by a boundary condition.

The three non-divisible tests all passed the CPU reference check, confirming that this boundary-handling strategy works for the tested cases.

---

## 9. Conclusion

The experiment demonstrates three main properties of tiled CUDA matrix multiplication.

First, shared-memory tiling improves data reuse. Threads in the same block cooperatively load input tiles from global memory and then reuse those values during computation.

Second, tiling increases arithmetic intensity. In the simplified access model used here,

$$
\mathrm{AI}_{\mathrm{naive}}
\approx
0.25\ \mathrm{FLOP/B},
$$

while

$$
\mathrm{AI}_{\mathrm{tiled}}
\approx
\frac{T}{4}\ \mathrm{FLOP/B}.
$$

For `TILE_WIDTH = 16`, this gives approximately

$$
4\ \mathrm{FLOP/B}.
$$

Third, the boundary tests confirm that the implementation correctly handles matrix dimensions that are not multiples of the tile size.

Overall:

> **Tiling improves matrix-multiplication performance by increasing input-data reuse and reducing repeated global-memory load demand. It does not reduce the number of arithmetic operations, and its practical speedup depends on matrix size, cache behavior, synchronization overhead, and the efficiency of the kernel implementation.**

The current kernel should therefore be viewed as a correct and instructive baseline tiled GEMM implementation rather than a fully optimized GPU GEMM.
