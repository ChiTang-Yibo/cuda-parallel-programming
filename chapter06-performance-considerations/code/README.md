# CUDA Chapter 6 — Coarsened GEMM Benchmark Summary

## 1. Purpose

This experiment benchmarks the thread-coarsened tiled matrix multiplication kernel developed in Chapter 6.

The goals are to:

- compare CPU and GPU execution time for different square matrix sizes;
- compare useful floating-point work with the approximate arithmetic work executed by the GPU kernel;
- study matrix sizes immediately below, exactly at, and immediately above multiples of 64;
- observe how tiling and thread coarsening create discrete execution boundaries;
- verify GPU correctness against a CPU reference implementation.

The kernel configuration is:

```math
T = \texttt{TILE\_WIDTH} = 16
```
```math
C = \texttt{COARSE\_FACTOR} = 4
```
Therefore, one coarse block covers:

```math
CT = 4 \times 16 = 64
```
output columns.

---

## 2. Benchmark Configuration

The tested square matrix sizes were:

```text
255, 256, 257
319, 320, 321
511, 512, 513
```

These groups compare matrix sizes immediately below, exactly at, and immediately above multiples of 64.

The benchmark used:

- 2 GPU warm-up iterations;
- 20 GPU timing iterations;
- 1 CPU timing iteration;
- CUDA events for GPU kernel timing;
- `std::chrono` for CPU timing;
- a CPU reference GEMM for numerical verification.

GPU timing measures **kernel execution only** and does not include host-to-device or device-to-host transfers.

---

## 3. FLOP Definitions

For a square $N \times N$ GEMM, the useful mathematical work is approximated by:

```math
F_{\text{useful}} = 2N^3
```
One multiplication and one addition are counted as two floating-point operations.

For the coarsened kernel, the benchmark also estimates the arithmetic work executed by all launched threads:

```math
F_{\text{kernel}}
\approx
2
\times
N_{\text{launched threads}}
\times
C
\times
N_{\text{phases}}
\times
T
```
This estimate includes arithmetic performed on zero-padded boundary regions.

The useful-work efficiency is defined as:

```math
\text{Efficiency}
=
\frac{F_{\text{useful}}}{F_{\text{kernel}}}
\times 100\%
```
This quantity helps reveal wasted work caused by partial tiles and partial coarse blocks.

---

## 4. Experimental Results

| N | Multiple of 64 | Useful GF | Kernel GF | Efficiency | CPU ms | CPU GFLOP/s | GPU ms | GPU GFLOP/s | Speedup | Max Error | Check |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 255 | No  | 0.033 | 0.034 | 98.8%  | 31.410  | 1.06 | 0.0253 | 1311.73 | 1242.41 | 6.20e-06 | PASS |
| 256 | Yes | 0.034 | 0.034 | 100.0% | 34.045  | 0.99 | 0.0250 | 1342.18 | 1361.78 | 5.72e-06 | PASS |
| 257 | No  | 0.034 | 0.047 | 71.7%  | 32.226  | 1.05 | 0.0358 | 948.09  | 899.97  | 5.72e-06 | PASS |
| 319 | No  | 0.065 | 0.066 | 99.1%  | 61.538  | 1.06 | 0.0399 | 1628.89 | 1543.95 | 7.63e-06 | PASS |
| 320 | Yes | 0.066 | 0.066 | 100.0% | 66.572  | 0.98 | 0.0398 | 1645.77 | 1671.78 | 9.54e-06 | PASS |
| 321 | No  | 0.066 | 0.087 | 76.3%  | 63.247  | 1.05 | 0.0538 | 1229.49 | 1175.48 | 7.63e-06 | PASS |
| 511 | No  | 0.267 | 0.268 | 99.4%  | 262.156 | 1.02 | 0.1485 | 1796.48 | 1764.78 | 1.14e-05 | PASS |
| 512 | Yes | 0.268 | 0.268 | 100.0% | 273.231 | 0.98 | 0.1477 | 1817.74 | 1850.21 | 1.14e-05 | PASS |
| 513 | No  | 0.270 | 0.321 | 84.1%  | 263.018 | 1.03 | 0.1702 | 1586.87 | 1545.77 | 1.14e-05 | PASS |

All tested cases passed numerical verification.

---

## 5. Main Observation: Execution Cost Changes at Tile Boundaries

GPU execution cost changes in discrete steps rather than continuously with matrix size.

For this coarsened kernel:

```math
\texttt{gridDim.x}
=
\left\lceil
\frac{W}{CT}
\right\rceil
=
\left\lceil
\frac{W}{64}
\right\rceil
```
```math
\texttt{gridDim.y}
=
\left\lceil
\frac{H}{T}
\right\rceil
=
\left\lceil
\frac{H}{16}
\right\rceil
```
and:

```math
N_{\text{phases}}
=
\left\lceil
\frac{K}{16}
\right\rceil
```
For square matrices, increasing the size from $64m$ to $64m+1$ may simultaneously increase:

- the number of coarse blocks in the $x$ direction;
- the number of block rows;
- the number of reduction phases.

Therefore, increasing the matrix dimension by only one can cause a much larger increase in executed GPU work.

---

## 6. Example: N = 256 vs. N = 257

For:

```math
N = 256
```
the launch geometry is:

```math
\texttt{gridDim.x}=4
```
```math
\texttt{gridDim.y}=16
```
```math
N_{\text{phases}}=16
```
All dimensions divide the execution granularity exactly, so:

```math
\text{Efficiency}=100\%
```
For:

```math
N = 257
```
the geometry becomes:

```math
\texttt{gridDim.x}=5
```
```math
\texttt{gridDim.y}=17
```
```math
N_{\text{phases}}=17
```
Although the mathematical problem size increases only slightly, the estimated kernel work increases from approximately:

```math
0.034\ \text{GF}
```
to:

```math
0.047\ \text{GF}
```
The useful-work efficiency falls to:

```math
71.7\%
```
The measured GPU time correspondingly rises from:

```math
0.0250\ \text{ms}
```
to:

```math
0.0358\ \text{ms}
```
This is a clear example of **boundary-induced padded work**.

---

## 7. Why N = 255 Behaves Like N = 256

Although 255 is not a multiple of 64:

```math
\left\lceil\frac{255}{64}\right\rceil
=
\left\lceil\frac{256}{64}\right\rceil
=
4
```
and:

```math
\left\lceil\frac{255}{16}\right\rceil
=
\left\lceil\frac{256}{16}\right\rceil
=
16
```
Therefore, the two sizes use essentially the same launch geometry.

Their measured GPU times are also nearly identical:

```text
N = 255: 0.0253 ms
N = 256: 0.0250 ms
```

This shows that the important question is not simply whether a matrix dimension is divisible by 64.

A better question is:

> Does the matrix size cross a new execution-tile boundary?

---

## 8. Boundary Overhead Becomes Relatively Smaller for Larger Matrices

The useful-work efficiencies immediately above a 64 boundary were:

```text
N = 257: 71.7%
N = 321: 76.3%
N = 513: 84.1%
```

The efficiency improves as matrix size increases because one extra boundary region becomes a smaller fraction of the total workload.

Therefore:

```math
\boxed{
\text{relative padding overhead decreases as matrix size grows}
}
```
The same discrete tile-boundary mechanism still exists, but its relative cost becomes smaller.

---

## 9. Useful GFLOP/s vs. Executed Arithmetic Work

The reported GPU GFLOP/s is based on:

```math
F_{\text{useful}} = 2N^3
```
Therefore, cases such as 257, 321, and 513 show a large decrease in useful GFLOP/s.

However, the GPU is still executing arithmetic on padded regions.

For example, at $N=257$, the useful work is about:

```math
0.034\ \text{GF}
```
while the estimated kernel work is about:

```math
0.047\ \text{GF}
```
This means the lower useful GFLOP/s is largely caused by extra arithmetic that does not contribute to the final matrix product.

The raw arithmetic execution capability of the GPU changes much less than the useful GFLOP/s suggests.

---

## 10. CPU Results

The CPU reference implementation remained close to:

```math
1\ \text{GFLOP/s}
```
in these tests.

The exact-multiple cases 256, 320, and 512 were slightly slower than neighboring sizes.

However, this benchmark used:

```cpp
constexpr int CPU_REPEAT = 1;
```

and the CPU implementation is a naive triple-loop reference GEMM.

Therefore, the current data is not sufficient to conclude that the slowdown is caused by cache conflicts or another specific microarchitectural effect.

Possible influences include:

- CPU frequency variation;
- OS scheduling;
- background processes;
- cache state;
- strided accesses to matrix `N`;
- compiler vectorization behavior.

A more careful CPU study should use repeated measurements and a more controlled implementation.

---

## 11. Interpretation of the Reported Speedup

The measured speedup is very large because the comparison is between:

- a simple single-threaded CPU reference implementation;
- GPU kernel execution time only.

The GPU timing does **not** include:

- host-to-device transfers;
- device-to-host transfers;
- allocation overhead.

Therefore, the reported speedup should be interpreted only as:

```math
\boxed{
\text{naive CPU reference time}
\quad\text{vs.}\quad
\text{GPU kernel-only time}
}
```
It is not a general claim that the GPU is intrinsically thousands of times faster than the CPU.

A more balanced CPU/GPU study would later include:

- an optimized CPU implementation;
- multiple CPU repetitions;
- SIMD/vectorization analysis;
- multithreading or BLAS;
- optional inclusion of GPU transfer overhead.

---

## 12. Main Conclusions

1. **GPU execution granularity is discrete.**  
   Kernel work changes according to block, tile, and phase boundaries rather than continuously with matrix size.

2. **Crossing an execution boundary can create a performance cliff.**  
   Moving from $64m$ to $64m+1$ can require additional coarse blocks, block rows, and reduction phases.

3. **Padding work matters.**  
   Boundary threads may load zeros but still execute arithmetic.

4. **A non-divisible size is not automatically inefficient.**  
   Sizes such as 255 and 319 remain close to the preceding execution boundary and therefore retain high useful-work efficiency.

5. **Relative padding overhead decreases for larger matrices.**  
   The boundary region becomes a smaller fraction of the total computation.

6. **Useful GFLOP/s and executed arithmetic throughput are different quantities.**  
   A reduction in useful GFLOP/s can result from extra padded work rather than a major reduction in the GPU's raw arithmetic rate.

7. **CPU/GPU benchmark results must be interpreted carefully.**  
   A naive CPU reference and GPU kernel-only timing are useful for learning, but they are not yet a fully balanced performance comparison.

---

## 13. Future Work

Future CUDA operator benchmarks can extend this experiment by:

- varying $H$, $K$, and $W$ independently;
- isolating `gridDim.x`, `gridDim.y`, and reduction-phase boundary effects;
- comparing naive, tiled, coarsened, and vectorized kernels;
- measuring end-to-end GPU time including transfers;
- testing multiple coarsening factors;
- studying register pressure and occupancy;
- using Nsight Compute to identify bottlenecks;
- comparing against optimized CPU implementations;
- exploring CPU cache behavior, SIMD, and multithreading.

This experiment provides a useful transition from Chapter 6 concepts to more systematic CUDA operator benchmarking and performance engineering.
