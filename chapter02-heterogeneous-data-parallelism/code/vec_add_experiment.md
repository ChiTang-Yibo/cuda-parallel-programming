# CUDA Vector Addition Experiment

## 1. Objective

This experiment verifies the basic CUDA execution model introduced in Chapter 2 using a one-dimensional vector-addition kernel:

```cpp
C[i] = A[i] + B[i];
```

Each CUDA thread is mapped to one vector element through

```math
i = \texttt{blockIdx.x}\times\texttt{blockDim.x}+\texttt{threadIdx.x}.
```

The experiment focuses on three questions:

1. How does the vector size determine the number of CUDA blocks and launched threads?
2. Why is the boundary check `if (i < n)` necessary?
3. How does changing the block size change the grid geometry?

No performance timing is included here. The purpose is to understand execution configuration and correctness rather than benchmark GPU speed.

---

## 2. Test Environment

- GPU: **NVIDIA GeForce RTX 5070 Laptop GPU**
- Data type: `float`
- Operation: element-wise vector addition
- Correctness check: CPU-side verification
- Default block size for Experiment 1: **256 threads/block**

The program reports:

- vector size;
- block size;
- number of blocks;
- total launched threads;
- extra threads;
- memory used by the three host/device vectors;
- correctness result.

---

## 3. Launch Configuration

For a vector containing \(N\) elements and a block containing \(B\) threads,

```math
N_{\mathrm{blocks}}
=
\left\lceil
\frac{N}{B}
\right\rceil.
```

The total number of launched threads is

```math
N_{\mathrm{launched}}
=
N_{\mathrm{blocks}}\times B.
```

The number of extra threads is therefore

```math
N_{\mathrm{extra}}
=
N_{\mathrm{launched}}-N.
```

In C/C++, the ceiling division is implemented as

```cpp
int numBlocks = (n + blockSize - 1) / blockSize;
```

Since CUDA launches complete blocks, the grid can contain more threads than valid vector elements. Therefore, the kernel uses

```cpp
if (i < n) {
    C[i] = A[i] + B[i];
}
```

to prevent out-of-range memory accesses.

---

## 4. Experiment 1 — Changing the Vector Size

The block size was fixed at

```text
256 threads/block
```

while the vector size was varied.

| Vector Size \(N\) | Block Size | Blocks | Threads Launched | Extra Threads | Thread Utilization | Result |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,000 | 256 | 4 | 1,024 | 24 | 97.6562% | PASS |
| 4,096 | 256 | 16 | 4,096 | 0 | 100.0000% | PASS |
| 10,000 | 256 | 40 | 10,240 | 240 | 97.6562% | PASS |
| 1,048,576 | 256 | 4,096 | 1,048,576 | 0 | 100.0000% | PASS |
| 10,000,000 | 256 | 39,063 | 10,000,128 | 128 | 99.9987% | PASS |

### Observations

When the vector size is an exact multiple of the block size, no extra threads are created.

For example,

```math
4096 = 16\times256,
```

so the launch configuration is exactly

```text
16 blocks × 256 threads = 4096 threads.
```

The same occurs for

```math
1,048,576 = 4096\times256.
```

In contrast, \(N=1000\) is not divisible by 256:

```math
\left\lceil\frac{1000}{256}\right\rceil = 4,
```

so the GPU launches

```math
4\times256=1024
```

threads. The last 24 threads have no valid data element to process.

For \(N=10000\),

```math
40\times256=10240,
```

so 240 threads are outside the valid data range.

For the largest test,

```math
N=10,000,000,
```

the program launches 10,000,128 threads. Only 128 threads are extra, so the thread utilization is still approximately 99.999%.

This demonstrates an important property of CUDA launch configuration:

> The number of launched threads does not need to exactly equal the number of data elements. It only needs to be large enough to cover them, while the kernel boundary check rejects invalid thread indices.

The kernel code itself does not change when the vector size changes from 1,000 elements to 10,000,000 elements. Only the grid size changes.

---

## 5. Memory Scaling

Each vector contains \(N\) single-precision floating-point values. Since

```math
\texttt{sizeof(float)}=4\ \text{bytes},
```

the memory required by one vector is

```math
M_{\mathrm{vector}} = 4N\ \text{bytes}.
```

Three vectors \(A\), \(B\), and \(C\) therefore require

```math
M_{\mathrm{total}} = 12N\ \text{bytes}.
```

The measured values follow this relation exactly.

| Vector Size \(N\) | Memory per Vector | Total for A, B, C |
|---:|---:|---:|
| 1,000 | 4,000 bytes | 12,000 bytes |
| 4,096 | 16,384 bytes | 49,152 bytes |
| 10,000 | 40,000 bytes | 120,000 bytes |
| 1,048,576 | 4,194,304 bytes | 12,582,912 bytes |
| 10,000,000 | 40,000,000 bytes | 120,000,000 bytes |

For example, for

```math
N=1,048,576=2^{20},
```

one vector requires

```math
1,048,576\times4
=
4,194,304\ \text{bytes}
=
4\ \text{MiB},
```

and three vectors require 12 MiB.

Thus, the memory requirement grows linearly with the vector length.

---

## 6. Experiment 2 — Changing the Block Size

The vector size was fixed at

```text
N = 1,000,000
```

while the number of threads per block was varied.

| Vector Size \(N\) | Block Size | Blocks | Threads Launched | Extra Threads | Thread Utilization | Result |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,000,000 | 64 | 15,625 | 1,000,000 | 0 | 100.0000% | PASS |
| 1,000,000 | 128 | 7,813 | 1,000,064 | 64 | 99.9936% | PASS |
| 1,000,000 | 256 | 3,907 | 1,000,192 | 192 | 99.9808% | PASS |
| 1,000,000 | 512 | 1,954 | 1,000,448 | 448 | 99.9552% | PASS |

### Observations

Increasing the block size reduces the number of blocks required to cover the same vector.

The relationship is approximately

```math
N_{\mathrm{blocks}}
\propto
\frac{1}{B}.
```

For example:

```text
64 threads/block   -> 15,625 blocks
128 threads/block  ->  7,813 blocks
256 threads/block  ->  3,907 blocks
512 threads/block  ->  1,954 blocks
```

This is purely a change in the organization of the grid. The total amount of useful work remains one vector addition per valid element.

The number of extra threads depends on whether \(N\) is divisible by the selected block size.

For a block size of 64,

```math
1,000,000 = 15,625\times64,
```

so no extra threads are needed.

For a block size of 512,

```math
1954\times512=1,000,448,
```

so 448 threads are outside the valid range.

However, even in this case, the fraction of extra threads is very small:

```math
\frac{448}{1,000,448}\times100\% \approx 0.0448\%.
```

Therefore, the extra threads introduced by ceiling division are generally a correctness issue rather than a major computational overhead for large vectors.

---

## 7. Correctness

Every tested configuration produced

```text
Result : PASS
```

and the sample value was consistently

```text
C[123] = 369.000000
```

because the input initialization uses

```cpp
A[i] = i;
B[i] = 2 * i;
```

and therefore

```math
C[123]
=
123+2\times123
=
369.
```

The successful tests confirm that the same kernel works correctly across:

- different vector lengths;
- different numbers of blocks;
- different block sizes;
- configurations with and without extra threads.

---

## 8. What This Experiment Demonstrates

The experiment connects the CUDA programming model directly to the observed launch configuration:

```text
vector size N
      ↓
choose block size B
      ↓
compute ceil(N / B)
      ↓
launch complete thread blocks
      ↓
each thread computes a global index
      ↓
if (i < N), process one element
```

Three main conclusions follow.

### 8.1 Grid size scales with the problem size

The kernel does not need to be rewritten when \(N\) changes. The host changes the number of blocks, allowing the same GPU kernel to process datasets ranging from thousands to millions of elements.

### 8.2 Complete blocks can create extra threads

CUDA launches whole blocks rather than an arbitrary number of individual threads. Therefore,

```math
N_{\mathrm{launched}} \ge N.
```

The boundary check ensures that extra threads do not access invalid memory.

### 8.3 Block size changes grid geometry

Changing the block size changes the number of blocks and the number of extra threads, but all tested configurations still produce the same correct result.

At this stage, the experiment does **not** establish which block size is fastest. Performance depends on additional GPU concepts such as warps, memory behavior, occupancy, instruction scheduling, and resource usage. Those issues require proper GPU timing and profiling and should be analyzed separately.

---

## 9. Key Takeaways

- One CUDA thread is mapped to one vector element.
- The global thread index is

```math
i=
\texttt{blockIdx.x}\times\texttt{blockDim.x}
+
\texttt{threadIdx.x}.
```

- The number of blocks is computed using ceiling division.
- CUDA launches complete blocks, so the number of threads can exceed the number of data elements.
- `if (i < n)` prevents the extra threads from accessing invalid memory.
- Increasing the problem size mainly increases the grid size; the kernel itself remains unchanged.
- Changing the block size changes the grid organization but does not change the mathematical result.
- All nine tested configurations passed the CPU-side correctness check.
- This experiment validates CUDA execution configuration and correctness, but it is not yet a performance benchmark.
