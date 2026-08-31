# CUDA Chapter 4 Exercises
## Complete Questions, Answers, and Explanations

This document contains the complete questions, final answers, and concise explanations for the Chapter 4 exercises.

Unless otherwise stated:

```math
1\ \text{warp} = 32\ \text{threads}.
```

---

# Problem 1

## Complete question

Consider the following CUDA kernel and host function:

```cpp
__global__ void foo_kernel(int* a, int* b) {
    unsigned int i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }

    if (i % 2 == 0) {
        a[i] = b[i] * 2;
    }

    for (unsigned int j = 0;
         j < 5 - (i % 3);
         ++j) {
        b[i] += j;
    }
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;

    foo_kernel<<<
        (N + 128 - 1) / 128,
        128
    >>>(a_d, b_d);
}
```

Answer:

1. How many warps are in one block?
2. How many warps are in the entire grid?
3. For `b[i] = a[i] + 1`, how many warps are active and divergent, and what are the SIMD efficiencies of warps 0, 1, and 3 of block 0?
4. For `a[i] = b[i] * 2`, how many warps are active and divergent, and what is the SIMD efficiency of warp 0 of block 0?
5. For the loop, how many iterations are non-divergent and how many are divergent?

## Final answers

| Part | Answer |
|---|---:|
| (a) | 4 warps per block |
| (b) | 32 warps in the grid |
| (c)(i) | 24 active warps |
| (c)(ii) | 16 divergent warps |
| (c)(iii) | 100% |
| (c)(iv) | 25% |
| (c)(v) | 75% |
| (d)(i) | 32 active warps |
| (d)(ii) | 32 divergent warps |
| (d)(iii) | 50% |
| (e)(i) | 3 iterations without divergence |
| (e)(ii) | 2 iterations with divergence |

## Explanation

The block size is 128 threads. The number of blocks is

```math
N_{\text{blocks}}
=
\frac{1024+128-1}{128}
=
8.
```

Therefore the grid contains

```math
8\times128=1024
```

threads.

Since a warp contains 32 threads,

```math
\frac{128}{32}=4
```

warps are created per block, and

```math
8\times4=32
```

warps are created in the grid.

The four warps in each block are:

- warp 0: `threadIdx.x = 0–31`
- warp 1: `threadIdx.x = 32–63`
- warp 2: `threadIdx.x = 64–95`
- warp 3: `threadIdx.x = 96–127`

### Part (c)

The first condition is

```cpp
if (threadIdx.x < 40 || threadIdx.x >= 104)
```

Within each block:

| Warp | Threads | Active threads | State |
|---|---:|---:|---|
| warp 0 | 0–31 | 32 | fully active |
| warp 1 | 32–63 | 8 | divergent |
| warp 2 | 64–95 | 0 | inactive |
| warp 3 | 96–127 | 24 | divergent |

Thus each block has 3 active warps and 2 divergent warps:

```math
8\times3=24
```

active warps, and

```math
8\times2=16
```

divergent warps.

For one instruction, the simplified SIMD efficiency is

```math
\text{SIMD efficiency}
=
\frac{\text{active threads}}{32}\times100\%.
```

Therefore:

```math
\text{warp 0: }
\frac{32}{32}\times100\%
=
100\%.
```

```math
\text{warp 1: }
\frac{8}{32}\times100\%
=
25\%.
```

```math
\text{warp 3: }
\frac{24}{32}\times100\%
=
75\%.
```

### Part (d)

The second condition is

```cpp
if (i % 2 == 0)
```

Because each warp contains alternating even and odd global indices, every warp contains 16 active threads and 16 inactive threads.

Therefore all 32 warps are active and all 32 warps diverge.

For warp 0:

```math
\frac{16}{32}\times100\%
=
50\%.
```

### Part (e)

The loop bound is

```cpp
5 - (i % 3)
```

so:

| `i % 3` | Iterations |
|---:|---:|
| 0 | 5 |
| 1 | 4 |
| 2 | 3 |

For `j = 0, 1, 2`, all threads are still executing the loop, so these three iterations are non-divergent.

For `j = 3` and `j = 4`, only subsets of the warp remain active, so these two iterations are divergent.

---

# Problem 2

## Complete question

For vector addition:

- vector length = 2000;
- one thread computes one output element;
- block size = 512 threads.

How many threads are launched in the grid?

## Final answer

```math
\boxed{2048\ \text{threads}}
```

## Explanation

The required number of blocks is

```math
\left\lceil
\frac{2000}{512}
\right\rceil
=
4.
```

Therefore,

```math
4\times512=2048
```

threads are launched.

The number of extra threads is

```math
2048-2000=48.
```

A boundary check prevents these threads from accessing invalid elements:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < 2000) {
    C[i] = A[i] + B[i];
}
```

---

# Problem 3

## Complete question

For the configuration in Problem 2, how many warps diverge because of the boundary check?

## Final answer

```math
\boxed{1\ \text{divergent warp}}
```

## Explanation

The grid contains

```math
\frac{2048}{32}=64
```

warps.

The valid thread indices are 0–1999.

Warp 62 contains threads 1984–2015:

- 1984–1999 are valid;
- 2000–2015 are invalid.

Therefore warp 62 diverges.

Warp 63 contains threads 2016–2047. All of them fail the boundary condition, so the warp is fully inactive for the vector-addition statement but is not divergent.

Thus only one warp diverges.

---

# Problem 4

## Complete question

A hypothetical block contains 8 threads. Their execution times before a barrier are:

```text
2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9 microseconds
```

What percentage of the threads' total elapsed time is spent waiting at the barrier?

## Final answer

```math
\boxed{17.08\%}
```

## Explanation

The slowest thread reaches the barrier after

```math
3.0\ \mu\text{s}.
```

The total waiting time is

```math
\begin{aligned}
T_{\text{wait}}
={}&(3.0-2.0)+(3.0-2.3)+(3.0-3.0)\\
&+(3.0-2.8)+(3.0-2.4)+(3.0-1.9)\\
&+(3.0-2.6)+(3.0-2.9)\\
={}&4.1\ \mu\text{s}.
\end{aligned}
```

The total elapsed thread time is

```math
8\times3.0
=
24.0\ \mu\text{s}.
```

Therefore,

```math
\frac{4.1}{24.0}\times100\%
=
17.08\%.
```

---

# Problem 5

## Complete question

A programmer says that if a kernel uses only 32 threads per block, `__syncthreads()` can always be removed wherever barrier synchronization is needed.

Is this a good idea?

## Final answer

**No. This is not generally safe.**

## Explanation

A 32-thread block normally consists of one warp, but program correctness should not rely on the assumption that all threads always advance in strict lockstep.

On Volta and later architectures, independent thread scheduling makes such assumptions particularly unsafe.

If threads exchange data through shared memory, explicit synchronization may still be required:

```cpp
__shared__ float s[32];

s[threadIdx.x] = input[threadIdx.x];

__syncwarp();

output[threadIdx.x] =
    s[(threadIdx.x + 1) % 32];
```

For synchronization that is strictly warp-local, `__syncwarp()` may be appropriate.

For block-wide synchronization, use `__syncthreads()`.

A block size of 32 does not by itself justify removing synchronization.

---

# Problem 6

## Complete question

An SM supports:

- at most 1536 threads;
- at most 4 resident blocks.

Which block size gives the largest number of resident threads?

1. 128 threads/block
2. 256 threads/block
3. 512 threads/block
4. 1024 threads/block

## Final answer

```math
\boxed{512\ \text{threads per block}}
```

## Explanation

| Threads/block | Resident blocks | Resident threads |
|---:|---:|---:|
| 128 | 4 | 512 |
| 256 | 4 | 1024 |
| 512 | 3 | 1536 |
| 1024 | 1 | 1024 |

For 512 threads per block:

```math
3\times512
=
1536.
```

This fills all available thread slots.

---

# Problem 7

## Complete question

A device supports:

- up to 64 blocks per SM;
- up to 2048 threads per SM.

Determine whether each assignment is possible and calculate occupancy.

## Final answers

| Assignment | Possible? | Occupancy |
|---|---|---:|
| 8 blocks × 128 threads | Yes | 50% |
| 16 blocks × 64 threads | Yes | 50% |
| 32 blocks × 32 threads | Yes | 50% |
| 64 blocks × 32 threads | Yes | 100% |
| 32 blocks × 64 threads | Yes | 100% |

## Explanation

Occupancy is

```math
\text{occupancy}
=
\frac{\text{resident threads}}{2048}
\times100\%.
```

The resident-thread counts are:

```math
8\times128=1024
\quad\Rightarrow\quad
50\%.
```

```math
16\times64=1024
\quad\Rightarrow\quad
50\%.
```

```math
32\times32=1024
\quad\Rightarrow\quad
50\%.
```

```math
64\times32=2048
\quad\Rightarrow\quad
100\%.
```

```math
32\times64=2048
\quad\Rightarrow\quad
100\%.
```

All assignments satisfy both the block and thread limits.

---

# Problem 8

## Complete question

A GPU has:

- 2048 threads per SM;
- 32 blocks per SM;
- 65,536 registers per SM.

Determine whether each configuration can reach full occupancy:

1. 128 threads/block, 30 registers/thread
2. 32 threads/block, 29 registers/thread
3. 256 threads/block, 34 registers/thread

## Final answers

| Configuration | Full occupancy? | Limiting factor |
|---|---|---|
| 128 threads/block, 30 registers/thread | Yes | None |
| 32 threads/block, 29 registers/thread | No | Block limit |
| 256 threads/block, 34 registers/thread | No | Register capacity |

## Explanation

### Configuration 1

Full occupancy requires

```math
\frac{2048}{128}=16
```

blocks.

This satisfies the 32-block limit.

The required registers are

```math
2048\times30
=
61{,}440
<
65{,}536.
```

Therefore full occupancy is possible.

### Configuration 2

Full occupancy would require

```math
\frac{2048}{32}
=
64
```

blocks.

The SM supports only 32 blocks, so at most

```math
32\times32
=
1024
```

threads can be resident.

Thus,

```math
\text{occupancy}
=
\frac{1024}{2048}
=
50\%.
```

The limiting factor is the maximum number of resident blocks.

### Configuration 3

Full occupancy would require

```math
2048\times34
=
69{,}632
```

registers, which exceeds the available 65,536 registers.

Each block requires

```math
256\times34
=
8704
```

registers.

Thus the maximum number of complete resident blocks is

```math
\left\lfloor
\frac{65{,}536}{8704}
\right\rfloor
=
7.
```

The resident thread count is

```math
7\times256
=
1792.
```

Therefore,

```math
\text{occupancy}
=
\frac{1792}{2048}\times100\%
=
87.5\%.
```

The limiting factor is register capacity.

---

# Problem 9

## Complete question

A student claims to have multiplied two $1024\times1024$ matrices with a CUDA kernel using a $32\times32$ thread block.

The device supports:

- at most 512 threads per block;
- at most 8 blocks per SM.

Each thread computes one output element.

Is the launch configuration valid?

## Final answer

**No. The launch configuration is invalid.**

## Explanation

A $32\times32$ block contains

```math
32\times32
=
1024
```

threads.

But the device allows at most 512 threads per block:

```math
1024>512.
```

Therefore the launch is illegal on the stated device.

A valid alternative is

```cpp
dim3 block(16, 16);
```

which contains

```math
16\times16
=
256
```

threads.

For a $1024\times1024$ output matrix:

```cpp
dim3 grid(
    (1024 + 16 - 1) / 16,
    (1024 + 16 - 1) / 16
);
```

which gives a $64\times64$ grid.

Another valid block is:

```cpp
dim3 block(32, 16);
```

because

```math
32\times16
=
512
```

threads exactly match the device limit.

---

# Complete Answer Summary

| Question | Final answer |
|---|---|
| 1(a) | 4 warps per block |
| 1(b) | 32 warps in the grid |
| 1(c)(i) | 24 active warps |
| 1(c)(ii) | 16 divergent warps |
| 1(c)(iii) | 100% |
| 1(c)(iv) | 25% |
| 1(c)(v) | 75% |
| 1(d)(i) | 32 active warps |
| 1(d)(ii) | 32 divergent warps |
| 1(d)(iii) | 50% |
| 1(e)(i) | 3 iterations without divergence |
| 1(e)(ii) | 2 iterations with divergence |
| 2 | 2048 threads |
| 3 | 1 divergent warp |
| 4 | 17.08% |
| 5 | No; explicit synchronization may still be necessary |
| 6 | 512 threads per block |
| 7(a) | Possible, 50% |
| 7(b) | Possible, 50% |
| 7(c) | Possible, 50% |
| 7(d) | Possible, 100% |
| 7(e) | Possible, 100% |
| 8(a) | Full occupancy is possible |
| 8(b) | No; limited by blocks per SM |
| 8(c) | No; limited by registers, maximum 87.5% occupancy |
| 9 | Invalid because 32 × 32 = 1024 > 512 threads/block |
