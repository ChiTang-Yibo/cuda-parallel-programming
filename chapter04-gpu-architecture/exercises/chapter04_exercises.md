# CUDA Chapter 4 Exercises  
## Complete Questions, Answers, and Explanations

This document contains three parts for every exercise:

1. **Complete question**
2. **Final answer**
3. **Detailed explanation**

Unless otherwise stated, one CUDA warp contains 32 threads.

---

# Problem 1

## Complete question

Consider the following CUDA kernel and the corresponding host function that calls it:

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

Answer the following questions.

### (a)

What is the number of warps per block?

### (b)

What is the number of warps in the grid?

### (c)

For the statement on line 04:

```cpp
b[i] = a[i] + 1;
```

1. How many warps in the grid are active?
2. How many warps in the grid are divergent?
3. What is the SIMD efficiency, in percent, of warp 0 of block 0?
4. What is the SIMD efficiency, in percent, of warp 1 of block 0?
5. What is the SIMD efficiency, in percent, of warp 3 of block 0?

### (d)

For the statement on line 07:

```cpp
a[i] = b[i] * 2;
```

1. How many warps in the grid are active?
2. How many warps in the grid are divergent?
3. What is the SIMD efficiency, in percent, of warp 0 of block 0?

### (e)

For the loop on line 09:

```cpp
for (unsigned int j = 0;
     j < 5 - (i % 3);
     ++j) {
    b[i] += j;
}
```

1. How many iterations have no divergence?
2. How many iterations have divergence?

---

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

---

## Detailed explanation

### Step 1: Determine the launch configuration

The block size is 128 threads:

\[
\text{threads per block}=128.
\]

The number of blocks is

\[
\frac{N+128-1}{128}
=
\frac{1024+127}{128}
=
8.
\]

Therefore, the grid contains 8 blocks and

\[
8\times128=1024
\]

threads.

Because one warp contains 32 threads, each block contains

\[
\frac{128}{32}=4
\]

warps.

The four warps in one block are:

- warp 0: `threadIdx.x = 0–31`
- warp 1: `threadIdx.x = 32–63`
- warp 2: `threadIdx.x = 64–95`
- warp 3: `threadIdx.x = 96–127`

The complete grid contains

\[
8\times4=32
\]

warps.

---

### Part (a): Number of warps per block

\[
\frac{128}{32}=4.
\]

Therefore,

\[
\boxed{4\text{ warps per block}}.
\]

---

### Part (b): Number of warps in the grid

\[
8\text{ blocks}\times4\text{ warps per block}
=
32.
\]

Therefore,

\[
\boxed{32\text{ warps in the grid}}.
\]

---

### Part (c): Statement on line 04

The statement is controlled by

```cpp
if (threadIdx.x < 40 || threadIdx.x >= 104)
```

The condition depends only on `threadIdx.x`, so the same pattern occurs in every block.

The threads that execute line 04 are:

\[
0\text{--}39
\]

and

\[
104\text{--}127.
\]

The warp behavior within each block is:

| Warp | Thread indices | Active threads | Behavior |
|---|---:|---:|---|
| warp 0 | 0–31 | 32 | fully active |
| warp 1 | 32–63 | 8 | divergent |
| warp 2 | 64–95 | 0 | fully inactive |
| warp 3 | 96–127 | 24 | divergent |

A warp is active if at least one thread executes the statement. Therefore, every block has three active warps: warps 0, 1, and 3.

\[
8\times3=24.
\]

Thus,

\[
\boxed{24\text{ active warps}}.
\]

A warp is divergent if some threads execute the statement and other threads in the same warp do not. In each block, warps 1 and 3 are divergent.

\[
8\times2=16.
\]

Thus,

\[
\boxed{16\text{ divergent warps}}.
\]

For a particular instruction, the simplified SIMD efficiency is

\[
\text{SIMD efficiency}
=
\frac{\text{active threads}}{32}
\times100\%.
\]

For warp 0:

\[
\frac{32}{32}\times100\%=100\%.
\]

Therefore,

\[
\boxed{100\%}.
\]

For warp 1, only threads 32–39 execute the statement, so 8 threads are active:

\[
\frac{8}{32}\times100\%=25\%.
\]

Therefore,

\[
\boxed{25\%}.
\]

For warp 3, only threads 104–127 execute the statement, so 24 threads are active:

\[
\frac{24}{32}\times100\%=75\%.
\]

Therefore,

\[
\boxed{75\%}.
\]

---

### Part (d): Statement on line 07

The condition is

```cpp
if (i % 2 == 0)
```

where

```cpp
i = blockIdx.x * blockDim.x + threadIdx.x;
```

Since the block size is 128, an even number, every warp contains alternating even and odd global indices.

Therefore, each warp contains:

- 16 threads with even `i`;
- 16 threads with odd `i`.

Every warp has at least one active thread, so all 32 warps in the grid are active:

\[
\boxed{32\text{ active warps}}.
\]

Every warp contains both active and inactive threads, so every warp diverges:

\[
\boxed{32\text{ divergent warps}}.
\]

For warp 0 of block 0, 16 of the 32 threads execute line 07:

\[
\frac{16}{32}\times100\%=50\%.
\]

Therefore,

\[
\boxed{50\%}.
\]

---

### Part (e): Loop on line 09

The loop condition is

```cpp
j < 5 - (i % 3)
```

The number of iterations depends on `i % 3`:

| \(i\bmod3\) | Number of iterations |
|---:|---:|
| 0 | 5 |
| 1 | 4 |
| 2 | 3 |

For the iterations with

\[
j=0,\ 1,\ 2,
\]

all threads are still inside the loop. Therefore, these three iterations do not diverge.

\[
\boxed{3\text{ iterations without divergence}}.
\]

For the iteration with \(j=3\), only threads requiring 4 or 5 iterations remain active. For the iteration with \(j=4\), only threads requiring 5 iterations remain active.

Therefore,

\[
\boxed{2\text{ iterations with divergence}}.
\]

---

# Problem 2

## Complete question

For a vector addition, assume that:

- the vector length is 2000;
- each thread calculates one output element;
- the thread block size is 512 threads.

How many threads will be in the grid?

---

## Final answer

\[
\boxed{2048\text{ threads}}
\]

---

## Detailed explanation

The number of required blocks is

\[
\left\lceil\frac{2000}{512}\right\rceil
=
4.
\]

Each block contains 512 threads, so the total number of launched threads is

\[
4\times512=2048.
\]

Therefore, the grid contains

\[
\boxed{2048\text{ threads}}.
\]

There are

\[
2048-2000=48
\]

extra threads. These threads must be prevented from accessing data outside the vector, typically with a boundary check:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < 2000) {
    C[i] = A[i] + B[i];
}
```

---

# Problem 3

## Complete question

For the vector-addition configuration in Problem 2, how many warps are expected to have divergence due to the boundary check on the vector length?

---

## Final answer

\[
\boxed{1\text{ divergent warp}}
\]

---

## Detailed explanation

The grid contains 2048 threads. Therefore, the number of warps is

\[
\frac{2048}{32}=64.
\]

The valid thread indices are

\[
0\text{--}1999.
\]

Warp 62 contains global thread indices

\[
1984\text{--}2015.
\]

Within this warp:

- threads 1984–1999 are valid;
- threads 2000–2015 are outside the vector.

Therefore, warp 62 diverges.

Warp 63 contains global thread indices

\[
2016\text{--}2047.
\]

All threads in warp 63 fail the boundary condition. Since all threads follow the same path, this warp does not diverge; it is simply fully inactive for the vector-addition statement.

Therefore,

\[
\boxed{1\text{ divergent warp}}.
\]

---

# Problem 4

## Complete question

Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amounts of time, in microseconds, to execute the section:

\[
2.0,\ 2.3,\ 3.0,\ 2.8,\ 2.4,\ 1.9,\ 2.6,\ 2.9.
\]

The threads spend the rest of their time waiting for the barrier.

What percentage of the threads' total execution time is spent waiting for the barrier?

---

## Final answer

\[
\boxed{17.08\%}
\]

---

## Detailed explanation

The barrier can release only when the slowest thread arrives. The maximum execution time is

\[
3.0\ \mu\text{s}.
\]

The waiting time of each thread is its difference from 3.0 microseconds:

\[
\begin{aligned}
T_{\mathrm{wait}}
={}&(3.0-2.0)+(3.0-2.3)+(3.0-3.0)\\
&+(3.0-2.8)+(3.0-2.4)+(3.0-1.9)\\
&+(3.0-2.6)+(3.0-2.9).
\end{aligned}
\]

Thus,

\[
T_{\mathrm{wait}}
=
1.0+0.7+0+0.2+0.6+1.1+0.4+0.1
=
4.1\ \mu\text{s}.
\]

Each of the 8 threads has a total elapsed time of 3.0 microseconds before leaving the barrier. Therefore, the total thread execution time is

\[
8\times3.0=24.0\ \mu\text{s}.
\]

The waiting percentage is

\[
\frac{4.1}{24.0}\times100\%
=
17.08\%.
\]

Therefore,

\[
\boxed{17.08\%}.
\]

---

# Problem 5

## Complete question

A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the `__syncthreads()` instruction wherever barrier synchronization is needed.

Do you think this is a good idea? Explain.

---

## Final answer

\[
\boxed{\text{No, this is not generally a safe idea.}}
\]

---

## Detailed explanation

A block with 32 threads normally contains one warp. However, this does not mean that synchronization can always be removed.

First, CUDA program correctness should not depend on the assumption that every thread in a warp always advances in strict lockstep. On Volta and later architectures, independent thread scheduling makes this assumption especially unsafe.

Second, when threads exchange data through shared memory, one thread may read data written by another thread. Synchronization is needed to ensure that the required writes have completed and become visible before the reads occur.

For example:

```cpp
__shared__ float s[32];

s[threadIdx.x] = input[threadIdx.x];

__syncwarp();

output[threadIdx.x] =
    s[(threadIdx.x + 1) % 32];
```

Without a synchronization operation, a thread may read a shared-memory location before the responsible thread has completed the write.

For communication that is strictly limited to one warp, `__syncwarp()` may be sufficient. For block-wide synchronization, `__syncthreads()` should be used.

Therefore, a block size of 32 does not justify automatically deleting synchronization instructions.

---

# Problem 6

## Complete question

If a CUDA device's SM can accommodate up to 1536 threads and up to 4 thread blocks, which of the following block configurations results in the largest number of threads residing in the SM?

1. 128 threads per block
2. 256 threads per block
3. 512 threads per block
4. 1024 threads per block

---

## Final answer

\[
\boxed{\text{512 threads per block}}
\]

---

## Detailed explanation

The number of resident blocks is limited by both:

- the maximum of 4 blocks per SM;
- the maximum of 1536 threads per SM.

For each block size:

| Threads per block | Maximum resident blocks | Resident threads |
|---:|---:|---:|
| 128 | 4 | \(4\times128=512\) |
| 256 | 4 | \(4\times256=1024\) |
| 512 | 3 | \(3\times512=1536\) |
| 1024 | 1 | \(1\times1024=1024\) |

For 512 threads per block, three blocks fit exactly:

\[
3\times512=1536.
\]

This uses all available thread slots.

Therefore,

\[
\boxed{\text{512 threads per block}}.
\]

---

# Problem 7

## Complete question

Assume a device allows:

- up to 64 blocks per SM;
- up to 2048 threads per SM.

Indicate which of the following assignments per SM are possible. For each possible assignment, indicate the occupancy level.

1. 8 blocks with 128 threads each
2. 16 blocks with 64 threads each
3. 32 blocks with 32 threads each
4. 64 blocks with 32 threads each
5. 32 blocks with 64 threads each

---

## Final answers

| Assignment | Possible? | Occupancy |
|---|---|---:|
| 8 blocks × 128 threads | Yes | 50% |
| 16 blocks × 64 threads | Yes | 50% |
| 32 blocks × 32 threads | Yes | 50% |
| 64 blocks × 32 threads | Yes | 100% |
| 32 blocks × 64 threads | Yes | 100% |

---

## Detailed explanation

The occupancy is

\[
\text{occupancy}
=
\frac{\text{resident threads}}{2048}
\times100\%.
\]

### Assignment 1: 8 blocks with 128 threads each

\[
8\times128=1024.
\]

The block count and thread count are both within the device limits.

\[
\text{occupancy}
=
\frac{1024}{2048}\times100\%
=
50\%.
\]

Therefore,

\[
\boxed{\text{possible, }50\%}.
\]

### Assignment 2: 16 blocks with 64 threads each

\[
16\times64=1024.
\]

Therefore,

\[
\boxed{\text{possible, }50\%}.
\]

### Assignment 3: 32 blocks with 32 threads each

\[
32\times32=1024.
\]

Therefore,

\[
\boxed{\text{possible, }50\%}.
\]

### Assignment 4: 64 blocks with 32 threads each

\[
64\times32=2048.
\]

This exactly reaches both the block limit and the thread limit.

Therefore,

\[
\boxed{\text{possible, }100\%}.
\]

### Assignment 5: 32 blocks with 64 threads each

\[
32\times64=2048.
\]

Therefore,

\[
\boxed{\text{possible, }100\%}.
\]

---

# Problem 8

## Complete question

Consider a GPU with the following hardware limits:

- 2048 threads per SM;
- 32 blocks per SM;
- 64K, or 65,536, registers per SM.

For each of the following kernel configurations, determine whether the kernel can achieve full occupancy. If it cannot, identify the limiting factor.

1. The kernel uses 128 threads per block and 30 registers per thread.
2. The kernel uses 32 threads per block and 29 registers per thread.
3. The kernel uses 256 threads per block and 34 registers per thread.

---

## Final answers

| Configuration | Full occupancy? | Limiting factor |
|---|---|---|
| 128 threads/block, 30 registers/thread | Yes | None |
| 32 threads/block, 29 registers/thread | No | Maximum blocks per SM |
| 256 threads/block, 34 registers/thread | No | Register capacity |

---

## Detailed explanation

Full occupancy requires 2048 resident threads while satisfying:

- maximum 32 resident blocks;
- maximum 65,536 registers.

### Configuration 1: 128 threads/block and 30 registers/thread

The number of blocks required for 2048 resident threads is

\[
\frac{2048}{128}=16.
\]

Since

\[
16\le32,
\]

the block limit is satisfied.

The register requirement is

\[
2048\times30=61,440.
\]

Since

\[
61,440\le65,536,
\]

the register limit is also satisfied.

Therefore,

\[
\boxed{\text{full occupancy is possible}}.
\]

### Configuration 2: 32 threads/block and 29 registers/thread

The number of blocks required for 2048 threads is

\[
\frac{2048}{32}=64.
\]

However, the hardware supports only 32 blocks per SM. Therefore, at most

\[
32\times32=1024
\]

threads can be resident.

The occupancy is

\[
\frac{1024}{2048}\times100\%=50\%.
\]

The limiting factor is the number of block slots.

Therefore,

\[
\boxed{\text{full occupancy is not possible; the block limit is the limiting factor}}.
\]

### Configuration 3: 256 threads/block and 34 registers/thread

The number of blocks required for 2048 threads is

\[
\frac{2048}{256}=8.
\]

The block limit is satisfied, but the register requirement is

\[
2048\times34=69,632.
\]

Since

\[
69,632>65,536,
\]

full occupancy is impossible.

Each block requires

\[
256\times34=8,704
\]

registers.

The maximum number of complete resident blocks is

\[
\left\lfloor
\frac{65,536}{8,704}
\right\rfloor
=
7.
\]

Thus, the maximum resident thread count is

\[
7\times256=1792.
\]

The maximum occupancy is

\[
\frac{1792}{2048}\times100\%
=
87.5\%.
\]

Therefore,

\[
\boxed{\text{full occupancy is not possible; register capacity is the limiting factor}}.
\]

---

# Problem 9

## Complete question

A student says that they successfully multiplied two \(1024\times1024\) matrices using a matrix-multiplication kernel with \(32\times32\) thread blocks.

The CUDA device permits:

- at most 512 threads per block;
- at most 8 blocks per SM.

The student also states that each thread in a thread block calculates one element of the result matrix.

What should your reaction be, and why?

---

## Final answer

The student's claim is inconsistent with the stated device limit because a \(32\times32\) block contains 1024 threads, which exceeds the device limit of 512 threads per block.

\[
\boxed{32\times32=1024>512}
\]

The launch configuration is therefore invalid on the stated device.

---

## Detailed explanation

A two-dimensional thread block with dimensions

```cpp
dim3 block(32, 32);
```

contains

\[
32\times32=1024
\]

threads.

However, the device permits no more than 512 threads in one block.

Therefore,

\[
1024>512,
\]

and the kernel launch configuration is illegal.

The kernel would normally fail with an invalid launch configuration error.

The statement that each thread computes one result element is not itself a problem. The problem is the selected block size.

A valid alternative is

```cpp
dim3 block(16, 16);
```

which contains

\[
16\times16=256
\]

threads per block.

For a \(1024\times1024\) output matrix, the corresponding grid is

```cpp
dim3 grid(
    (1024 + 16 - 1) / 16,
    (1024 + 16 - 1) / 16
);
```

which produces a \(64\times64\) grid.

Another valid option is

```cpp
dim3 block(32, 16);
```

because

\[
32\times16=512
\]

threads per block, exactly matching the device limit.

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
| 9 | Invalid configuration because \(32\times32=1024>512\) |
