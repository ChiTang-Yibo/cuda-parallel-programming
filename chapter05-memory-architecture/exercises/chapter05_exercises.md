# CUDA Chapter 5 Exercises — Solutions and Review Notes

This document summarizes the Chapter 5 exercises on CUDA memory, tiling, synchronization, arithmetic intensity, and occupancy.

---

## Exercise 1

### Question

Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption?  
Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

### Answer

No, shared memory generally does not reduce global-memory traffic for a simple matrix-addition kernel.

For

```math
C_{ij}=A_{ij}+B_{ij},
```

the thread that computes $C_{ij}$ reads $A_{ij}$ and $B_{ij}$. Other threads do not normally reuse these same input values.

Therefore, there is little or no inter-thread data reuse.

### Analysis

Shared memory is useful when a value loaded from global memory can be reused by multiple threads in the same block.

In matrix addition:

- each input element is normally loaded once;
- each input element is used by one output computation;
- placing the value in shared memory would add an extra shared-memory store and load without reducing the required global-memory loads.

The important principle is:

```math
\boxed{\text{Shared memory is useful when there is data reuse across threads.}}
```

---

## Exercise 2

### Question

Draw the equivalent of Fig. 5.7 for an $8\times8$ matrix multiplication with $2\times2$ tiling and $4\times4$ tiling. Verify that the reduction in global memory bandwidth is proportional to the tile dimension.

### Answer

For an $8\times8$ matrix multiplication:

- $2\times2$ tiling produces $4\times4=16$ output tiles.
- $4\times4$ tiling produces $2\times2=4$ output tiles.

However, the important reason for the bandwidth reduction is not simply that there are fewer tiles. The key is that each value loaded into shared memory is reused by multiple threads.

### No Tiling

There are

```math
8\times8=64
```

output elements.

Each output element requires

```math
8 \text{ elements from } M
+
8 \text{ elements from } N.
```

Thus the software-level input load demand is

```math
64\times16=1024
```

element loads.

### $2\times2$ Tiling

Each block has

```math
2\times2=4
```

threads.

Each phase loads

```math
4 \text{ elements from } M
+
4 \text{ elements from } N
=
8
```

elements.

The shared dimension has length 8, so there are

```math
8/2=4
```

phases.

Loads per block:

```math
8\times4=32.
```

There are 16 blocks, so

```math
32\times16=512
```

global element loads.

### $4\times4$ Tiling

Each phase loads

```math
16+16=32
```

elements.

There are

```math
8/4=2
```

phases.

Loads per block:

```math
32\times2=64.
```

There are 4 blocks, so

```math
64\times4=256
```

loads.

### Comparison

| Method | Input element loads |
|---|---:|
| No tiling | 1024 |
| $2\times2$ tiling | 512 |
| $4\times4$ tiling | 256 |

Therefore the idealized reduction factor is proportional to the tile width $T$.

---

## Exercise 3

### Question

What type of incorrect execution behavior can happen if one forgets to use one or both `__syncthreads()` calls in the tiled matrix-multiplication kernel?

### Answer

If the first `__syncthreads()` is omitted, some threads may begin computation before all threads have finished loading the current tiles into shared memory.

This can cause a thread to read shared-memory values that have not yet been written.

If the second `__syncthreads()` is omitted, some threads may begin loading the next phase and overwrite shared-memory data while other threads are still using the current phase data.

### Analysis

The two barriers have different roles:

```math
\text{load}
\rightarrow
\boxed{\text{synchronize}}
\rightarrow
\text{compute}
```

The first barrier guarantees that the current shared-memory tiles are complete before computation starts.

Then:

```math
\text{compute}
\rightarrow
\boxed{\text{synchronize}}
\rightarrow
\text{next phase load}
```

The second barrier guarantees that all threads have finished reading the current tiles before those tiles are overwritten.

The general rule is:

```math
\boxed{\text{Synchronization is required by inter-thread data dependency.}}
```

---

## Exercise 4

### Question

Assuming capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory.

### Answer

Registers are private to individual threads, while shared memory is visible to all threads in the same block.

Therefore, a value stored in one thread's register cannot be directly reused by other threads, while a value stored in shared memory can be reused across the block.

### Analysis

For data reuse across threads:

```math
\boxed{\text{registers: thread-private}}
```

```math
\boxed{\text{shared memory: block-shared}}
```

This is why tiled matrix multiplication stores input tiles in shared memory rather than in individual thread registers.

---

## Exercise 5

### Question

For the tiled matrix-matrix multiplication kernel, if a $32\times32$ tile is used, what is the reduction of memory bandwidth usage for input matrices $M$ and $N$?

### Answer

A $32\times32$ block contains

```math
32\times32=1024
```

threads.

For one shared-dimension segment of length 32, the naive implementation requires each thread to load

```math
32 \text{ elements from } M
+
32 \text{ elements from } N.
```

Therefore,

```math
N_{\text{load, naive}}
=
1024\times64
=
65536.
```

With tiling, the block cooperatively loads only two $32\times32$ tiles:

```math
N_{\text{load, tiled}}
=
32\times32+32\times32
=
2048.
```

Thus,

```math
\text{reduction factor}
=
\frac{65536}{2048}
=
\boxed{32}.
```

The idealized global-memory load demand is reduced by approximately $32\times$.

---

## Exercise 6

### Question

Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created throughout the lifetime of the kernel?

### Answer

A local kernel variable is logically private to each thread.

Total number of threads:

```math
1000\times512=512000.
```

Therefore,

```math
\boxed{512000\text{ versions}}
```

are created logically.

### Analysis

A normal automatic variable declared inside a kernel is thread-private. It is normally placed in registers if possible, but may spill to CUDA local memory if necessary.

Thus,

```math
\boxed{\text{thread-private variable} \Rightarrow \text{one logical copy per thread}}
```

---

## Exercise 7

### Question

In the previous question, if a variable is declared as a shared-memory variable, how many versions of the variable will be created throughout the lifetime of the kernel?

### Answer

Shared memory is allocated once per block.

There are 1000 blocks, so

```math
\boxed{1000\text{ versions}}
```

of the shared variable are created.

### Key distinction

| Variable type | Logical copies |
|---|---|
| thread-private/local | one per thread |
| shared | one per block |

---

## Exercise 8

### Question

Consider multiplying two $N\times N$ matrices. How many times is each element in the input matrices requested from global memory when:

1. there is no tiling?
2. tiles of size $T\times T$ are used?

### (a) No Tiling

Consider one element of matrix $A$, for example $A_{00}$.

It contributes to every output element in the corresponding output row:

```math
C_{00},C_{01},\ldots,C_{0,N-1}.
```

Therefore each element of $A$ is requested approximately

```math
\boxed{N}
```

times.

The same is true for each element of $B$.

The total input-element load demand is therefore

```math
2N^3.
```

### (b) $T\times T$ Tiling

A value loaded into shared memory can be reused by $T$ threads within the block.

Therefore each input element needs to be requested from global memory approximately

```math
\boxed{\frac{N}{T}}
```

times when $N$ is divisible by $T$.

The total input-element load demand becomes

```math
\boxed{\frac{2N^3}{T}}.
```

Thus the idealized reduction factor is

```math
\boxed{T}.
```

### Non-divisible dimensions

If $N$ is not divisible by $T$, the number of tile positions along one dimension is

```math
\left\lceil\frac{N}{T}\right\rceil.
```

With boundary checks, invalid tile positions are not actually loaded from global memory. A useful valid-load count is

```math
\boxed{
2N^2
\left\lceil\frac{N}{T}\right\rceil
}.
```

---

## Exercise 9

### Question

A kernel performs 36 floating-point operations and seven 32-bit global-memory accesses per thread. Determine whether the kernel is compute-bound or memory-bound for:

**(a)** Peak FLOPS = 200 GFLOP/s, peak memory bandwidth = 100 GB/s

**(b)** Peak FLOPS = 300 GFLOP/s, peak memory bandwidth = 250 GB/s

### Step 1: Compute Arithmetic Intensity

A 32-bit access transfers

```math
32\text{ bits}=4\text{ bytes}.
```

Seven accesses correspond to

```math
7\times4=28\text{ B}.
```

Therefore,

```math
AI
=
\frac{36}{28}
\approx
1.286\text{ FLOP/B}.
```

### (a)

Machine balance:

```math
\frac{200\text{ GFLOP/s}}
{100\text{ GB/s}}
=
2\text{ FLOP/B}.
```

Since

```math
1.286<2,
```

the kernel is

```math
\boxed{\text{memory-bound}}.
```

Equivalently, the memory roof is

```math
1.286\times100
=
128.6\text{ GFLOP/s},
```

which is below the 200 GFLOP/s compute peak.

### (b)

Machine balance:

```math
\frac{300}{250}
=
1.2\text{ FLOP/B}.
```

Since

```math
1.286>1.2,
```

the kernel is

```math
\boxed{\text{compute-bound}}.
```

### General procedure

For this type of problem:

1. Convert global-memory accesses to bytes.
2. Compute

```math
AI=\frac{\text{FLOPs}}{\text{Bytes}}.
```

3. Compute the machine balance

```math
\frac{\text{Peak FLOP/s}}{\text{Peak bandwidth}}.
```

4. Compare the two values.

Equivalently,

```math
P_{\max}
\approx
\min
\left(
P_{\text{compute peak}},
AI\times BW
\right).
```

This is the basic idea of the Roofline model.

---

## Exercise 10

### Question

A CUDA kernel transposes each tile of a matrix using shared memory. `BLOCK_WIDTH` may range from 1 to 20.

The relevant code is conceptually:

```cpp
__shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

blockA[threadIdx.y][threadIdx.x]
    = A_elements[baseIdx];

A_elements[baseIdx]
    = blockA[threadIdx.x][threadIdx.y];
```

### (a) For which values of `BLOCK_WIDTH` does the kernel execute correctly?

The block contains

```math
BLOCK\_WIDTH^2
```

threads.

Under the textbook's warp-synchronous reasoning, the code can appear to work without a block-wide barrier if the entire block fits in one warp:

```math
BLOCK\_WIDTH^2\le32.
```

Therefore,

```math
\boxed{BLOCK\_WIDTH=1,2,3,4,5}.
```

Because

```math
5^2=25\le32,
```

while

```math
6^2=36>32.
```

### (b) What is the root cause and how should it be fixed?

The problem is a missing synchronization barrier between the shared-memory write and the transposed shared-memory read.

Thread $(x,y)$ writes

```math
blockA[y][x]
```

but reads

```math
blockA[x][y],
```

which is usually written by another thread.

With multiple warps, one warp may begin reading before another warp has completed its writes.

The fix is:

```cpp
blockA[threadIdx.y][threadIdx.x]
    = A_elements[baseIdx];

__syncthreads();

A_elements[baseIdx]
    = blockA[threadIdx.x][threadIdx.y];
```

No additional `__syncthreads()` is needed after the final global-memory store because there is no later inter-thread dependency on the shared-memory data.

### Modern CUDA note

The textbook result relies on traditional warp-synchronous execution reasoning. In production CUDA code, cross-thread shared-memory dependencies should be synchronized explicitly rather than relying on accidental lockstep behavior.

---

## Exercise 11

### Question

Consider the following kernel configuration:

```cpp
unsigned int N = 1024;
foo_kernel<<<(N + 128 - 1)/128, 128>>>(a_d, b_d);
```

The kernel contains:

```cpp
unsigned int i = ...;
float x[4];
__shared__ float y_s;
__shared__ float b_s[128];
```

### Preliminary calculation

The number of blocks is

```math
\frac{1024+127}{128}=8.
```

Each block has

```math
128
```

threads.

Total threads:

```math
8\times128=1024.
```

### (a) How many versions of `i` are there?

`i` is thread-private.

Therefore,

```math
\boxed{1024}
```

versions.

### (b) How many versions of `x[]` are there?

`x[4]` is also thread-private.

Each thread has its own four-element array.

Therefore,

```math
\boxed{1024\text{ versions of }x[]}.
```

This corresponds to

```math
1024\times4=4096
```

float elements in total.

### (c) How many versions of `y_s` are there?

`y_s` is shared, so there is one copy per block.

There are 8 blocks:

```math
\boxed{8}
```

versions.

### (d) How many versions of `b_s[]` are there?

`b_s[128]` is also shared.

There is one whole array per block:

```math
\boxed{8\text{ versions of }b_s[]}.
```

### (e) How much shared memory is used per block?

`y_s`:

```math
1\times4=4\text{ B}.
```

`b_s[128]`:

```math
128\times4=512\text{ B}.
```

Total:

```math
\boxed{516\text{ B/block}}.
```

### (f) What is the floating-point to global-memory access ratio?

The output expression contains:

- 5 floating-point multiplications;
- 5 floating-point additions.

Therefore,

```math
10\text{ FLOPs/thread}.
```

Global-memory accesses:

- 4 reads from `a[]`;
- 1 read from `b[i]`;
- 1 write to `b[i]`.

Total:

```math
6\text{ global-memory accesses}.
```

Each access is 4 bytes:

```math
6\times4=24\text{ B}.
```

Thus,

```math
\boxed{
\frac{10}{24}
\approx
0.417\text{ OP/B}
}
```

### Why is `(threadIdx.x + 3) % 128` not counted as a FLOP?

Because it is integer arithmetic.

`threadIdx.x` is an integer, so:

- `+ 3` is an integer addition;
- `% 128` is an integer modulo operation.

They are instructions, but they are not floating-point operations.

Also, accesses to `b_s[]` are shared-memory accesses, not global-memory accesses.

---

## Exercise 12

### Question

Consider a GPU with the following hardware limits:

- 2048 threads/SM
- 32 blocks/SM
- 65,536 registers/SM
- 96 KB shared memory/SM

Determine whether each kernel can achieve full occupancy.

### Important wording issue

The printed exercise states:

- 4 KB shared memory/SM in part (a);
- 8 KB shared memory/SM in part (b).

This wording is unusual for an occupancy calculation, because kernel resource usage is normally specified as **shared memory per block**.

Therefore, both interpretations are discussed below.

### (a) 64 threads/block, 27 registers/thread, 4 KB shared memory

#### Thread/block limit

To reach 2048 threads:

```math
2048/64=32\text{ blocks}.
```

This equals the hardware limit of 32 blocks/SM.

#### Register limit

Registers per block:

```math
64\times27=1728.
```

For 32 blocks:

```math
1728\times32=55296.
```

Since

```math
55296<65536,
```

registers do not prevent full occupancy.

#### Literal interpretation: 4 KB shared memory/SM

If the kernel truly uses only 4 KB total shared memory per SM, then shared memory is not limiting.

Under the wording as printed:

```math
\boxed{\text{full occupancy is achievable}}.
```

#### If the intended wording is 4 KB shared memory/block

The shared-memory limit allows:

```math
96/4=24\text{ blocks}.
```

Therefore resident threads are

```math
24\times64=1536.
```

Occupancy:

```math
\frac{1536}{2048}=0.75.
```

Thus:

```math
\boxed{75\%\text{ occupancy}}
```

with shared memory as the limiting factor.

### (b) 256 threads/block, 31 registers/thread, 8 KB shared memory

Full occupancy requires

```math
2048/256=8\text{ blocks}.
```

Registers per block:

```math
256\times31=7936.
```

For 8 blocks:

```math
7936\times8=63488<65536.
```

If shared memory is interpreted as 8 KB/block:

```math
8\times8\text{ KB}=64\text{ KB}<96\text{ KB}.
```

Therefore all given constraints permit full occupancy:

```math
\boxed{100\%\text{ occupancy}}.
```

The same conclusion holds under the literal 8 KB/SM interpretation.

---

# Chapter 5 Review Summary

The exercises reinforce the main logic of this chapter:

```math
\boxed{
\text{Global memory traffic is expensive}
}
```

```math
\downarrow
```

```math
\boxed{
\text{Identify data reuse}
}
```

```math
\downarrow
```

```math
\boxed{
\text{Use shared memory and tiling}
}
```

```math
\downarrow
```

```math
\boxed{
\text{Synchronize threads correctly}
}
```

```math
\downarrow
```

```math
\boxed{
\text{Reduce repeated global-memory loads}
}
```

```math
\downarrow
```

```math
\boxed{
\text{Increase arithmetic intensity}
}
```

```math
\downarrow
```

```math
\boxed{
\text{Balance shared-memory/register usage against occupancy}
}
```

The most useful performance quantities to remember are:

```math
\boxed{
\text{Bandwidth}
=
\frac{\text{Bytes transferred}}{\text{time}}
}
```

```math
\boxed{
\text{Compute throughput}
=
\frac{\text{FLOPs}}{\text{time}}
}
```

```math
\boxed{
AI
=
\frac{\text{FLOPs}}{\text{Bytes}}
}
```

and the simplified Roofline relation:

```math
\boxed{
P_{\max}
\approx
\min
\left(
P_{\text{peak}},
AI\times BW
\right)
}
```

These ideas form the foundation for later CUDA topics such as reduction, convolution, optimized GEMM, and attention kernels.
