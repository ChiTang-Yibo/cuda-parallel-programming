# CUDA Chapter 6 Exercises 1–4 — Summary

## Exercise 1 — Column-Major Matrix Multiplication and Corner Turning

The first experiment studies how the layout of matrix $B$ affects CPU and GPU performance.

For the GPU, four implementations were compared:

1. row-major $B$ with a standard tiled kernel;
2. column-major $B$ without corner turning;
3. column-major $B$ with corner turning;
4. column-major $B$ with corner turning and shared-memory padding.

### Experimental results

| $N$ | Row-major B (ms) | Column-major, no corner turn (ms) | Corner turning (ms) | Corner + padding (ms) |
|---:|---:|---:|---:|---:|
| 128  | 0.0105 | 0.0103 | 0.0103 | 0.0103 |
| 256  | 0.0251 | 0.0295 | 0.0268 | 0.0249 |
| 512  | 0.1524 | 0.1850 | 0.1650 | 0.1534 |
| 768  | 0.4384 | 0.5213 | 0.4762 | 0.4388 |
| 1024 | 1.0391 | 1.2691 | 1.1289 | 1.0499 |

For medium and large matrices, using column-major $B$ with the original thread mapping is about 18–22% slower because neighboring threads access addresses separated by $K$ floats.

Corner turning changes the mapping so that neighboring threads load neighboring elements from global memory. This gives about a 10–12% speedup over the uncoalesced column-major version.

Adding one column of shared-memory padding,

```math
[T][T]
\rightarrow
[T][T+1],
```
further reduces shared-memory bank conflicts. The padded corner-turning kernel reaches performance very close to the standard row-major tiled kernel.

The CPU experiment shows the opposite layout preference for the current $i$-$j$-$k$ loop order: column-major $B$ makes the inner $k$ loop contiguous and is therefore faster than row-major $B$. At $N=1024$, the measured CPU speedup is about $1.52\times$.

The main conclusion is:

```math
\boxed{
\text{performance depends on data layout, thread mapping, and memory-access pattern together}
}
```
Column-major storage is not inherently slow on the GPU. It becomes inefficient only when the thread mapping does not follow its contiguous memory direction.

---

## Exercise 2 — BLOCK_SIZE and Fully Coalesced Tiled MatMul

For a square CUDA block,

```math
\texttt{blockDim}=(B,B),
```
the linear thread ID is

```math
tid=t_yB+t_x.
```
To make a full warp correspond to one complete row of threads, we want

```math
t_x=0,1,\ldots,31
```
while $t_y$ remains fixed. Therefore the square block width should be one warp wide:

```math
\boxed{\texttt{BLOCK\_SIZE}=32}
```
Since a square $32\times32$ block already contains 1024 threads, it is also at the usual CUDA maximum threads-per-block limit.

The key rule is that **coalescing is judged across threads in the same warp executing the same global-memory instruction**, not by looking at one thread in isolation.

---

## Exercise 3 — Coalesced / Uncoalesced / Not Applicable

Given

```cpp
__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float a_s[256];
    __shared__ float bc_s[4 * 256];

    a_s[threadIdx.x] = a[i];

    for (unsigned int j = 0; j < 4; ++j) {
        bc_s[j * 256 + threadIdx.x] =
            b[j * blockDim.x * gridDim.x + i]
            + c[i * 4 + j];
    }

    __syncthreads();

    d[i + 8] = a_s[threadIdx.x];
    e[i * 8] = bc_s[threadIdx.x * 4];
}
```

| Item | Access | Classification | Reason |
|---|---|---|---|
| a | `a[i]` | **Coalesced** | Neighboring threads access neighboring floats. |
| b | `a_s[threadIdx.x]` | **Coalescing not applicable** | `a_s` is shared memory; shared-memory bank conflicts are the relevant issue. |
| c | `b[j * blockDim.x * gridDim.x + i]` | **Coalesced** | For fixed `j`, neighboring threads differ only by `i`, so the index stride is 1 float. |
| d | `c[i * 4 + j]` | **Uncoalesced / strided** | For fixed `j`, neighboring threads are separated by 4 floats = 16 B. |
| e | `bc_s[j * 256 + threadIdx.x]` | **Coalescing not applicable** | Shared-memory access. |
| f | `a_s[threadIdx.x]` | **Coalescing not applicable** | Shared-memory access. |
| g | `d[i + 8]` | **Coalesced** | The constant offset does not change the 1-float stride between neighboring threads. |
| h | `bc_s[threadIdx.x * 4]` | **Coalescing not applicable** | Shared-memory access; possible bank conflicts are a separate issue. |
| i | `e[i * 8]` | **Uncoalesced / strided** | Neighboring threads are separated by 8 floats = 32 B. |

### Diagnostic rule

For a global-memory instruction, ask:

```math
\boxed{
\text{When }\texttt{threadIdx.x}\text{ increases by 1, how much does the memory index increase?}
}
```
For shared memory, global-memory coalescing does not apply.

---

## Exercise 4 — Floating-Point to Global-Memory Access Ratio

The ratio is measured in operations per byte:

```math
\boxed{
\text{OP/B}
=
\frac{\text{floating-point operations}}
{\text{global-memory bytes transferred}}
}
```
The calculation below follows the chapter convention of focusing on input global-memory traffic and ignoring the final output store.

### (a) Naive Matrix Multiplication

For one output element,

```math
P_{ij}
=
\sum_{k=0}^{W-1} M_{ik}N_{kj}.
```
Per output:

- $W$ loads from $M$,
- $W$ loads from $N$,
- approximately $2W$ floating-point operations.

Thus

```math
\text{bytes}=2W\times4=8W,
```
and

```math
\boxed{
\frac{2W}{8W}
=
0.25\ \text{OP/B}
}
```
---

### (b) $32\times32$ Shared-Memory Tiled MatMul

For one phase:

```math
2\times32^2
```
input floats are loaded, while the output tile performs

```math
2\times32^3
```
floating-point operations.

Therefore,

```math
\boxed{
AI
=
\frac{2\times32^3}
{2\times32^2\times4}
=
8\ \text{OP/B}
}
```
---

### (c) $32\times32$ Tiling + Thread Coarsening, $C=4$

For $C$ adjacent output tiles in one phase:

- one $M$ tile is loaded once: $T^2$ floats,
- $C$ different $N$ tiles are loaded: $CT^2$ floats.

Total input traffic:

```math
(C+1)T^2
```
floats.

The computation is

```math
2CT^3
```
operations.

Hence

```math
AI
=
\frac{2CT^3}
{4(C+1)T^2}
=
\frac{CT}{2(C+1)}.
```
For

```math
T=32,\qquad C=4,
```
```math
\boxed{
AI
=
\frac{4\times32}{2\times5}
=
12.8\ \text{OP/B}
}
```
---

## Final Answers

- **Exercise 1:** Corner turning restores coalesced global-memory loads for column-major $B$; adding shared-memory padding removes most of the remaining bank-conflict overhead.
- **Exercise 2:** $\texttt{BLOCK\_SIZE}=32$
- **Exercise 4(a):** $0.25\ \text{OP/B}$
- **Exercise 4(b):** $8\ \text{OP/B}$
- **Exercise 4(c):** $12.8\ \text{OP/B}$
