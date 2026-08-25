# Chapter 3 Exercises 1--5: Questions, Answers, Results, and Explanations

## Exercise 1

### Question

In this chapter, the basic matrix-multiplication kernel assigns one CUDA thread to one output matrix element. This exercise asks for two alternative designs:

1. Write a kernel in which each thread computes one complete output matrix row.
2. Write a kernel in which each thread computes one complete output matrix column.
3. Compare the advantages and disadvantages of the two designs.

Assume square matrices:

\[
P = MN,
\]

where each output element is:

\[
P_{row,col}
=
\sum_{k=0}^{Width-1}
M_{row,k}N_{k,col}.
\]

---

### Exercise 1(a): One thread computes one output row

Each thread first determines one output row:

```cpp
int row = blockIdx.x * blockDim.x + threadIdx.x;
```

If the row is valid, the thread computes every column in that row:

```cpp
__global__
void matMulRowKernel(const float* M,
                     const float* N,
                     float* P,
                     int width)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width) {
        for (int col = 0; col < width; ++col) {
            float value = 0.0f;

            for (int k = 0; k < width; ++k) {
                value += M[row * width + k]
                       * N[k * width + col];
            }

            P[row * width + col] = value;
        }
    }
}
```

### Execution configuration

Only `width` threads are required because there is one thread per output row.

```cpp
dim3 block(256, 1, 1);

dim3 grid(
    (width + block.x - 1) / block.x,
    1,
    1
);
```

For a \(256\times256\) matrix:

```text
Block = (256, 1, 1)
Grid  = (1, 1, 1)
```

For a \(512\times512\) matrix:

```text
Block = (256, 1, 1)
Grid  = (2, 1, 1)
```

---

### Exercise 1(b): One thread computes one output column

Each thread first determines one output column:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

If the column is valid, the thread computes every row in that column:

```cpp
__global__
void matMulColumnKernel(const float* M,
                        const float* N,
                        float* P,
                        int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width) {
        for (int row = 0; row < width; ++row) {
            float value = 0.0f;

            for (int k = 0; k < width; ++k) {
                value += M[row * width + k]
                       * N[k * width + col];
            }

            P[row * width + col] = value;
        }
    }
}
```

The execution configuration is also one-dimensional:

```cpp
dim3 block(256, 1, 1);

dim3 grid(
    (width + block.x - 1) / block.x,
    1,
    1
);
```

---

### Exercise 1(c): Comparison of the two designs

Both designs perform the same arithmetic work:

\[
O(Width^3).
\]

However, they differ in their global-memory access patterns.

#### One thread per row

At the same loop iteration, neighboring threads process different rows.

For matrix \(M\), neighboring threads access:

```cpp
M[row * width + k]
```

Their addresses are separated by `width` elements. This is a strided and poorly coalesced access pattern.

For matrix \(N\), threads may read the same element during the same loop iteration, but the output writes:

```cpp
P[row * width + col]
```

are also separated by an entire row.

**Advantages**

- The mapping is easy to understand.
- Each thread naturally produces one complete row.
- Elements of \(M\) within one thread are read consecutively as `k` increases.

**Disadvantages**

- Neighboring threads access different rows of \(M\), producing strided accesses.
- Neighboring threads write output elements separated by `width`.
- The number of active threads is only `width`, which is much smaller than the `width²` threads used by the one-element-per-thread design.
- Parallelism is limited for moderate matrix sizes.

#### One thread per column

At the same loop iteration, neighboring threads have consecutive `col` values.

For matrix \(N\), neighboring threads access:

```cpp
N[k * width + col]
```

These addresses are consecutive and can be coalesced.

For the output matrix, neighboring threads write:

```cpp
P[row * width + col]
```

which is also consecutive across threads.

The threads also read the same value:

```cpp
M[row * width + k]
```

at the same iteration, which can benefit from broadcast/cache behavior.

**Advantages**

- Consecutive threads read consecutive elements of \(N\).
- Consecutive threads write consecutive elements of \(P\).
- The global-memory access pattern is more suitable for CUDA hardware.

**Disadvantages**

- Only `width` threads are launched.
- Each thread still performs \(Width^2\) work.
- It is much less parallel than the standard one-thread-per-output-element design.
- It does not use shared-memory tiling or data reuse explicitly.

---

### Experimental results

#### Matrix size: \(256\times256\)

| Design | Average kernel time | Check |
|---|---:|---|
| One thread per row | 6.965608 ms | PASSED |
| One thread per column | 1.439275 ms | PASSED |

The column-based kernel is faster by:

\[
\frac{6.965608}{1.439275}
\approx
\boxed{4.84\times}.
\]

#### Matrix size: \(512\times512\)

| Design | Average kernel time | Check |
|---|---:|---|
| One thread per row | 24.813755 ms | PASSED |
| One thread per column | 5.236139 ms | PASSED |

The column-based kernel is faster by:

\[
\frac{24.813755}{5.236139}
\approx
\boxed{4.74\times}.
\]

### Interpretation

The column-based design is consistently about \(4.7\)--\(4.8\times\) faster in these tests. The main reason is not a difference in arithmetic complexity; both kernels perform the same number of multiply-add operations. The difference comes mainly from memory access:

- the column kernel gives neighboring threads consecutive accesses to \(N\);
- the column kernel gives neighboring threads consecutive writes to \(P\);
- the row kernel produces strided accesses and writes across neighboring threads.

Both kernels are still inferior to the chapter's original one-thread-per-output-element design because they launch only `width` threads rather than `width²` threads.

---

## Exercise 2

### Question

A matrix-vector multiplication takes:

- an input matrix \(B\),
- an input vector \(C\),

and produces an output vector \(A\).

Each output element is the dot product of one row of \(B\) and the vector \(C\):

\[
A_i
=
\sum_{j=0}^{Width-1}
B_{i,j}C_j.
\]

Write:

1. a CUDA matrix-vector multiplication kernel;
2. a host stub with four parameters:
   - pointer to the output vector;
   - pointer to the input matrix;
   - pointer to the input vector;
   - number of elements in each dimension.

Use one thread to compute one output-vector element.

---

### Kernel design

Each thread computes one row index:

```cpp
int row = blockIdx.x * blockDim.x + threadIdx.x;
```

It then computes one dot product:

```cpp
__global__
void matrixVectorMulKernel(float* A,
                           const float* B,
                           const float* C,
                           int width)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width) {
        float value = 0.0f;

        for (int k = 0; k < width; ++k) {
            value += B[row * width + k] * C[k];
        }

        A[row] = value;
    }
}
```

The matrix index:

```cpp
B[row * width + k]
```

selects the \(k\)-th element in row `row`.

The vector index:

```cpp
C[k]
```

selects the corresponding vector element.

---

### Host-stub design

The host stub performs the complete CUDA workflow:

1. allocate device memory for \(A\), \(B\), and \(C\);
2. copy \(B\) and \(C\) from host to device;
3. configure the grid and block;
4. launch the kernel;
5. copy \(A\) from device to host;
6. free device memory.

The execution configuration is:

```cpp
dim3 block(256, 1, 1);

dim3 grid(
    (width + block.x - 1) / block.x,
    1,
    1
);
```

Only `width` threads are required because the output vector contains `width` elements.

---

### Experimental results

#### Width \(=1024\)

```text
Matrix size : 1024 x 1024
Vector size : 1024
Block       : (256, 1, 1)
Grid        : (4, 1, 1)
Sample check: PASSED
```

The total number of launched threads is:

\[
4\times256=1024.
\]

This exactly matches the number of output-vector elements.

#### Width \(=2048\)

```text
Matrix size : 2048 x 2048
Vector size : 2048
Block       : (256, 1, 1)
Grid        : (8, 1, 1)
Sample check: PASSED
```

The total number of launched threads is:

\[
8\times256=2048.
\]

Again, this exactly matches the output-vector size.

### Interpretation

The `PASSED` results show that the CUDA output agrees with the sampled CPU reference calculations.

The mapping is:

\[
\text{one CUDA thread}
\longrightarrow
\text{one row of }B
\longrightarrow
\text{one element of }A.
\]

Each thread performs `width` multiply-add operations. Therefore, the total arithmetic complexity is:

\[
O(Width^2).
\]

Unlike matrix-matrix multiplication, matrix-vector multiplication has only `width` independent output elements. This limits the number of independent CUDA threads compared with the `width²` outputs of matrix-matrix multiplication.

---

## Exercise 3

### Question

Consider the following CUDA kernel and host function:

```cpp
__global__
void foo_kernel(float* a,
                float* b,
                unsigned int M,
                unsigned int N)
{
    unsigned int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        b[row * N + col] =
            a[row * N + col] / 2.1f + 4.8f;
    }
}

void foo(float* a_d, float* b_d)
{
    unsigned int M = 150;
    unsigned int N = 300;

    dim3 bd(16, 32);

    dim3 gd(
        (N - 1) / 16 + 1,
        (M - 1) / 32 + 1
    );

    foo_kernel<<<gd, bd>>>(a_d, b_d, M, N);
}
```

Answer:

1. What is the number of threads per block?
2. What is the number of threads in the grid?
3. What is the number of blocks in the grid?
4. What is the number of threads that execute the assignment inside the `if` statement?

---

### Answer 3(a): Threads per block

\[
16\times32
=
\boxed{512}
\]

---

### Answer 3(b): Threads in the grid

The grid dimensions are:

\[
\text{gridDim.x}
=
\left\lceil\frac{300}{16}\right\rceil
=
19,
\]

\[
\text{gridDim.y}
=
\left\lceil\frac{150}{32}\right\rceil
=
5.
\]

Therefore:

\[
19\times5\times512
=
\boxed{48{,}640}
\]

threads are launched.

---

### Answer 3(c): Blocks in the grid

\[
19\times5
=
\boxed{95}
\]

---

### Answer 3(d): Threads executing the assignment

Only threads satisfying:

```cpp
row < M && col < N
```

execute the assignment.

The valid region contains:

\[
150\times300
=
\boxed{45{,}000}
\]

elements and therefore 45,000 active threads.

The remaining number of threads is:

\[
48{,}640-45{,}000
=
3{,}640.
\]

These extra threads fail the boundary check.

---

## Exercise 4

### Question

A two-dimensional matrix has:

- width \(=400\),
- height \(=500\).

Find the one-dimensional array index of the element at:

\[
row=20,
\qquad
column=10.
\]

Calculate the index for:

1. row-major order;
2. column-major order.

All indices are zero-based.

---

### Answer 4(a): Row-major order

\[
\text{index}
=
row\times width+column
\]

\[
\text{index}
=
20\times400+10
=
\boxed{8{,}010}
\]

---

### Answer 4(b): Column-major order

\[
\text{index}
=
column\times height+row
\]

\[
\text{index}
=
10\times500+20
=
\boxed{5{,}020}
\]

---

## Exercise 5

### Question

A three-dimensional tensor has:

- width \(=400\),
- height \(=500\),
- depth \(=300\).

It is stored as a one-dimensional array in row-major order.

Find the array index of the element at:

\[
x=10,
\qquad
y=20,
\qquad
z=5.
\]

All indices are zero-based.

---

### Answer

Each \(z\)-plane contains:

\[
height\times width
\]

elements.

The row-major index is:

\[
\text{index}
=
z(height\times width)
+
y\times width
+
x.
\]

Substitution gives:

\[
\begin{aligned}
\text{index}
&=
5(500\times400)
+
20\times400
+
10\\
&=
1{,}000{,}000
+
8{,}000
+
10\\
&=
\boxed{1{,}008{,}010}.
\end{aligned}
\]

The depth value determines the valid range \(0\le z<300\), while the offset of one plane is `height × width`.

---

## Final Answer Summary

| Exercise | Result |
|---|---:|
| 1, row kernel, \(256\times256\) | 6.965608 ms |
| 1, column kernel, \(256\times256\) | 1.439275 ms |
| 1, column-kernel speedup | \(4.84\times\) |
| 1, row kernel, \(512\times512\) | 24.813755 ms |
| 1, column kernel, \(512\times512\) | 5.236139 ms |
| 1, column-kernel speedup | \(4.74\times\) |
| 2, width 1024 | Grid \(=(4,1,1)\), PASSED |
| 2, width 2048 | Grid \(=(8,1,1)\), PASSED |
| 3(a) Threads per block | 512 |
| 3(b) Threads in grid | 48,640 |
| 3(c) Blocks in grid | 95 |
| 3(d) Threads executing assignment | 45,000 |
| 4(a) Row-major index | 8,010 |
| 4(b) Column-major index | 5,020 |
| 5 3D row-major index | 1,008,010 |
