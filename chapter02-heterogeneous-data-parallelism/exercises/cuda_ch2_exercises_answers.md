# CUDA Chapter 2 Exercises: Answers and Explanations

## Answer Key

| Problem | Answer |
|---|---|
| 1 | C |
| 2 | C |
| 3 | D |
| 4 | C |
| 5 | D |
| 6 | D |
| 7 | C |
| 8 | C |
| 9a | 128 |
| 9b | 200064 |
| 9c | 1563 |
| 9d | 200064 |
| 9e | 200000 |
| 10 | Use `__host__ __device__` for functions that need both host and device versions. |

---

## 1. Mapping one thread to one vector element

**Question:**  
If each thread calculates one output element of a vector addition, what is the expression for mapping the thread/block indices to the data index `i`?

**Answer: C**

```cpp
i = blockIdx.x * blockDim.x + threadIdx.x;
```

**Explanation:**

For a one-dimensional CUDA grid, `blockIdx.x` gives the block index, and `threadIdx.x` gives the thread index inside that block. Since each previous block contains `blockDim.x` threads, the global data index is

```cpp
i = blockIdx.x * blockDim.x + threadIdx.x;
```

This maps each CUDA thread to one vector element.

---

## 2. Each thread calculates two adjacent elements

**Question:**  
If each thread calculates two adjacent elements of a vector addition, what is the expression for the first element processed by a thread?

**Answer: C**

```cpp
i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
```

**Explanation:**

First compute the ordinary global thread index:

```cpp
g = blockIdx.x * blockDim.x + threadIdx.x;
```

If each thread processes two adjacent elements, then thread `g` should process

```cpp
2 * g
```

and

```cpp
2 * g + 1
```

Therefore, the first element index is

```cpp
i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
```

Example:

```text
thread 0 -> elements 0 and 1
thread 1 -> elements 2 and 3
thread 2 -> elements 4 and 5
```

---

## 3. Each block processes two sections

**Question:**  
Each thread calculates two elements. Each block processes `2 * blockDim.x` consecutive elements that form two sections. All threads process the first section first, then move to the next section. What is the expression for the first element?

**Answer: D**

```cpp
i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
```

**Explanation:**

Each block processes

```cpp
2 * blockDim.x
```

elements. Therefore, the starting index of block `blockIdx.x` is

```cpp
blockIdx.x * blockDim.x * 2
```

Inside the first section, each thread handles one element, so we add `threadIdx.x`:

```cpp
i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
```

The second element processed by the same thread would usually be

```cpp
i + blockDim.x;
```

Example with `blockDim.x = 256`:

```text
thread 0 -> elements 0 and 256
thread 1 -> elements 1 and 257
thread 2 -> elements 2 and 258
```

This is different from Problem 2. In Problem 2, each thread processes two adjacent elements. In Problem 3, each thread processes one element in the first section and one element in the second section.

---

## 4. Number of threads for vector length 8000 and block size 1024

**Question:**  
For vector addition, vector length is 8000, each thread calculates one output element, and block size is 1024. The programmer configures the kernel call to use the minimum number of thread blocks to cover all elements. How many threads are in the grid?

**Answer: C**

```text
8192
```

**Explanation:**

The number of blocks is

```math
\left\lceil \frac{8000}{1024} \right\rceil
=
\left\lceil 7.8125 \right\rceil
=
8.
```

Each block has 1024 threads, so the total number of threads is

```math
8 \times 1024 = 8192.
```

The extra threads are needed because CUDA launches whole thread blocks. The kernel should use

```cpp
if (i < n)
```

to prevent the extra threads from accessing invalid elements.

---

## 5. Second argument of `cudaMalloc` for `v` integer elements

**Question:**  
If we want to allocate an array of `v` integer elements in CUDA device global memory, what should be the second argument of `cudaMalloc`?

**Answer: D**

```cpp
v * sizeof(int)
```

**Explanation:**

The second argument of `cudaMalloc` is the number of bytes to allocate, not the number of elements. An array of `v` integers requires

```cpp
v * sizeof(int)
```

bytes.

Example:

```cpp
int *A_d;
cudaMalloc((void**)&A_d, v * sizeof(int));
```

---

## 6. First argument of `cudaMalloc` for a floating-point pointer `A_d`

**Question:**  
If we want to allocate an array of `n` floating-point elements and have a floating-point pointer variable `A_d` point to the allocated memory, what should be the first argument of `cudaMalloc`?

**Answer: D**

```cpp
(void**)&A_d
```

**Explanation:**

`cudaMalloc` needs to modify the pointer variable `A_d` so that it stores the device memory address. Therefore, we pass the address of the pointer variable:

```cpp
&A_d
```

Since `cudaMalloc` expects a generic pointer-to-pointer type, we cast it to

```cpp
(void**)&A_d
```

Example:

```cpp
float *A_d;
cudaMalloc((void**)&A_d, n * sizeof(float));
```

The key idea is:

```text
A_d          -> device pointer variable
&A_d         -> address of the pointer variable
(void**)&A_d -> generic pointer-to-pointer passed to cudaMalloc
```

---

## 7. Copying 3000 bytes from host array `A_h` to device array `A_d`

**Question:**  
If we want to copy 3000 bytes of data from host array `A_h` to device array `A_d`, what is the appropriate CUDA API call?

**Answer: C**

```cpp
cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);
```

**Explanation:**

The syntax of `cudaMemcpy` is

```cpp
cudaMemcpy(destination, source, number_of_bytes, direction);
```

Here the destination is the device array `A_d`, and the source is the host array `A_h`.

---

## 8. Declaring an error variable for CUDA API calls

**Question:**  
How should we declare a variable `err` that can receive the returned value of a CUDA API call?

**Answer: C**

```cpp
cudaError_t err;
```

**Explanation:**

Most CUDA Runtime API functions return a value of type `cudaError_t`.

Example:

```cpp
cudaError_t err = cudaMalloc((void**)&A_d, size);

if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
```

---

## 9. Kernel launch analysis

Given:

```cpp
__global__ void foo_kernel(float* a, float* b, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        b[i] = 2.7f * a[i] - 4.3f;
    }
}

void foo(float* a_d, float* b_d) {
    unsigned int N = 200000;
    foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
}
```

### 9a. Number of threads per block

**Answer:**

```text
128
```

The second execution configuration parameter is the number of threads per block:

```cpp
foo_kernel<<<..., 128>>>(...);
```

---

### 9b. Number of threads in the grid

**Answer:**

```text
200064
```

The number of blocks is

```math
\frac{N + 128 - 1}{128}
=
\frac{200000 + 127}{128}
=
1563.
```

The total number of threads is

```math
1563 \times 128 = 200064.
```

---

### 9c. Number of blocks in the grid

**Answer:**

```text
1563
```

Because

```cpp
(N + 128 - 1) / 128
```

performs ceiling division for `N = 200000`.

---

### 9d. Number of threads that execute line 02

Line 02 is:

```cpp
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
```

**Answer:**

```text
200064
```

All launched threads execute this line.

---

### 9e. Number of threads that execute line 04

Line 04 is:

```cpp
b[i] = 2.7f * a[i] - 4.3f;
```

**Answer:**

```text
200000
```

This line is inside:

```cpp
if (i < N)
```

Only threads with

```cpp
i < 200000
```

execute the assignment. The total grid has 200064 threads, so the last 64 threads do not execute line 04.

---

## 10. Using the same function on both host and device

**Question:**  
A new user complains that CUDA is tedious because they need to declare many functions twice: once as a host function and once as a device function. What is the response?

**Answer:**

CUDA allows a function to be declared with both `__host__` and `__device__`.

```cpp
__host__ __device__
float square(float x) {
    return x * x;
}
```

**Explanation:**

If a helper function needs to be used on both the CPU and the GPU, we do not need to manually write two separate versions. We can declare it as:

```cpp
__host__ __device__
```

Then the compiler generates two versions:

```text
one version for host execution
one version for device execution
```

This is different from `__global__`. The keyword `__global__` declares a kernel function. A kernel is called from the host and launches a new grid of GPU threads.

In contrast, `__host__ __device__` is used for ordinary helper functions that need both CPU and GPU versions.

A concise answer is:

> The complaint is not fully correct. If the same helper function needs to be used on both the host and the device, CUDA allows the function to be declared with both `__host__` and `__device__`. The compiler then generates two versions of the function, one for host execution and one for device execution. Therefore, the programmer does not need to manually write two separate functions.

