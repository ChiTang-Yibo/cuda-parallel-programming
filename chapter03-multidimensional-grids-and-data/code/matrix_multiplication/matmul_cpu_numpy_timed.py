import time
from typing import Dict, List, Tuple

import numpy as np


def initialize_matrix(width: int, seed_offset: int) -> np.ndarray:
    """Create the same matrix pattern as the CUDA benchmark."""
    num_elements = width * width
    values = (np.arange(num_elements, dtype=np.int32) + seed_offset) % 100
    return (values.astype(np.float32) / 100.0).reshape(width, width)


def reference_element(M: np.ndarray, N: np.ndarray, row: int, col: int) -> np.float32:
    """Compute one output element with an explicit dot product for checking."""
    value = np.float32(0.0)
    width = M.shape[0]
    for k in range(width):
        value += M[row, k] * N[k, col]
    return value


def sample_check(M: np.ndarray, N: np.ndarray, P: np.ndarray, width: int) -> bool:
    samples: List[Tuple[int, int]] = [
        (0, 0),
        (0, width - 1),
        (width - 1, 0),
        (width - 1, width - 1),
        (width // 2, width // 2),
    ]

    tolerance = 1.0e-3

    for row, col in samples:
        cpu_ref = reference_element(M, N, row, col)
        numpy_val = P[row, col]
        diff = abs(float(cpu_ref) - float(numpy_val))

        if diff > tolerance:
            print(
                f"Check failed at P[{row}][{col}] "
                f"Reference = {cpu_ref:.6f}, NumPy = {numpy_val:.6f}, diff = {diff:.6f}"
            )
            return False

    return True


def run_matmul_case(width: int, repeat: int) -> Dict[str, float]:
    print()
    print("============================================================")
    print(f"Matrix size: {width} x {width}")
    print(f"Repeat: {repeat}")
    print("============================================================")

    total_start = time.perf_counter()

    init_start = time.perf_counter()
    M = initialize_matrix(width, 0)
    N = initialize_matrix(width, 17)
    init_end = time.perf_counter()

    host_init_ms = (init_end - init_start) * 1000.0

    # Warm-up. This also initializes the CPU BLAS backend if NumPy uses one.
    P = M @ N

    compute_start = time.perf_counter()
    for _ in range(repeat):
        P = M @ N
    compute_end = time.perf_counter()

    compute_total_ms = (compute_end - compute_start) * 1000.0
    compute_avg_ms = compute_total_ms / repeat

    check_passed = sample_check(M, N, P, width)

    total_end = time.perf_counter()
    total_case_ms = (total_end - total_start) * 1000.0

    operations = 2.0 * width * width * width
    compute_seconds = compute_avg_ms / 1000.0
    gflops = operations / compute_seconds / 1.0e9 if compute_seconds > 0 else 0.0

    num_elements = width * width
    num_bytes_per_matrix = num_elements * np.dtype(np.float32).itemsize

    print()
    print("Timing result")
    print("----------------------------------------")
    print(f"Host init time       : {host_init_ms:.4f} ms")
    print(f"CPU compute total    : {compute_total_ms:.4f} ms")
    print(f"CPU compute average  : {compute_avg_ms:.4f} ms")
    print(f"Total case time      : {total_case_ms:.4f} ms")
    print("----------------------------------------")
    print(f"Matrix elements      : {num_elements}")
    print(f"Bytes per matrix     : {num_bytes_per_matrix}")
    print(f"Sample check         : {'PASSED' if check_passed else 'FAILED'}")
    print(f"Estimated GFLOPS     : {gflops:.4f}")

    return {
        "width": float(width),
        "repeat": float(repeat),
        "host_init_ms": host_init_ms,
        "compute_total_ms": compute_total_ms,
        "compute_avg_ms": compute_avg_ms,
        "total_case_ms": total_case_ms,
        "gflops": gflops,
        "check_passed": 1.0 if check_passed else 0.0,
    }


def main() -> None:
    print("Python NumPy CPU matrix multiplication benchmark")
    print("This version uses NumPy matmul on CPU, usually backed by BLAS.")
    print("It is not a pure Python triple-loop implementation.")

    cases = [
        (10, 1000),
        (100, 200),
        (1000, 20),
    ]

    results = []
    for width, repeat in cases:
        results.append(run_matmul_case(width, repeat))

    print()
    print("Summary")
    print("----------------------------------------")
    print(f"{'Size':>10} {'Repeat':>8} {'Avg ms':>12} {'GFLOPS':>12} {'Check':>10}")
    for r in results:
        size_str = f"{int(r['width'])}x{int(r['width'])}"
        check_str = "PASSED" if r["check_passed"] > 0.5 else "FAILED"
        print(
            f"{size_str:>10} "
            f"{int(r['repeat']):>8} "
            f"{r['compute_avg_ms']:>12.4f} "
            f"{r['gflops']:>12.4f} "
            f"{check_str:>10}"
        )


if __name__ == "__main__":
    main()
