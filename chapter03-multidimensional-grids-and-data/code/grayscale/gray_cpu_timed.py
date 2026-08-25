import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_gray_cpu_numpy(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using CPU NumPy operations."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    gray = 0.21 * r + 0.71 * g + 0.07 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def main() -> None:
    input_path = Path(r"C:\Users\12504\Desktop\6-8\CUDA\chapter3\gray\sample.png")
    output_path = Path(r"C:\Users\12504\Desktop\6-8\CUDA\chapter3\gray\sample_gray_cpu_timed.png")
    repeat = 100

    # Optional command-line override:
    # python gray_cpu_timed.py input.png output.png 100
    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    if len(sys.argv) >= 4:
        repeat = int(sys.argv[3])
        if repeat <= 0:
            repeat = 1

    total_start = time.perf_counter()

    # 1. Load image.
    load_start = time.perf_counter()
    img = Image.open(input_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    load_end = time.perf_counter()

    height, width, channels = rgb.shape

    print(f"Input image: {input_path}")
    print(f"Width  = {width}")
    print(f"Height = {height}")
    print(f"Loaded channels = {channels}")
    print(f"Repeat = {repeat}")

    # 2. Warm-up.
    gray = rgb_to_gray_cpu_numpy(rgb)

    # 3. Timed CPU computation.
    compute_start = time.perf_counter()
    for _ in range(repeat):
        gray = rgb_to_gray_cpu_numpy(rgb)
    compute_end = time.perf_counter()

    # 4. Save image.
    save_start = time.perf_counter()
    Image.fromarray(gray, mode="L").save(output_path)
    save_end = time.perf_counter()

    total_end = time.perf_counter()

    load_ms = (load_end - load_start) * 1000.0
    compute_total_ms = (compute_end - compute_start) * 1000.0
    compute_avg_ms = compute_total_ms / repeat
    save_ms = (save_end - save_start) * 1000.0
    total_ms = (total_end - total_start) * 1000.0

    print(f"Saved grayscale image: {output_path}")
    print("\nTiming result")
    print("----------------------------------------")
    print(f"Image load time        : {load_ms:.4f} ms")
    print(f"CPU compute total time : {compute_total_ms:.4f} ms")
    print(f"CPU compute avg time   : {compute_avg_ms:.4f} ms")
    print(f"Image save time        : {save_ms:.4f} ms")
    print(f"Total program time     : {total_ms:.4f} ms")
    print("----------------------------------------")
    print(f"Pixels                 : {width * height}")


if __name__ == "__main__":
    main()
