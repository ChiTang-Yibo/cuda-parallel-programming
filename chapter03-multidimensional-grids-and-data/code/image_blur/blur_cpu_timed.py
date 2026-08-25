import time
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def blur_rgb_cpu(img: np.ndarray, radius: int) -> np.ndarray:
    """
    CPU NumPy implementation of RGB image blur.

    Boundary rule:
    Only valid neighboring pixels are averaged.
    This matches the CUDA version:
        if curRow/curCol is outside the image, skip it.
    """
    h, w, c = img.shape
    assert c == 3

    sums = np.zeros((h, w, 3), dtype=np.int32)
    counts = np.zeros((h, w), dtype=np.int32)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dst_y0 = max(0, -dy)
            dst_y1 = min(h, h - dy)
            dst_x0 = max(0, -dx)
            dst_x1 = min(w, w - dx)

            src_y0 = dst_y0 + dy
            src_y1 = dst_y1 + dy
            src_x0 = dst_x0 + dx
            src_x1 = dst_x1 + dx

            sums[dst_y0:dst_y1, dst_x0:dst_x1, :] += img[src_y0:src_y1, src_x0:src_x1, :]
            counts[dst_y0:dst_y1, dst_x0:dst_x1] += 1

    out = sums // counts[:, :, None]
    return out.astype(np.uint8)


def benchmark_one_radius(img: np.ndarray,
                         input_path: Path,
                         output_dir: Path,
                         radius: int,
                         repeat: int):
    window = 2 * radius + 1

    # Warm-up
    out = blur_rgb_cpu(img, radius)

    start = time.perf_counter()
    for _ in range(repeat):
        out = blur_rgb_cpu(img, radius)
    end = time.perf_counter()

    compute_total_ms = (end - start) * 1000.0
    compute_avg_ms = compute_total_ms / repeat

    output_path = output_dir / f"{input_path.stem}_blur_cpu_r{radius}.png"

    save_start = time.perf_counter()
    Image.fromarray(out).save(output_path)
    save_end = time.perf_counter()

    save_ms = (save_end - save_start) * 1000.0

    print()
    print(f"Radius = {radius}")
    print(f"Blur window = {window} x {window}")
    print("----------------------------------------")
    print(f"CPU compute total time : {compute_total_ms:.4f} ms")
    print(f"CPU compute avg time   : {compute_avg_ms:.4f} ms")
    print(f"Image save time        : {save_ms:.4f} ms")
    print(f"Saved blurred image    : {output_path}")


def main():
    input_path = Path(r"C:\Users\12504\Desktop\6-8\CUDA\chapter3\blur\sample.png")
    repeat = 100

    # These match your CUDA experiments:
    # radius = 3 -> 7 x 7
    # radius = 5 -> 11 x 11
    radius_list = [5]

    # Optional command line:
    # python blur_cpu_timed.py sample.png 100
    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        repeat = int(sys.argv[2])
        if repeat < 1:
            repeat = 1

    output_dir = input_path.parent

    load_start = time.perf_counter()
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    load_end = time.perf_counter()

    h, w, c = img_np.shape
    load_ms = (load_end - load_start) * 1000.0

    print(f"Input image: {input_path}")
    print(f"Width  = {w}")
    print(f"Height = {h}")
    print(f"Loaded channels = {c}")
    print(f"Repeat = {repeat}")
    print(f"Pixels = {w * h}")
    print(f"Image load time = {load_ms:.4f} ms")

    for radius in radius_list:
        benchmark_one_radius(
            img=img_np,
            input_path=input_path,
            output_dir=output_dir,
            radius=radius,
            repeat=repeat
        )


if __name__ == "__main__":
    main()