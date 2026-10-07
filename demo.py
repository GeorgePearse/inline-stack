"""Assembly inside Rust inside Python, all via inline mechanisms.

    uv run python demo.py [N]

Three routes to the same kernel are compared against numpy:
  1. numpy reference        - np.sum
  2. rustimport (pyo3)      - `import kernels_py` compiles kernels_py.rs on first import
  3. ctypes (no pyo3)       - the same crate built as a cdylib, called through its C ABI
"""

from __future__ import annotations

import ctypes
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent

# --- Route 2: rustimport ----------------------------------------------------
# Must be set before the import hook runs; otherwise cargo builds a debug
# binary and the timings are meaningless.
import rustimport.settings  # noqa: E402

rustimport.settings.compile_release_binaries = True

import rustimport.import_hook  # noqa: E402,F401  (installs the hook)

t0 = time.perf_counter()
import kernels_py  # noqa: E402  (compiled from ./kernels_py.rs on first import)

RUSTIMPORT_SECS = time.perf_counter() - t0


# --- Route 3: ctypes ----------------------------------------------------------
def load_cdylib() -> ctypes.CDLL:
    """Build the pyo3-free cdylib with cargo and bind its C ABI."""
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=ROOT, check=True)
    ext = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    prefix = "" if sys.platform == "win32" else "lib"
    lib = ctypes.CDLL(str(ROOT / "target" / "release" / f"{prefix}inline_kernels{ext}"))

    # int inline_stack_sum_f64(const double* ptr, size_t len, double* out)
    lib.inline_stack_sum_f64.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)]
    lib.inline_stack_sum_f64.restype = ctypes.c_int
    # int inline_stack_sum_u64(const uint64_t* ptr, size_t len, uint64_t* out)
    lib.inline_stack_sum_u64.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
    lib.inline_stack_sum_u64.restype = ctypes.c_int
    # const char* inline_stack_which_impl(void)
    lib.inline_stack_which_impl.argtypes = []
    lib.inline_stack_which_impl.restype = ctypes.c_char_p
    return lib


t0 = time.perf_counter()
LIB = load_cdylib()
CTYPES_SECS = time.perf_counter() - t0

STATUS = {0: "ok", 1: "null pointer", 2: "misaligned pointer"}


def ctypes_sum_f64(a: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    out = ctypes.c_double()
    # Zero-copy: hand Rust the raw data pointer and element count.
    rc = LIB.inline_stack_sum_f64(a.ctypes.data, a.size, ctypes.byref(out))
    if rc != 0:
        raise ValueError(f"inline_stack_sum_f64 failed: {STATUS.get(rc, rc)}")
    return out.value


def ctypes_sum_u64(a: np.ndarray) -> int:
    a = np.ascontiguousarray(a, dtype=np.uint64)
    out = ctypes.c_uint64()
    rc = LIB.inline_stack_sum_u64(a.ctypes.data, a.size, ctypes.byref(out))
    if rc != 0:
        raise ValueError(f"inline_stack_sum_u64 failed: {STATUS.get(rc, rc)}")
    return out.value


# --- Correctness ---------------------------------------------------------------
def check_correctness() -> None:
    rng = np.random.default_rng(42)
    # 0, 1, below/at/above the vector widths, and non-multiples of every unroll.
    sizes = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 1001, (1 << 20) + 3]
    for n in sizes:
        f = rng.standard_normal(n) * 100.0
        ref = float(np.sum(f))
        for name, fn in (("rustimport", kernels_py.sum_f64), ("ctypes", ctypes_sum_f64)):
            got = fn(f)
            # Different summation order than numpy's pairwise sum -> tolerance.
            tol = 1e-9 * (1.0 + abs(ref)) * (1.0 + n)
            assert abs(got - ref) <= tol, f"sum_f64/{name} n={n}: {got} != {ref}"

        u = rng.integers(0, 2**64, size=n, dtype=np.uint64)
        ref_u = sum(int(x) for x in u) % 2**64 if n <= 1001 else int(np.sum(u, dtype=np.uint64))
        for name, fn in (("rustimport", kernels_py.sum_u64), ("ctypes", ctypes_sum_u64)):
            got_u = fn(u)
            assert got_u == ref_u, f"sum_u64/{name} n={n}: {got_u} != {ref_u}"

    # Unaligned-to-16/32-byte views (still 8-byte aligned) must also work.
    base = rng.standard_normal(256)
    for off in range(1, 4):
        view = base[off:]
        ref = float(np.sum(view))
        assert abs(kernels_py.sum_f64(view) - ref) <= 1e-9 * (1 + abs(ref)) * 257
        assert abs(ctypes_sum_f64(view) - ref) <= 1e-9 * (1 + abs(ref)) * 257

    # Error paths through the C ABI are status codes, not crashes.
    out = ctypes.c_double()
    assert LIB.inline_stack_sum_f64(None, 5, ctypes.byref(out)) == 1  # null ptr
    assert LIB.inline_stack_sum_f64(None, 0, ctypes.byref(out)) == 0 and out.value == 0.0

    # Non-contiguous input is rejected by the pyo3 route (no silent copy).
    try:
        kernels_py.sum_f64(base[::2])
    except ValueError:
        pass
    else:
        raise AssertionError("strided view should be rejected")

    print(f"correctness: all {len(sizes)} sizes + unaligned views + error paths OK")


# --- Timing ----------------------------------------------------------------------
def best_of(fn, arg, reps: int, inner: int = 1) -> float:
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(inner):
            fn(arg)
        best = min(best, (time.perf_counter() - t) / inner)
    return best


def bench(n: int) -> None:
    f = np.random.default_rng(0).standard_normal(n)
    u = np.arange(n, dtype=np.uint64)
    reps = 7 if n >= 1_000_000 else 20
    inner = 1 if n >= 1_000_000 else 2000
    rows = [
        ("numpy  np.sum(f64)", best_of(np.sum, f, reps, inner)),
        ("rust-asm via rustimport sum_f64", best_of(kernels_py.sum_f64, f, reps, inner)),
        ("rust-asm via ctypes     sum_f64", best_of(ctypes_sum_f64, f, reps, inner)),
        ("numpy  np.sum(u64)", best_of(np.sum, u, reps, inner)),
        ("rust-naked-asm via rustimport sum_u64", best_of(kernels_py.sum_u64, u, reps, inner)),
        ("rust-naked-asm via ctypes     sum_u64", best_of(ctypes_sum_u64, u, reps, inner)),
    ]
    gb = n * 8 / 1e9
    size = f"{n * 8 / 1e6:.0f} MB" if n * 8 >= 1e6 else f"{n * 8 / 1e3:.0f} KB"
    print(f"\nn = {n:,} elements ({size}), best of {reps}:")
    for name, secs in rows:
        unit = f"{secs * 1e6:9.1f} us" if secs < 1e-3 else f"{secs * 1e3:9.2f} ms"
        print(f"  {name:<40} {unit}   {gb / secs:7.1f} GB/s")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}")
    print(f"rustimport import (compile on first run): {RUSTIMPORT_SECS:.2f}s")
    print(f"ctypes cdylib build + load:               {CTYPES_SECS:.2f}s")
    print(f"which_impl via rustimport: {kernels_py.which_impl()}")
    print(f"which_impl via ctypes:     {LIB.inline_stack_which_impl().decode()}")
    check_correctness()
    bench(n)
    bench(1_000)  # small: shows per-call overhead of each route


if __name__ == "__main__":
    main()
