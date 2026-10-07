# inline-stack

Assembly inside Rust inside Python, each layer "inline": no `.s` files, no
build scripts, no maturin step.

```
Python (demo.py)
 ├─ rustimport ──► kernels_py.rs  (pyo3, compiled on first `import`)
 │                     └─ depends on ──┐
 └─ ctypes ─────► libinline_kernels.so ┴─ src/*.rs   ◄── asm! / naked_asm! blocks
```

The kernel is deliberately boring, a sum over a slice, so the mechanisms are
the point and the assembly stays short enough to read and verify.

## The three inline layers

### 1. Assembly in Rust: `asm!` and `#[unsafe(naked)]` + `naked_asm!`

[`src/x86_64.rs`](src/x86_64.rs) and [`src/aarch64.rs`](src/aarch64.rs).

* `sum_f64` uses `core::arch::asm!`. The block runs the unrolled vector loop;
  the ragged tail (`len % unroll`) is summed in plain Rust so the asm has one
  job. Every operand and every `options(...)` flag is commented in the source.
  In short: `readonly` (we read the slice, so `nomem` would be a lie),
  `nostack`, `pure`, and no `preserves_flags` because the loop counters write
  flags.
* `sum_u64` is a `#[unsafe(naked)]` function (stable since Rust 1.88) whose
  entire body is `naked_asm!`. rustc emits no prologue or epilogue: what you
  see is the whole function, with arguments and return value in the ABI
  registers (`extern "sysv64"` on x86_64 so it is the same on every OS,
  `extern "C"` on aarch64).

### 2. Rust in Python, route A: `rustimport` (pyo3 under the hood)

[`kernels_py.rs`](kernels_py.rs) starts with `// rustimport:pyo3`. After
`import rustimport.import_hook`, a plain `import kernels_py` finds that file,
generates a Cargo crate around it (rustimport uses **pyo3** for the bindings,
you just do not write the `#[pymodule]` boilerplate), builds it with cargo,
and caches the `.so` next to the source. Its `//:` header lines add a path
dependency on the kernel crate in `./src`, and `//d:` lines make rustimport
rebuild when those files change. Buffers come in through the Python buffer
protocol (`PyBuffer<f64>`), so a numpy array is read in place with no copy.

### 3. Rust in Python, route B: `ctypes` (no pyo3 anywhere)

[`src/ffi.rs`](src/ffi.rs) exports the same kernels as
`#[unsafe(no_mangle)] extern "C"` symbols and `Cargo.toml` adds `cdylib` to
the crate types. `demo.py` runs `cargo build --release`, opens
`target/release/libinline_kernels.so` with `ctypes.CDLL`, and passes
`arr.ctypes.data, arr.size` straight through: zero-copy, no Python headers,
no pyo3. The C functions are panic-free (a panic cannot cross `extern "C"`;
it would abort the interpreter), so null and misaligned pointers come back as
status codes rather than crashes.

## The cfg / arch story

Two levels of selection, visible in `which_impl()`:

| Decided at     | Mechanism                                   | Variants                                       |
|----------------|---------------------------------------------|------------------------------------------------|
| compile time   | `#[cfg(target_arch = ...)]` modules         | `x86_64`, `aarch64`, portable Rust elsewhere   |
| run time       | `is_x86_feature_detected!("avx")`           | AVX (`vaddpd ymm`, 8/iter) or SSE2 (`addpd`, 4/iter) |

The build is baseline x86_64 (no `-C target-cpu=native`), so the binary is
portable; the AVX block lives in a `#[target_feature(enable = "avx")]`
function and is only called after the runtime check. Every variant is tested
against the pure-Rust fallback in [`src/portable.rs`](src/portable.rs) on
lengths 0, 1, 2..9, 15, 17 and 1001, plus 8-byte-aligned-but-not-16/32-aligned
views, because a wrong clobber or lane reduction is silent corruption, not a
compile error.

## Run it

Requires `uv`, a Rust toolchain (1.88+; naked functions), and a C linker.

```sh
uv run python demo.py            # 10M elements
uv run python demo.py 1000000    # pick N
cargo test                       # asm vs portable, FFI error paths
cargo clippy --all-targets
cargo rustc --lib --target aarch64-unknown-linux-gnu --crate-type rlib --release   # assembles the aarch64 asm
```

First run compiles pyo3 for the rustimport route (about 11 s here); later runs
hit the cache.

## Sample output (real run, x86_64 Linux, 16 cores, AVX2 box)

```
python 3.14.6  numpy 2.5.3
rustimport import (compile on first run): 0.00s
ctypes cdylib build + load:               0.03s
which_impl via rustimport: x86_64/avx (asm!, runtime-detected) + naked_asm!
which_impl via ctypes:     x86_64/avx (asm!, runtime-detected) + naked_asm!
correctness: all 16 sizes + unaligned views + error paths OK

n = 10,000,000 elements (80 MB), best of 7:
  numpy  np.sum(f64)                            3.18 ms      25.1 GB/s
  rust-asm via rustimport sum_f64               2.52 ms      31.8 GB/s
  rust-asm via ctypes     sum_f64               2.31 ms      34.7 GB/s
  numpy  np.sum(u64)                            2.79 ms      28.6 GB/s
  rust-naked-asm via rustimport sum_u64         3.73 ms      21.4 GB/s
  rust-naked-asm via ctypes     sum_u64         3.74 ms      21.4 GB/s

n = 1,000 elements (8 KB), best of 20:
  numpy  np.sum(f64)                             1.7 us       4.8 GB/s
  rust-asm via rustimport sum_f64                0.3 us      29.6 GB/s
  rust-asm via ctypes     sum_f64                2.2 us       3.6 GB/s
  numpy  np.sum(u64)                             1.6 us       4.9 GB/s
  rust-naked-asm via rustimport sum_u64          0.5 us      16.4 GB/s
  rust-naked-asm via ctypes     sum_u64          2.5 us       3.2 GB/s
```

Reading it: at 80 MB everything is memory-bound and the AVX `asm!` sum edges
numpy. The naked `sum_u64` is a single scalar `add` chain per iteration, by
design (it is a readable demo of a naked function, not a tuned kernel), and
numpy's vectorised reduction beats it. At 8 KB the per-call overhead
dominates: the pyo3 route costs ~0.3 µs, the ctypes route ~2 µs.

## Honest caveats

* **rustimport is pyo3 under the hood.** You skip the boilerplate, not the
  dependency: first import compiles pyo3 and its macros, and the result is
  tied to the Python ABI it was built for. The ctypes route has none of that,
  at the cost of a hand-written C ABI.
* **ctypes call overhead.** Each ctypes call goes through `ascontiguousarray`,
  argument marshalling and a `byref` out-parameter: roughly 2 µs here, which
  swamps a 1 000-element sum. Batch the work or use the pyo3 route when calls
  are small and frequent.
* **LLVM cannot optimise through `asm!`.** The block is opaque: no inlining,
  no auto-vectorisation of the tail into the loop, no constant folding if the
  input is known. The `pure`/`readonly` options let LLVM reorder or drop the
  call, but never rewrite it. Write asm when you have measured that the
  compiler's output is the problem; the SSE2 path here is not faster than
  what rustc emits for `iter().sum()` with `-C target-cpu=native`, it is just
  explicit.
* **Floating-point order differs.** Vector lanes are reduced at the end, so
  the asm sum and numpy's pairwise sum agree to rounding, not bit-for-bit.
  The checks use a tolerance.
* **aarch64 is compiled, not executed.** This machine has no aarch64 hardware,
  qemu, or cross-linker. The NEON `asm!` and naked function were assembled by
  LLVM via `cargo rustc --target aarch64-unknown-linux-gnu --crate-type rlib`
  and pass clippy for that target, but the unit tests have not been run on
  arm64. If you run it on a Mac or Graviton box and it misbehaves, open an
  issue.
* **No Windows x86_64 testing.** The naked function uses `extern "sysv64"`
  precisely so the Windows ABI is not a concern, but nothing here was run
  there.

## Layout

```
Cargo.toml        crate inline_kernels: rlib (for rustimport) + cdylib (for ctypes)
src/lib.rs        public API, cfg dispatch, which_impl(), cross-variant tests
src/x86_64.rs     asm! SSE2 + AVX sum_f64, naked_asm! sum_u64
src/aarch64.rs    asm! NEON sum_f64, naked_asm! sum_u64
src/portable.rs   pure-Rust reference used as the fallback and the test oracle
src/ffi.rs        extern "C" surface, panic-free, status codes
kernels_py.rs     rustimport single-file pyo3 module (buffer-protocol, zero-copy)
demo.py           driver: both routes vs numpy, correctness, timings
pyproject.toml    uv project: numpy + rustimport, Python >= 3.11
```

MIT, George Pearse.
