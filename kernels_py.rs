// rustimport:pyo3
//
// This file is a complete Python extension module: `import kernels_py` (after
// `import rustimport.import_hook`) compiles it on first use. rustimport wraps
// it in a generated Cargo crate using pyo3, so everything below is ordinary
// pyo3 code. The kernels themselves come from the pyo3-free crate in ./src
// via the path dependency declared in the `//:` manifest lines.
//
//: [dependencies]
//: inline_kernels = { path = "." }
//: pyo3 = { version = "0.29", features = ["extension-module"] }
//
// Recompile when the underlying crate changes, not just this file:
//d: ./Cargo.toml
//d: ./src/*.rs

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Borrow any C-contiguous buffer of `T` (a numpy array, array.array,
/// memoryview...) as a slice, with no copy. Valid while `buf` is alive and
/// the GIL is held, which is for the duration of `f`.
fn with_slice<T: pyo3::buffer::Element + Copy, R>(
    obj: &Bound<'_, PyAny>,
    f: impl FnOnce(&[T]) -> R,
) -> PyResult<R> {
    let buf: PyBuffer<T> = PyBuffer::get(obj)?; // also checks dtype/itemsize
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err("expected a C-contiguous buffer"));
    }
    let n = buf.item_count();
    let ptr = buf.buf_ptr() as *const T;
    if n == 0 {
        return Ok(f(&[]));
    }
    if ptr.is_null() || !ptr.is_aligned() {
        return Err(PyValueError::new_err("buffer pointer is null or misaligned"));
    }
    // SAFETY: the buffer protocol guarantees `ptr` is valid for `n` items of
    // `T` while `buf` lives; we only read; `buf` outlives the call to `f`.
    let xs = unsafe { std::slice::from_raw_parts(ptr, n) };
    Ok(f(xs))
}

/// Sum of a float64 buffer via the inline-asm kernel.
#[pyfunction]
fn sum_f64(xs: &Bound<'_, PyAny>) -> PyResult<f64> {
    with_slice(xs, inline_kernels::sum_f64)
}

/// Wrapping sum of a uint64 buffer via the naked-asm kernel.
#[pyfunction]
fn sum_u64(xs: &Bound<'_, PyAny>) -> PyResult<u64> {
    with_slice(xs, inline_kernels::sum_u64)
}

/// Which architecture / feature variant the kernels selected.
#[pyfunction]
fn which_impl() -> &'static str {
    inline_kernels::which_impl()
}
