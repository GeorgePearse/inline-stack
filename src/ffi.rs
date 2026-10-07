//! Plain C ABI for ctypes. No pyo3, no Python headers.
//!
//! Every function is panic-free: `extern "C"` cannot unwind, so a panic
//! here would abort the Python process. Inputs are validated and a status
//! code is returned instead.

use core::ffi::{c_char, c_int};

/// Status codes returned by the `inline_stack_sum_*` functions.
pub const STATUS_OK: c_int = 0;
pub const STATUS_NULL_PTR: c_int = 1;
pub const STATUS_MISALIGNED: c_int = 2;

/// Sum `len` f64s at `ptr` into `*out`. Returns a status code.
///
/// # Safety
/// If non-null, `ptr` must point at `len` readable f64s and `out` must be
/// writable. `len == 0` is fine with any `ptr`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inline_stack_sum_f64(ptr: *const f64, len: usize, out: *mut f64) -> c_int {
    if out.is_null() {
        return STATUS_NULL_PTR;
    }
    if len == 0 {
        // SAFETY: `out` is non-null and the caller promised it is writable.
        unsafe { *out = 0.0 };
        return STATUS_OK;
    }
    if ptr.is_null() {
        return STATUS_NULL_PTR;
    }
    if !ptr.is_aligned() {
        return STATUS_MISALIGNED;
    }
    // SAFETY: non-null, aligned, and the caller promised `len` readable elements.
    let xs = unsafe { core::slice::from_raw_parts(ptr, len) };
    // SAFETY: `out` is non-null and writable.
    unsafe { *out = crate::sum_f64(xs) };
    STATUS_OK
}

/// Wrapping sum of `len` u64s at `ptr` into `*out`. Returns a status code.
///
/// # Safety
/// As for [`inline_stack_sum_f64`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inline_stack_sum_u64(ptr: *const u64, len: usize, out: *mut u64) -> c_int {
    if out.is_null() {
        return STATUS_NULL_PTR;
    }
    if len == 0 {
        // SAFETY: `out` is non-null and writable.
        unsafe { *out = 0 };
        return STATUS_OK;
    }
    if ptr.is_null() {
        return STATUS_NULL_PTR;
    }
    if !ptr.is_aligned() {
        return STATUS_MISALIGNED;
    }
    // SAFETY: non-null, aligned, `len` readable elements.
    let xs = unsafe { core::slice::from_raw_parts(ptr, len) };
    // SAFETY: `out` is non-null and writable.
    unsafe { *out = crate::sum_u64(xs) };
    STATUS_OK
}

/// NUL-terminated static string naming the selected implementation.
/// The pointer is valid for the lifetime of the process; do not free it.
#[unsafe(no_mangle)]
pub extern "C" fn inline_stack_which_impl() -> *const c_char {
    // Built once from the &'static str; leaking a single small CString is
    // the simplest way to hand C a stable NUL-terminated pointer.
    use std::sync::OnceLock;
    static NAME: OnceLock<std::ffi::CString> = OnceLock::new();
    NAME.get_or_init(|| {
        // `which_impl()` contains no interior NULs, so this cannot fail; the
        // fallback keeps the function panic-free regardless.
        std::ffi::CString::new(crate::which_impl()).unwrap_or_default()
    })
    .as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_and_empty_inputs_do_not_panic() {
        let mut out = f64::NAN;
        assert_eq!(unsafe { inline_stack_sum_f64(core::ptr::null(), 0, &mut out) }, STATUS_OK);
        assert_eq!(out, 0.0);
        assert_eq!(unsafe { inline_stack_sum_f64(core::ptr::null(), 5, &mut out) }, STATUS_NULL_PTR);
        assert_eq!(unsafe { inline_stack_sum_f64([1.0].as_ptr(), 1, core::ptr::null_mut()) }, STATUS_NULL_PTR);
        let mut out_u = 7u64;
        assert_eq!(unsafe { inline_stack_sum_u64(core::ptr::null(), 3, &mut out_u) }, STATUS_NULL_PTR);
        assert_eq!(out_u, 7);
    }

    #[test]
    fn misaligned_pointer_is_rejected() {
        let buf = [0u8; 32];
        let p = unsafe { buf.as_ptr().add(1) } as *const f64;
        let mut out = 0.0;
        assert_eq!(unsafe { inline_stack_sum_f64(p, 1, &mut out) }, STATUS_MISALIGNED);
    }

    #[test]
    fn ffi_sum_matches_safe_api() {
        let xs: Vec<f64> = (0..37).map(|i| i as f64 * 0.5).collect();
        let mut out = 0.0;
        assert_eq!(unsafe { inline_stack_sum_f64(xs.as_ptr(), xs.len(), &mut out) }, STATUS_OK);
        assert_eq!(out, crate::sum_f64(&xs));
        let us: Vec<u64> = (0..37).collect();
        let mut out_u = 0;
        assert_eq!(unsafe { inline_stack_sum_u64(us.as_ptr(), us.len(), &mut out_u) }, STATUS_OK);
        assert_eq!(out_u, 36 * 37 / 2);
    }

    #[test]
    fn which_impl_is_nul_terminated_and_matches() {
        let s = unsafe { core::ffi::CStr::from_ptr(inline_stack_which_impl()) };
        assert_eq!(s.to_str().unwrap(), crate::which_impl());
    }
}
