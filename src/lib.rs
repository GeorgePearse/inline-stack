//! Three kernels, each with a per-architecture inline-assembly implementation
//! and a portable pure-Rust fallback:
//!
//! * [`sum_f64`]  - sum of an `f64` slice, written with `core::arch::asm!`
//!   (x86_64: SSE2 baseline + runtime-detected AVX; aarch64: NEON).
//! * [`sum_u64`]  - sum of a `u64` slice, written as a `#[unsafe(naked)]`
//!   function whose *entire* body is `naked_asm!` (x86_64 and aarch64).
//! * [`which_impl`] - reports which variant the running binary selected.
//!
//! The `ffi` module re-exports them as `extern "C"` symbols for ctypes.
//! Nothing here depends on pyo3; the pyo3 glue lives in `kernels_py.rs`
//! at the repo root and is compiled by rustimport.

pub mod portable;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod ffi;

/// Sum of a slice of `f64`, computed in inline assembly where available.
///
/// Summation order differs between variants (vector lanes are reduced at the
/// end), so results agree with a naive loop only up to floating-point
/// rounding. Callers compare with a tolerance; see the tests.
pub fn sum_f64(xs: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            // SAFETY: the CPU reports AVX, which is the only precondition.
            return unsafe { x86_64::sum_f64_avx(xs) };
        }
        return x86_64::sum_f64_sse2(xs);
    }
    #[cfg(target_arch = "aarch64")]
    {
        return aarch64::sum_f64_neon(xs);
    }
    #[allow(unreachable_code)]
    portable::sum_f64(xs)
}

/// Sum of a slice of `u64` (wrapping), computed by a naked assembly function
/// where available.
pub fn sum_u64(xs: &[u64]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: ptr/len come from a valid slice.
        return unsafe { x86_64::sum_u64_naked(xs.as_ptr(), xs.len()) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ptr/len come from a valid slice.
        return unsafe { aarch64::sum_u64_naked(xs.as_ptr(), xs.len()) };
    }
    #[allow(unreachable_code)]
    portable::sum_u64(xs)
}

/// Which implementation [`sum_f64`] will run on this machine.
///
/// The architecture half is decided at compile time by `cfg(target_arch)`;
/// the `avx`/`sse2` half on x86_64 is decided at runtime by
/// `is_x86_feature_detected!`.
pub fn which_impl() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            "x86_64/avx (asm!, runtime-detected) + naked_asm!"
        } else {
            "x86_64/sse2 (asm!, baseline) + naked_asm!"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "aarch64/neon (asm!) + naked_asm!"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "portable Rust fallback (no inline asm on this target)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lengths chosen to hit: empty, single, below one vector, exactly one
    /// unrolled block (8 for AVX, 4 for SSE2/NEON), and non-multiples of
    /// every unroll factor so the tail path is exercised.
    const LENS: [usize; 12] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 17, 1001];

    fn f64_input(n: usize) -> Vec<f64> {
        // Deterministic, mixed-sign, not all representable sums - so a wrong
        // lane reduction or clobber shows up as a real mismatch.
        (0..n).map(|i| ((i * 7919) % 1000) as f64 * 0.37 - 180.0).collect()
    }

    fn u64_input(n: usize) -> Vec<u64> {
        (0..n).map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect()
    }

    fn assert_close(a: f64, b: f64, n: usize) {
        let tol = 1e-9 * (1.0 + b.abs()) * (1.0 + n as f64);
        assert!((a - b).abs() <= tol, "n={n}: asm {a} vs portable {b}");
    }

    #[test]
    fn sum_f64_matches_portable_on_all_lengths() {
        for &n in &LENS {
            let xs = f64_input(n);
            assert_close(sum_f64(&xs), portable::sum_f64(&xs), n);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_x86_variant_matches_portable() {
        for &n in &LENS {
            let xs = f64_input(n);
            let want = portable::sum_f64(&xs);
            assert_close(x86_64::sum_f64_sse2(&xs), want, n);
            if std::arch::is_x86_feature_detected!("avx") {
                assert_close(unsafe { x86_64::sum_f64_avx(&xs) }, want, n);
            }
        }
    }

    #[test]
    fn sum_u64_matches_portable_on_all_lengths() {
        for &n in &LENS {
            let xs = u64_input(n);
            assert_eq!(sum_u64(&xs), portable::sum_u64(&xs), "n={n}");
        }
    }

    #[test]
    fn sum_u64_wraps_like_portable() {
        let xs = [u64::MAX, 2, u64::MAX];
        assert_eq!(sum_u64(&xs), portable::sum_u64(&xs));
    }

    #[test]
    fn unaligned_f64_slices_are_fine() {
        // Offsetting by one element keeps 8-byte alignment (required for
        // &[f64]) but breaks 16/32-byte alignment, which the loads must not
        // assume.
        let xs = f64_input(64);
        for off in 0..4 {
            let s = &xs[off..];
            assert_close(sum_f64(s), portable::sum_f64(s), s.len());
        }
    }

    #[test]
    fn which_impl_is_consistent_with_cfg() {
        let s = which_impl();
        if cfg!(target_arch = "x86_64") {
            assert!(s.starts_with("x86_64/"));
        } else if cfg!(target_arch = "aarch64") {
            assert!(s.starts_with("aarch64/"));
        } else {
            assert!(s.contains("portable"));
        }
    }
}
