//! aarch64 inline assembly (NEON). Compile-checked and assembled in CI-style
//! with `cargo rustc --target aarch64-unknown-linux-gnu`; see README for what
//! was and was not executed.
//!
//! `options(nostack, readonly, pure)` for the same reasons as on x86_64; no
//! `preserves_flags` because `subs` writes NZCV.

use core::arch::{asm, naked_asm};

/// Sum of an f64 slice with NEON `fadd .2d`, 4 doubles per iteration via
/// two accumulators, tail in Rust.
pub fn sum_f64_neon(xs: &[f64]) -> f64 {
    let blocks = xs.len() / 4;
    let vec_sum: f64;
    // SAFETY: ptr covers `blocks * 4` readable f64s; the asm only reads.
    unsafe {
        asm!(
            "movi   {a0:v}.2d, #0",                      // a0 = [0.0, 0.0]
            "movi   {a1:v}.2d, #0",                      // a1 = [0.0, 0.0]
            "cbz    {cnt}, 3f",                        // blocks == 0 -> skip
            "2:",
            "ldp    {t0:q}, {t1:q}, [{p}], #32",       // t0,t1 = xs[i..i+4]; p += 32
            "fadd   {a0:v}.2d, {a0:v}.2d, {t0:v}.2d",        // a0 += t0
            "fadd   {a1:v}.2d, {a1:v}.2d, {t1:v}.2d",        // a1 += t1
            "subs   {cnt}, {cnt}, #1",                 // blocks -= 1
            "b.ne   2b",
            "3:",
            "fadd   {a0:v}.2d, {a0:v}.2d, {a1:v}.2d",        // a0 = 2 partial sums
            "faddp  {a0:d}, {a0:v}.2d",                  // d = a0.lo + a0.hi
            p   = inout(reg) xs.as_ptr() => _,
            cnt = inout(reg) blocks => _,
            a0  = out(vreg) vec_sum,                   // scalar d-view = result
            a1  = out(vreg) _,
            t0  = out(vreg) _,
            t1  = out(vreg) _,
            options(nostack, readonly, pure),
        );
    }
    vec_sum + xs[blocks * 4..].iter().sum::<f64>()
}

/// Wrapping sum of `len` u64s at `ptr` as a naked function under AAPCS64
/// (x0 = ptr, x1 = len, x0 = return).
///
/// # Safety
/// `ptr` must point at `len` readable, 8-byte-aligned u64s (or `len == 0`).
#[unsafe(naked)]
pub unsafe extern "C" fn sum_u64_naked(ptr: *const u64, len: usize) -> u64 {
    naked_asm!(
        "mov    x2, #0",            // acc = 0
        "cbz    x1, 2f",            // len == 0 -> done
        "1:",
        "ldr    x3, [x0], #8",      // x3 = *ptr; ptr += 1
        "add    x2, x2, x3",        // acc += x3 (wrapping)
        "subs   x1, x1, #1",        // len -= 1
        "b.ne   1b",
        "2:",
        "mov    x0, x2",            // return acc
        "ret",
    )
}
