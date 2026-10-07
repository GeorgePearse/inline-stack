//! x86_64 inline assembly. Intel syntax (the `asm!` default).
//!
//! `options(...)` used throughout:
//! * `nostack`  - the asm never pushes/pops or touches memory below rsp, so
//!   the compiler need not keep the stack aligned/red-zone-safe around it.
//! * `readonly` - the asm reads memory (the slice) but never writes it, so
//!   the compiler may keep other memory in registers across the block. We do
//!   NOT claim `nomem`, which would be a lie and let LLVM reorder stores to
//!   the slice past the asm.
//! * `pure`     - no side effects; outputs depend only on inputs + memory.
//!   Together with `readonly` this lets the call be CSE'd/eliminated if its
//!   result is unused. Every block here has an output, as `pure` requires.
//!
//! We never claim `preserves_flags` because `test`/`dec` write RFLAGS.
//! Every register the asm writes is declared as an operand or a clobber,
//! so the compiler never relies on a value we trash (wrong clobbers =
//! silent corruption, which is why the tests compare against pure Rust).

use core::arch::{asm, naked_asm};

/// Sum of an f64 slice using baseline SSE2 (`addpd`, 2 lanes), 4 doubles
/// per loop iteration via two independent accumulators.
pub fn sum_f64_sse2(xs: &[f64]) -> f64 {
    let blocks = xs.len() / 4; // doubles handled by the asm
    let vec_sum: f64;
    // SAFETY: `ptr` points at `blocks * 4` readable f64s inside `xs`; the asm
    // only reads them. All written registers are declared as outputs.
    unsafe {
        asm!(
            "xorpd  {a0}, {a0}",            // a0 = [0.0, 0.0]  accumulator 0
            "xorpd  {a1}, {a1}",            // a1 = [0.0, 0.0]  accumulator 1
            "test   {cnt}, {cnt}",          // blocks == 0 ?
            "jz     3f",                    //   -> skip loop
            "2:",
            "movupd {t0}, xmmword ptr [{p}]",      // t0 = xs[i..i+2]  (unaligned OK)
            "movupd {t1}, xmmword ptr [{p} + 16]", // t1 = xs[i+2..i+4]
            "addpd  {a0}, {t0}",            // a0 += t0 (two lanes)
            "addpd  {a1}, {t1}",            // a1 += t1
            "add    {p}, 32",               // advance 4 doubles
            "dec    {cnt}",                 // one block done
            "jnz    2b",                    // loop while blocks remain
            "3:",
            "addpd  {a0}, {a1}",            // a0 = [a0.lo+a1.lo, a0.hi+a1.hi]
            "movapd {t0}, {a0}",            // t0 = a0
            "unpckhpd {t0}, {t0}",          // t0.lo = a0.hi
            "addsd  {a0}, {t0}",            // a0.lo = a0.lo + a0.hi  (horizontal sum)
            p   = inout(reg) xs.as_ptr() => _, // data pointer, advanced, then dead
            cnt = inout(reg) blocks => _,      // loop counter, counted down to 0
            a0  = out(xmm_reg) vec_sum,        // low lane holds the result
            a1  = out(xmm_reg) _,              // scratch accumulator
            t0  = out(xmm_reg) _,              // scratch load register
            t1  = out(xmm_reg) _,              // scratch load register
            options(nostack, readonly, pure),
        );
    }
    // Tail (len % 4) in plain Rust: keeps the asm short and obviously correct.
    vec_sum + xs[blocks * 4..].iter().sum::<f64>()
}

/// Sum of an f64 slice using AVX (`vaddpd` on ymm, 4 lanes), 8 doubles per
/// iteration via two accumulators.
///
/// # Safety
/// The CPU must support AVX. Callers gate on `is_x86_feature_detected!("avx")`.
/// (`#[target_feature]` is what lets us name `ymm_reg` operands at all; it
/// does not change how the *rest* of the crate is compiled.)
#[target_feature(enable = "avx")]
pub unsafe fn sum_f64_avx(xs: &[f64]) -> f64 {
    let blocks = xs.len() / 8;
    let vec_sum: f64;
    // SAFETY: as in sum_f64_sse2; VEX-encoded loads tolerate any alignment.
    unsafe {
        asm!(
            "vxorpd {a0:y}, {a0:y}, {a0:y}",      // a0 = 4 x 0.0
            "vxorpd {a1:y}, {a1:y}, {a1:y}",      // a1 = 4 x 0.0
            "test   {cnt}, {cnt}",
            "jz     3f",
            "2:",
            "vaddpd {a0:y}, {a0:y}, ymmword ptr [{p}]",      // a0 += xs[i..i+4]
            "vaddpd {a1:y}, {a1:y}, ymmword ptr [{p} + 32]", // a1 += xs[i+4..i+8]
            "add    {p}, 64",               // advance 8 doubles
            "dec    {cnt}",
            "jnz    2b",
            "3:",
            "vaddpd {a0:y}, {a0:y}, {a1:y}",      // a0 = 4 partial sums
            "vextractf128 {t:x}, {a0:y}, 1",  // t.xmm = upper 2 lanes of a0
            "vaddpd {a0:x}, {a0:x}, {t:x}", // a0.xmm = 2 partial sums
            "vunpckhpd {t:x}, {a0:x}, {a0:x}", // t.lo = a0.hi
            "vaddsd {a0:x}, {a0:x}, {t:x}", // a0.lo = final sum
            "vzeroupper",                   // avoid SSE/AVX transition penalty on return
            p   = inout(reg) xs.as_ptr() => _,
            cnt = inout(reg) blocks => _,
            a0  = out(ymm_reg) vec_sum,     // low lane = result ({a0:x} = its xmm name)
            a1  = out(ymm_reg) _,
            t   = out(ymm_reg) _,
            options(nostack, readonly, pure),
        );
    }
    vec_sum + xs[blocks * 8..].iter().sum::<f64>()
}

/// Wrapping sum of `len` u64s starting at `ptr`, as a *naked* function:
/// no prologue/epilogue is generated, the body is exactly the assembly below,
/// and arguments/return value follow the explicit `sysv64` ABI
/// (rdi = ptr, rsi = len, rax = return) on every OS, including Windows.
///
/// # Safety
/// `ptr` must point at `len` readable, 8-byte-aligned u64s (or `len == 0`).
#[unsafe(naked)]
pub unsafe extern "sysv64" fn sum_u64_naked(ptr: *const u64, len: usize) -> u64 {
    naked_asm!(
        "xor    eax, eax",              // rax = 0 (zero-extends)        accumulator
        "test   rsi, rsi",              // len == 0 ?
        "jz     2f",
        "1:",
        "add    rax, qword ptr [rdi]",  // rax += *ptr   (wrapping)
        "add    rdi, 8",                // ptr += 1
        "dec    rsi",                   // len -= 1
        "jnz    1b",
        "2:",
        "ret",                          // return value already in rax
    )
}
