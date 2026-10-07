//! Pure-Rust reference implementations. These are what every other target
//! runs, and what the assembly variants are tested against.

/// Straightforward left-to-right sum.
pub fn sum_f64(xs: &[f64]) -> f64 {
    xs.iter().sum()
}

/// Wrapping sum, matching the assembly's `add` semantics on overflow.
pub fn sum_u64(xs: &[u64]) -> u64 {
    xs.iter().fold(0u64, |acc, &x| acc.wrapping_add(x))
}
