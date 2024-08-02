//! This basic example demonstrates a simple factorial `n!` implementation using
//! `MPint` for results.
//!
//! Since factorials grow _very_ rapidly with increasing `n`, a native integer
//! type will overflow for relatively small values of `n`.
//! For example, a `u64` already overflows for `n>20` and `u128` for `n>34`.
//!
use mpa_lib::mp_int::*;

/// Calculates `n!` without the usual danger of overflow.
fn factorial_mpint(n: u64) -> MPint {
    let mut fac = mpint![1];
    for i in 2..=n {
        fac = &fac * &mpint![i];
    }
    fac
}

fn main() {
    let large_n = 419;
    let big_fac = factorial_mpint(large_n);
    let big_fac_str = big_fac.to_dec_string();
    println!("{}! = {}", large_n, big_fac_str);
    /*
    You can verify the results for yourself using python:
        import math
        math.factorial(419) == <the_result_from_this_example>
    */
}
