//! This example compares two different implementations of multiple precision integer
//! exponentiation:
//! 1. A "naive" approach using repeated multiplication.
//! 2. A more efficient approach known as "exponentiation by sqauring" or
//!     "binary exponentiation".
//!
//! For each implementation, a simple "benchmark" is performed measuring the mean
//! time required when performing several exponentiations.
//!
//! Note that the naive implementation might need `~10s` to finish, so pleae be
//! patient (the more efficient `~100 ms`).

use mpa_lib::mp_int::*;

use std::time::{Duration, Instant};

/// Calculates `n^k` using "exponentiation by sqauring". This algorithm exploits
/// properties of binary numbers to efficiently convert `n^k` into a sequence of
/// multiplications and squarings, dramatically reducing the total number of
/// multiplications required.
pub fn pow_by_square(mut n: MPint, mut k: u64) -> MPint {
    let mut result = mpint!(1);
    while k != 0 {
        if (k & 1) == 1 {
            result = &result * &n;
        }
        // square `n` in each iteration
        // (=> next binary exponentiation `n^(10)_2 * n^(100)_2 * ...`)
        n = &n * &n;
        k >>= 1;
    }

    result
}

/// Calculates `n^k` by multiplying `n` `k` times.
pub fn naive_pow(n: MPint, k: u64) -> MPint {
    let mut result = mpint![1];
    for _ in 0..k {
        result = &result * &n;
    }
    result
}

fn main() {
    let (min_k, max_k) = (900, 905);
    let n = mpint!(17);

    let mean_time_naive = simple_pow_bench(naive_pow, &n, min_k, max_k);
    println!("naive: {:.2} ms per exponentiation", mean_time_naive.as_secs_f64() * 1e3);

    let mean_time_squaring = simple_pow_bench(pow_by_square, &n, min_k, max_k);
    println!("sqauring: {:.2} ms per exponentiation", mean_time_squaring.as_secs_f64() * 1e3);
}

fn simple_pow_bench(
    pow_fn: fn(MPint, u64) -> MPint,
    n: &MPint,
    min_k: u64,
    max_k: u64,
) -> Duration {
    let mut _results = vec![];
    let start = Instant::now();
    for k in min_k..max_k {
        let res = pow_fn(n.clone(), k);
        _results.push(res);
    }
    let duration = start.elapsed();
    println!("Finished bench in {} ms", duration.as_millis());
    // Return mean time
    duration.div_f64((max_k.abs_diff(min_k)) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_naive_pow() {
        assert_eq!(naive_pow(mpint!(0), 3), 0);
        assert_eq!(naive_pow(mpint!(0), 0), 1);
        assert_eq!(naive_pow(-mpint!(1), 3), -mpint!(1));
        assert_eq!(naive_pow(mpint!(1, 2, 3), 0), 1);
        assert_eq!(naive_pow(mpint!(1, 2, 3), 17), MPint::from_dec_str("1420142403509359177852781719698652221009590078782686034616231072772471190771824266694269633952445461131746532424518782608858954155716817095491700832218624065755982606553003847284748687407657725783927992844963838627338642290463714478968478601766551896999468258907115547484662803164109781732514764424523041227660273920819776218265036408997121629795436250875811839771810913944306662358639526162339844751518475880125944729162078439445352841685526549053889132356962892542880797839030509019711619561660254709912440242894049403689742792681168533637739949352654502683156644045090431164441586732028589507859177118468434554803341658826675730992764510609085643677246894899201").unwrap());
    }

    #[test]
    fn test_pow_by_squaring() {
        assert_eq!(pow_by_square(mpint!(0), 3), 0);
        assert_eq!(pow_by_square(mpint!(1), 0), 1);
        assert_eq!(pow_by_square(-mpint!(1), 3), -mpint!(1));
        assert_eq!(pow_by_square(mpint!(1, 2, 3), 0), 1);
        assert_eq!(pow_by_square(mpint!(1, 2, 3), 17), MPint::from_dec_str("1420142403509359177852781719698652221009590078782686034616231072772471190771824266694269633952445461131746532424518782608858954155716817095491700832218624065755982606553003847284748687407657725783927992844963838627338642290463714478968478601766551896999468258907115547484662803164109781732514764424523041227660273920819776218265036408997121629795436250875811839771810913944306662358639526162339844751518475880125944729162078439445352841685526549053889132356962892542880797839030509019711619561660254709912440242894049403689742792681168533637739949352654502683156644045090431164441586732028589507859177118468434554803341658826675730992764510609085643677246894899201").unwrap());
    }
}
