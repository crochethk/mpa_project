use mpa_lib::mp_int::*;

/// Caculates the binomial coefficient `(n|k)` ("n choose k") using *Pascal's triangle*.
/// This allows to use additions only and thus avoid division of potentially big
/// numbers.
fn binomial(n: u64, mut k: u64) -> MPint {
    if k > n / 2 {
        k = n - k;
    }
    let (n, k) = (n as usize, k as usize);

    let mut coeffs = vec![mpint![0]; k + 1];
    coeffs[0] = mpint![1];

    for _row in 1..=n {
        for col in (1..=k).rev() {
            coeffs[col] = &coeffs[col] + &coeffs[col - 1];
        }
    }

    coeffs.pop().unwrap()
}

fn main() {
    test_mpint_binom();

    let n = 10000;
    let k = 42;
    let result = binomial(n, k);
    let exp_str = "652945894617920230937165266791194236925532551338164487629981313209097028731092090795095199544337067516501827475720000";
    let expected = MPint::from_dec_str(exp_str).unwrap();
    assert_eq!(result, expected);
    println!("(n|k) = ({n} | {k}) = {result}")
}

fn test_mpint_binom() {
    // test triangle middle value
    assert_eq!(binomial(4, 2), 6);
    assert_eq!(binomial(2, 1), 2);

    // test triangle symmetry
    assert_eq!(binomial(4, 1), 4);
    assert_eq!(binomial(4, 3), 4);
    assert_eq!(binomial(5, 2), 10);
    assert_eq!(binomial(5, 3), 10);
    assert_eq!(binomial(5, 1), binomial(5, 4));

    // test triangle boundaries
    assert_eq!(binomial(6, 0), 1);
    assert_eq!(binomial(6, 6), 1);

    // test big values
    let b = binomial(100, 49);
    let expected = MPint::from_dec_str("98913082887808032681188722800").unwrap();
    assert_eq!(b, expected);

    let b = binomial(1000, 42);
    let dec_str = "297242911333923795640059429176065863139989673213703918037987737481286092000";
    let expected = MPint::from_dec_str(dec_str).unwrap();
    assert_eq!(b, expected);

    let b = binomial(10000, 42);
    let dec_str = "652945894617920230937165266791194236925532551338164487629981313209097028731092090795095199544337067516501827475720000";
    let expected = MPint::from_dec_str(dec_str).unwrap();
    assert_eq!(b, expected);
}
