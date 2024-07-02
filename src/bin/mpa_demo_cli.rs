///!
/// Demo CLI to manually test arithmetics on multiple-precision numbers implemented
/// in `mpa_lib`.
///
use mpa_lib::mp_int::*;
use rand::{Rng, RngCore};
use rand_pcg::Pcg64Mcg;

fn main() {
    let usr_seed: u128 = 0xcafef00dd15ea5e5; // default PCG seed

    let usr_num_width: usize = 1234;
    let _usr_test_count: usize = 123;

    // Get a RNG
    // let mut rng = rand::thread_rng();
    let mut rng = Pcg64Mcg::new(usr_seed);

    let lhs = random_mpint(&mut rng, usr_num_width);
    let rhs = random_mpint(&mut rng, usr_num_width);

    let operation = "some operation verb";

    let result = match operation {
        "add" => &lhs + &rhs,
        _ => &lhs + &rhs,
    };

    println!("+--------------------------------+");
    println!("| Performing: lhs + rhs = result |");
    println!("+--------------------------------+");

    let mut test_cnt = 0;
    test_cnt += 1;
    println!("~~~~ TEST {test_cnt} ~~~~");
    println!("lhs = {lhs}");
    println!("rhs = {rhs}");
    println!("result = {result}");
    println!("");

    test_cnt += 1;
    println!("~~~~ TEST {test_cnt} ~~~~");
    println!("lhs = {lhs}");
    println!("rhs = {rhs}");
    println!("result = {result}");
    println!("");
}

fn random_mpint<R: RngCore>(rng: &mut R, width: usize) -> MPint {
    let mut bins = vec![0 as DigitT; MPint::width_to_bins_count(width)];
    rng.fill(&mut bins[..]);

    let num = MPint::new(bins);
    if rng.gen_bool(0.5) {
        -num
    } else {
        num
    }
}
