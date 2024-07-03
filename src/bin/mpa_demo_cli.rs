use clap::{Parser, ValueEnum};
use mpa_lib::mp_int::*;
use rand::{Rng, RngCore};
use rand_pcg::Pcg64Mcg;
use std::fmt::{Display, Write};
///!
/// Demo CLI to manually test arithmetics on multiple-precision numbers implemented
/// in `mpa_lib`.
///
/// Usage: mpa_demo_cli [OPTIONS] <OPERATION>
///
/// Arguments:
///   <OPERATION>  [possible values: add, sub, mul]
///
/// Options:
///   -w, --width <WIDTH>            Bit-width of operands to perform tests with [default: 256]
///   -n, --test-count <TEST_COUNT>  Number of operations to perform [default: 10]
///   -s, --seed <SEED>              RNG seed (128 bit integer) used for random operands [default: random]
///   -h, --help                     Print help
///   -V, --version                  Print version
///

#[derive(Debug, Parser)]
#[command(version, about, long_about = None, arg_required_else_help = true)]
struct Cli {
    #[arg(value_enum)]
    operation: Operation,

    /// Bit-width of operands to perform tests with
    #[arg(long, short, default_value_t = 256)]
    width: usize,

    /// Number of operations to perform
    #[arg(long, short('n'), default_value_t = 10)]
    test_count: usize,

    /// RNG seed (128 bit integer) used for random operands [default: random]
    #[arg(long, short)]
    seed: Option<u128>,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum Operation {
    Add,
    Sub,
    Mul,
}

impl Operation {
    fn apply(&self, lhs: MPint, rhs: MPint) -> MPint {
        match self {
            Operation::Add => lhs + rhs,
            Operation::Sub => lhs - rhs,
            Operation::Mul => &lhs * &rhs,
        }
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char(match self {
            Operation::Add => '+',
            Operation::Sub => '-',
            Operation::Mul => '*',
        })
    }
}

fn main() {
    let args = Cli::parse();

    run_random_mode(&args);
}

/// Runs randomized tests based on provided args
fn run_random_mode(args: &Cli) {
    //
    // Run radnomized test operations
    //

    let mut header = String::new();
    _ = writeln!(header, "+----------- Test: lhs {} rhs -----------+", args.operation);
    _ = writeln!(header, "| - Mode: Random operands");
    _ = writeln!(header, "| - Operands width: {} bits", MPint::new(args.width).width());
    _ = writeln!(header, "| - Test count: {}", args.test_count);
    _ = writeln!(header, "+---------------------------------------+");
    print!("{}", header);

    let seed: u128 = args.seed.unwrap_or_else(|| rand::random());

    // Get a RNG
    // let mut rng = rand::thread_rng();
    let mut rng = Pcg64Mcg::new(seed);

    let mut test_cnt = args.test_count;
    while test_cnt > 0 {
        // Get random operands
        let lhs = random_mpint(&mut rng, args.width);
        let rhs = random_mpint(&mut rng, args.width);

        let mut str_buff = String::new();
        _ = writeln!(str_buff, "~~~~ TEST {} ~~~~", args.test_count - test_cnt + 1);
        _ = writeln!(str_buff, "lhs = {lhs}");
        _ = writeln!(str_buff, "rhs = {rhs}");
        let result = args.operation.apply(lhs, rhs);

        _ = writeln!(str_buff, "result = {result}");

        print!("{}", str_buff);

        test_cnt -= 1;
    }
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
