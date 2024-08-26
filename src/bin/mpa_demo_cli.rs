//! # Demo CLI
//!
//! This CLI allows for manual testing of the multiple precision arithmetics implemented
//! in `mpa_lib`.
//!
//! ## Usage
//!
//! There are two modes, each of which is introduced below.
//! Both require the user to specify an operation (e.g., "add").
//!
//! By default (w/o arguments) "add" and the "random test operations mode" is used.
//!
//! *For a comprehensive list of all available options and further usage
//! information, please refer to `cli --help`.*
//!
//! ### Random test operations mode
//!
//! This mode is meant to automatically generate and simulate a number of runs of
//! a given operation type, on random operands.
//!
//! There are two randomized aspects here:
//! - the operands' `width`s
//! - the operand value within that random width.
//!
//! The __width range__ can be adjusted by specifying the lower and upper
//! boundaries, so that then `min_width ≤ width ≤ max_width`. You can also set a
//! particular width by specify `min_width == max_width`.
//! *Note that this doesn't directly influence the mimimum values of the operands,
//! which might still fit inside a bit-width, shorter than the specified.*
//!
//! Optionally you can __export__ the generated operations and their calculated
//! results into a text file. Each line in this file will have the format
//! `<lhs> <operator> <rhs> == <result>` (e.g. "12+3==15"), where all values are
//! represented in decimals by default. This way you can easily copy 'n paste one
//! or multiple lines into a Python REPL to verify the respective results.
//!
//! #### Examples
//!
//! - Run with defaults:
//!     ```shell
//!     cargo run --bin cli
//!     ```
//!
//! - Run 2 random addition operations:
//!     ```shell
//!     cargo run --bin cli -- add -n 2
//!     ```
//!
//! - Export results of 15 random multiply operations to `./out.txt`:
//!     ```shell
//!     cargo run --bin cli -- mul -n 15 --export "./out.txt"
//!     ```
//!
//! ### Interactive mode
//!
//! This mode starts a basic "Read–Eval–Print Loop" (REPL), where you can manually
//! specify the operands, whereafter the chosen operation is applied and the result
//! printed.
//!
//! The `base` for operands and result output can be set to either `10` or `16`.
//!
//! #### Example
//!
//! - Interactive multiplication mode using hexadecimals:
//!     ```shell
//!     cargo run --bin cli -- mul -i -b 16
//!     ```
//!

use clap::{Parser, ValueEnum};
use mpa_lib::{mp_int::*, utils::ParseError};
use rand::{Rng, RngCore};
use rand_pcg::Pcg64Mcg;
use std::{
    fmt::{Display, Write as _},
    fs::File,
    io::{self, stdin, Write as _},
    ops::RangeInclusive,
    str::FromStr,
};

/// Options exlusive to the random test mode
const RAND_TEST_MODE_OPTS: [&'static str; 5] =
    ["min_width", "max_width", "test_count", "seed", "export"];

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[arg(value_enum, default_value_t = Operation::Add)]
    operation: Operation,

    /// Base for number outputs and where applicable inputs.
    #[arg(long, short, default_value="10",
        value_parser(clap::builder::PossibleValuesParser::new(["10", "16"])))]
    base: String,

    /// Number of test operations to perform
    #[arg(long, short('n'), default_value_t = 5)]
    test_count: usize,

    /// RNG seed (128 bit integer) used for random operands [default: random]
    #[arg(long, short)]
    seed: Option<u128>,

    /// Export the random operations along with the results into the specified file.
    #[arg(long)]
    export: Option<String>,

    /// Lower bit-width boundary for randomly chosen operands (≥64).
    #[arg(long, default_value_t = 64)]
    min_width: usize,

    /// Upper bit-width boundary for randomly chosen operands (≥64).
    #[arg(long, default_value_t = 512)]
    max_width: usize,

    /// Interactive mode. Allows manually specify operands in a loop.
    /// Enter `q` to quit.
    #[arg(long, short, conflicts_with_all(RAND_TEST_MODE_OPTS))]
    interactive: bool,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
pub enum Operation {
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

    if args.interactive {
        run_interactive_mode(&args);
    } else {
        run_randomized_mode(&args);
    }
}

const EXIT_CMD: &str = "q";
fn run_interactive_mode(args: &Cli) {
    println!("Enter decimal operand, then hit RETURN");

    let base = u32::from_str(args.base.as_str()).unwrap();
    let (to_base_string, from_base_str) = base_converters(base);

    loop {
        let lhs = match get_operand_from_user(from_base_str, "lhs: ") {
            UserInputResult::Operand(x) => x,
            UserInputResult::Error => continue,
            UserInputResult::ExitCmd => break,
        };

        let rhs = match get_operand_from_user(from_base_str, "rhs: ") {
            UserInputResult::Operand(x) => x,
            UserInputResult::Error => continue,
            UserInputResult::ExitCmd => break,
        };

        println!(
            ">>> Calculating <<<\n{}\n{}\n{}",
            to_base_string(&lhs),
            args.operation,
            to_base_string(&rhs)
        );
        println!(">>> Result <<<\n{}\n", to_base_string(&args.operation.apply(lhs, rhs)));
    }

    println!("Good bye!");
}

type FromBaseResult = Result<MPint, ParseError>;

/// Returns function pointers for the conversions `to` and `from` the given `base`
/// string representation.
fn base_converters(base: u32) -> (fn(&MPint) -> String, fn(&str) -> FromBaseResult) {
    let result: (fn(&MPint) -> String, fn(&str) -> FromBaseResult) = match base {
        16 => (MPint::to_hex_string, MPint::from_hex_str),
        10 => (MPint::to_dec_string, MPint::from_dec_str),
        _ => panic!("illegal base"),
    };
    result
}

enum UserInputResult {
    Operand(MPint),
    Error,
    ExitCmd,
}

/// Tries to get an operand from user. The result is wrapped in `UserInputResult`,
/// which can contain the actual operand, an error indicator or the exit command.
fn get_operand_from_user(from_base_fn: fn(&str) -> FromBaseResult, msg: &str) -> UserInputResult {
    let input = prompt_user_input(msg);
    let input = input.trim();
    if input == EXIT_CMD {
        return UserInputResult::ExitCmd;
    }

    match from_base_fn(&input) {
        Ok(x) => UserInputResult::Operand(x),
        Err(e) => {
            eprintln!("{e}");
            UserInputResult::Error
        }
    }
}

/// Gets input from `stdin` showing the given message.
fn prompt_user_input(msg: &str) -> String {
    print!("{msg}");
    io::stdout().flush().unwrap();
    let mut buff = String::new();
    stdin().read_line(&mut buff).unwrap();
    buff
}

//--------------------------------------------------------------------------------------------------
/// Runs randomized tests based on provided args
fn run_randomized_mode(args: &Cli) {
    //
    // Run randomized test operations
    //

    let base = u32::from_str(args.base.as_str()).unwrap();
    let (to_base_string, _) = base_converters(base);

    let mut header = String::new();
    _ = writeln!(header, "+----------- Test: lhs {} rhs -----------+", args.operation);
    _ = writeln!(header, " - Mode: Random operands");
    _ = writeln!(header, " - output base: {}", base);
    _ = writeln!(header, " - min_width: {} bits", MPint::new_with_width(args.min_width).width());
    _ = writeln!(header, " - max_width: {} bits", MPint::new_with_width(args.max_width).width());
    _ = writeln!(header, " - Test count: {}", args.test_count);
    _ = writeln!(header, "+---------------------------------------+");
    print!("{}", header);

    // Get a RNG
    let seed: u128 = args.seed.unwrap_or_else(|| rand::random());
    let mut rng = Pcg64Mcg::new(seed);

    // Perform operations
    let mut out_buff = String::new();
    let mut test_cnt = args.test_count;
    let export_enabled = args.export.is_some();
    while test_cnt > 0 {
        // Get random operands
        let width_range = args.min_width..=args.max_width;
        let lhs = random_mpint(&mut rng, width_range.clone());
        let rhs = random_mpint(&mut rng, width_range);

        if export_enabled {
            _ = write!(
                out_buff,
                "{}{}{}==",
                to_base_string(&lhs),
                args.operation,
                to_base_string(&rhs)
            );
            let result = args.operation.apply(lhs, rhs);
            _ = write!(out_buff, "{}\n", to_base_string(&result));
        } else {
            _ = writeln!(out_buff, "~~~~ TEST {} ~~~~", args.test_count - test_cnt + 1);
            _ = writeln!(out_buff, "lhs = {}", to_base_string(&lhs));
            _ = writeln!(out_buff, "rhs = {}", to_base_string(&rhs));
            let result = args.operation.apply(lhs, rhs);
            _ = writeln!(out_buff, "result = {}", to_base_string(&result));
        }

        test_cnt -= 1;
    }

    if export_enabled {
        let filepath = args.export.as_ref().unwrap();
        match write_to_file(filepath, &out_buff) {
            Ok(()) => println!("Results exported to '{}'", filepath),
            Err(e) => eprintln!("Export failed: {}", e),
        }
    } else {
        print!("{}", out_buff);
    }
}

fn random_mpint<R: RngCore>(rng: &mut R, width_range: RangeInclusive<usize>) -> MPint {
    let width = rng.gen_range(width_range);
    let mut bins = vec![0 as DigitT; MPint::width_to_bins_count(width)];
    rng.fill(&mut bins[..]);

    let num = MPint::new(bins);
    if rng.gen_bool(0.5) {
        -num
    } else {
        num
    }
}

fn write_to_file(path: &str, data: &str) -> io::Result<()> {
    let mut file = File::create(path)?;
    write!(file, "{}", data)
}
