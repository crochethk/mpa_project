use clap::{Parser, ValueEnum};
use mpa_lib::mp_int::*;
use rand::{Rng, RngCore};
use rand_pcg::Pcg64Mcg;
use std::{
    fmt::{Display, Write as _},
    io::{self, stdin, Write as _},
    str::FromStr,
};
///!
/// Demo CLI to manually test arithmetics on multiple-precision numbers implemented.
/// Run `mpa_demo_cli --help` for usage info.
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

    /// Manually specify operands (base 10) in a loop.
    /// Enter `q` to quit.
    #[arg(long, short, conflicts_with_all(["test_count", "seed"]))]
    interactive: bool,

    /// Base of the input in interactive mode. Output is hex regardless.
    #[arg(long, short, conflicts_with_all(["test_count", "seed"]), default_value="10",
        value_parser(clap::builder::PossibleValuesParser::new(["10", "16"])))]
    base: String,
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

    if args.interactive {
        run_interactive_mode(&args);
    } else {
        run_randomized_mode(&args);
    }
}

const EXIT_CMD: &str = "q";
fn run_interactive_mode(args: &Cli) {
    println!("Enter decimal operand, then hit RETURN");
    loop {
        let lhs = match get_operand_from_user(args, "lhs: ") {
            UserInputResult::Operand(x) => x,
            UserInputResult::Error => continue,
            UserInputResult::ExitCmd => break,
        };

        let rhs = match get_operand_from_user(args, "rhs: ") {
            UserInputResult::Operand(x) => x,
            UserInputResult::Error => continue,
            UserInputResult::ExitCmd => break,
        };

        println!(">>> Calculating <<<\n{}\n{}\n{}", lhs, args.operation, rhs);
        println!(">>> Result <<<\n{}\n", args.operation.apply(lhs, rhs));
    }

    println!("Good bye!");
}

enum UserInputResult {
    Operand(MPint),
    Error,
    ExitCmd,
}

/// Tries to get an operand from user. The result is wrapped in `UserInputResult`,
/// which can contain the actual operand, an error indicator or the exit command.
fn get_operand_from_user(args: &Cli, msg: &str) -> UserInputResult {
    let input = prompt_user_input(msg);
    let input = input.trim();
    if input == EXIT_CMD {
        return UserInputResult::ExitCmd;
    }
    let in_base = u32::from_str(args.base.as_str()).unwrap();

    match {
        match in_base {
            16 => MPint::from_hex_str(&input, args.width),
            _ => MPint::from_dec_str(&input, args.width),
        }
    } {
        Ok(x) => UserInputResult::Operand(x),
        Err(e) => {
            println!("{e}");
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
