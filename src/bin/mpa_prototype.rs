use crate::mp_num_reader::*;

fn main() {
    let num_str = "1234";
    let digits = parse_to_digits(num_str);
    println!("{digits:?}");

    println!("\n---- Convert very large number from string to a list of u8 digits ----");
    let num_str = BIG_NUM_PI_INT_1000;
    let digits = parse_to_digits(num_str);
    println!("{digits:?}");
    println!("{num_str:?}");

    let num_str = BIG_NUM_200_DIGITS;
    let digits = parse_to_digits(BIG_NUM_200_DIGITS);
    println!("{digits:?}");
    println!("{num_str:?}");

    println!("\n---- Determining required digitd amount ----");
    let mut b_w = 2048;
    println!(
        "→ e.g. {b_w} Bit require up to → {} ← decimal digits",
        bit_to_dec_width(b_w)
    );

    b_w = 617;
    println!(
        "→ e.g. {b_w} Bit require up to → {} ← decimal digits",
        bit_to_dec_width(b_w)
    );

    b_w = 10; // → ≤(1023)_10 → 4 digits
    println!(
        "→ e.g. {b_w} Bit require up to → {} ← decimal digits",
        bit_to_dec_width(b_w)
    );

    b_w = 9;
    println!(
        "→ e.g. {b_w} Bit require up to → {} ← decimal digits",
        bit_to_dec_width(b_w)
    );

    b_w = 33;
    println!(
        "→ e.g. {b_w} Bit require up to → {} ← decimal digits",
        bit_to_dec_width(b_w)
    );
    b_w = 1024;
    println!(
        "→ e.g. {b_w} Bit require up to → {} ← decimal digits",
        bit_to_dec_width(b_w)
    );
}

mod mp_num_reader {
    use std::f64::consts::LOG10_2;

    ///
    /// Calculates amount of digits required for representing an `bit_width` binary
    /// number as decimal.
    ///
    pub fn bit_to_dec_width(bit_width: u64) -> u64 {
        // (bit_width as f64 / LOG2_10).ceil() as u64
        (bit_width as f64 * LOG10_2).ceil() as u64
    }

    ///
    /// Note: using bit_to_dec_width() we can determine in advance how many decimal digits
    /// a given bit-width number will require, so we *could* actually use fixed size arrays.
    ///
    pub fn parse_to_digits(num: &str) -> Vec<u8> {
        let mut digits: Vec<u8> = Vec::new();
        for b in num.bytes() {
            digits.push(ascii_to_digit(b).unwrap())
        }
        digits
    }

    ///
    /// Converts provided digit from its ascii representation to the actual decimal digit.
    ///
    fn ascii_to_digit(ch: u8) -> Option<u8> {
        match ch {
            b'0' => Some(0_u8),
            b'1' => Some(1_u8),
            b'2' => Some(2_u8),
            b'3' => Some(3_u8),
            b'4' => Some(4_u8),
            b'5' => Some(5_u8),
            b'6' => Some(6_u8),
            b'7' => Some(7_u8),
            b'8' => Some(8_u8),
            b'9' => Some(9_u8),
            _ => None,
        }
    }
}

const BIG_NUM_PI_INT_1000: &str = concat!(
    "31415926535897932384626433832795028841971693993751",
    "05820974944592307816406286208998628034825342117067",
    "98214808651328230664709384460955058223172535940812",
    "84811174502841027019385211055596446229489549303819",
    "64428810975665933446128475648233786783165271201909",
    "14564856692346034861045432664821339360726024914127",
    "37245870066063155881748815209209628292540917153643",
    "67892590360011330530548820466521384146951941511609",
    "43305727036575959195309218611738193261179310511854",
    "80744623799627495673518857527248912279381830119491",
    "29833673362440656643086021394946395224737190702179",
    "86094370277053921717629317675238467481846766940513",
    "20005681271452635608277857713427577896091736371787",
    "21468440901224953430146549585371050792279689258923",
    "54201995611212902196086403441815981362977477130996",
    "05187072113499999983729780499510597317328160963185",
    "95024459455346908302642522308253344685035261931188",
    "17101000313783875288658753320838142061717766914730",
    "35982534904287554687311595628638823537875937519577",
    "818577805321712268066130019278766111959092164201989"
);

const BIG_NUM_200_DIGITS: &str = concat!(
    "12345678901234567890123456789012345678901234567890", // 50 digits
    "12345678901234567890123456789012345678901234567890",
    "12345678901234567890123456789012345678901234567890",
    "12345678901234567890123456789012345678901234567890"
);
