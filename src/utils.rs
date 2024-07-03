use std::error::Error;
use std::fmt::Display;
use std::ops::{Div, Rem};

/// Basically a full adder for `u64`
///
/// # Explanation
/// - Adding `lhs` to `rhs` can _at most_ have a result of `2^64 + 2^64-1`.
///     - This is similar to "carry-bit + `u64::MAX-1`"
/// - If the `carry` input was provided, the total maximum sum will then at
/// most saturate the lower "bit"
/// - The implication is, that there is no way, that both additions will result
/// in a carry-bit simultaneously
/// - So it's safe to say, that in the worst-case the end result will be a
/// saturated "u64 value + carry-bit"
/// - Simplified example using a ficitonal u4:
///     ```
///     Inputs: lhs=rhs=1111, carry=1
///
///             1111          1110 (r1)
///           + 1111        + 0001 (input carry)
///     c1= 1 ← ¹¹¹     ==>
///         ––––––––        ––––––
///     r1=     1110          1111
///
///     ==> Endresult: (1111, c1)
///     ```
pub fn add_with_carry(lhs: u64, rhs: u64, mut carry: bool) -> (u64, bool) {
    let (mut sum, c) = lhs.overflowing_add(rhs);
    (sum, carry) = sum.overflowing_add(carry as u64);
    (sum, carry || c)
}

/// !untested
/// Simple Division with remainder, i.e. `a = q*b + r`, where `(q, r)` is the
/// returned result.
pub fn div_with_rem<T: Div<Output = T> + Rem<Output = T> + Copy>(a: T, b: T) -> (T, T) {
    // These two operations are optimized away into one assembly instruction
    (a / b, a % b)
}

///
/// Parses given decimal digits string into a vector of digits.
/// # Returns
/// - On Success: `Ok(Vec<u8>)` containing the parsed digits
/// - `Err` when a non-digit byte was encountered
///
pub fn parse_to_digits(num: &str) -> Result<Vec<u8>, ParseError> {
    if num.is_empty() {
        return Err("`num` must be non-empty".into());
    }
    let mut digits: Vec<u8> = Vec::new();
    for b in num.bytes() {
        match digit_char_to_value(b) {
            Some(value) => digits.push(value),
            None => return Err("invalid non-decimal digit encountered".into()),
        };
    }
    Ok(digits)
}

///
/// Converts provided digit from its ascii representation to the actual decimal digit.
///
fn digit_char_to_value(ch: u8) -> Option<u8> {
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

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    msg: &'static str,
}
impl Error for ParseError {}
impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error while parsing: {}", self.msg)
    }
}
impl From<&'static str> for ParseError {
    fn from(value: &'static str) -> Self {
        Self { msg: value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod test_add_with_carry {
        use super::*;
        const MAX: u64 = u64::MAX;
        const CARRY: bool = true;

        #[test]
        // TC1,2
        fn both_summands_zero() {
            assert_eq!(add_with_carry(0, 0, !CARRY), (0, false));
            assert_eq!(add_with_carry(0, 0, CARRY), (1, false));
        }

        #[test]
        // TC3,4
        fn both_summands_max() {
            assert_eq!(add_with_carry(MAX, MAX, !CARRY), (MAX - 1, true));
            assert_eq!(add_with_carry(MAX, MAX, CARRY), (MAX, true));
        }

        #[test]
        /// TC5,6,7,8
        fn one_summand_max_other_zero() {
            assert_eq!(add_with_carry(MAX, 0, !CARRY), (MAX, false));
            assert_eq!(add_with_carry(MAX, 0, CARRY), (0, true));
            assert_eq!(add_with_carry(0, MAX, !CARRY), (MAX, false));
            assert_eq!(add_with_carry(0, MAX, CARRY), (0, true));
        }

        #[test]
        // TC9,10
        fn single_summand_plus_carry() {
            assert_eq!(add_with_carry(MAX, 0, !CARRY), (MAX, false));
            assert_eq!(add_with_carry(MAX, 0, CARRY), (0, true));
            assert_eq!(add_with_carry(0, MAX, !CARRY), (MAX, false));
            assert_eq!(add_with_carry(0, MAX, CARRY), (0, true));
        }

        #[test]
        // TC11,12
        fn add_normal_values() {
            assert_eq!(add_with_carry(123, 456, !CARRY), (579, false));
            assert_eq!(add_with_carry(123, 456, CARRY), (580, false));
        }

        #[test]
        /// TC13,14
        fn one_summand_max_other_normal() {
            assert_eq!(add_with_carry(MAX, 42, !CARRY), (41, true));
            assert_eq!(add_with_carry(42, MAX, !CARRY), (41, true));
        }
    }

    mod test_digit_char_to_value {
        use super::*;
        #[test]
        fn valid_digits() {
            assert_eq!(digit_char_to_value(b'0'), Some(0));
            assert_eq!(digit_char_to_value(b'1'), Some(1));
            assert_eq!(digit_char_to_value(b'2'), Some(2));
            assert_eq!(digit_char_to_value(b'3'), Some(3));
            assert_eq!(digit_char_to_value(b'4'), Some(4));
            assert_eq!(digit_char_to_value(b'5'), Some(5));
            assert_eq!(digit_char_to_value(b'6'), Some(6));
            assert_eq!(digit_char_to_value(b'7'), Some(7));
            assert_eq!(digit_char_to_value(b'8'), Some(8));
            assert_eq!(digit_char_to_value(b'9'), Some(9));
        }

        #[test]
        fn unknown_digits() {
            assert_eq!(digit_char_to_value(b'/'), None);
            assert_eq!(digit_char_to_value(b':'), None);
            assert_eq!(digit_char_to_value(58), None); // 58 == b':'
            assert_eq!(digit_char_to_value(b'O'), None); // O vs. 0
            assert_eq!(digit_char_to_value(b'a'), None);
        }
    }

    mod test_parse_to_digits {
        use super::*;

        #[test]
        /// Test input containing invalid chars among valid ones
        fn invalid_chars() {
            let num_str = " 123- 4\n";
            assert!(parse_to_digits(num_str).is_err());
        }

        #[test]
        /// Test empty input string
        fn empty_str() {
            let num_str = "";
            assert!(parse_to_digits(num_str).is_err());
        }

        #[test]
        /// Test input where all chars are invalid
        fn all_chars_invalid() {
            let num_str = "foo";
            assert!(parse_to_digits(num_str).is_err());
        }

        #[test]
        /// Test long but valid decimal input strings
        fn large_nums() {
            const NUM_PI_INT_1001: &str = concat!(
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

            #[rustfmt::skip]
            let expected_200 = Ok(vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 
                1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 
                1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 
                1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0
            ]);

            #[rustfmt::skip]
            let expected_1001 = Ok(vec![
                3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 
                3, 8, 3, 2, 7, 9, 5, 0, 2, 8, 8, 4, 1, 9, 7, 1, 6, 9, 3, 9, 9, 3, 7, 5, 1, 
                0, 5, 8, 2, 0, 9, 7, 4, 9, 4, 4, 5, 9, 2, 3, 0, 7, 8, 1, 6, 4, 0, 6, 2, 8, 
                6, 2, 0, 8, 9, 9, 8, 6, 2, 8, 0, 3, 4, 8, 2, 5, 3, 4, 2, 1, 1, 7, 0, 6, 7, 
                9, 8, 2, 1, 4, 8, 0, 8, 6, 5, 1, 3, 2, 8, 2, 3, 0, 6, 6, 4, 7, 0, 9, 3, 8, 
                4, 4, 6, 0, 9, 5, 5, 0, 5, 8, 2, 2, 3, 1, 7, 2, 5, 3, 5, 9, 4, 0, 8, 1, 2, 
                8, 4, 8, 1, 1, 1, 7, 4, 5, 0, 2, 8, 4, 1, 0, 2, 7, 0, 1, 9, 3, 8, 5, 2, 1, 
                1, 0, 5, 5, 5, 9, 6, 4, 4, 6, 2, 2, 9, 4, 8, 9, 5, 4, 9, 3, 0, 3, 8, 1, 9, 
                6, 4, 4, 2, 8, 8, 1, 0, 9, 7, 5, 6, 6, 5, 9, 3, 3, 4, 4, 6, 1, 2, 8, 4, 7, 
                5, 6, 4, 8, 2, 3, 3, 7, 8, 6, 7, 8, 3, 1, 6, 5, 2, 7, 1, 2, 0, 1, 9, 0, 9, 
                1, 4, 5, 6, 4, 8, 5, 6, 6, 9, 2, 3, 4, 6, 0, 3, 4, 8, 6, 1, 0, 4, 5, 4, 3, 
                2, 6, 6, 4, 8, 2, 1, 3, 3, 9, 3, 6, 0, 7, 2, 6, 0, 2, 4, 9, 1, 4, 1, 2, 7, 
                3, 7, 2, 4, 5, 8, 7, 0, 0, 6, 6, 0, 6, 3, 1, 5, 5, 8, 8, 1, 7, 4, 8, 8, 1, 
                5, 2, 0, 9, 2, 0, 9, 6, 2, 8, 2, 9, 2, 5, 4, 0, 9, 1, 7, 1, 5, 3, 6, 4, 3, 
                6, 7, 8, 9, 2, 5, 9, 0, 3, 6, 0, 0, 1, 1, 3, 3, 0, 5, 3, 0, 5, 4, 8, 8, 2, 
                0, 4, 6, 6, 5, 2, 1, 3, 8, 4, 1, 4, 6, 9, 5, 1, 9, 4, 1, 5, 1, 1, 6, 0, 9, 
                4, 3, 3, 0, 5, 7, 2, 7, 0, 3, 6, 5, 7, 5, 9, 5, 9, 1, 9, 5, 3, 0, 9, 2, 1, 
                8, 6, 1, 1, 7, 3, 8, 1, 9, 3, 2, 6, 1, 1, 7, 9, 3, 1, 0, 5, 1, 1, 8, 5, 4, 
                8, 0, 7, 4, 4, 6, 2, 3, 7, 9, 9, 6, 2, 7, 4, 9, 5, 6, 7, 3, 5, 1, 8, 8, 5, 
                7, 5, 2, 7, 2, 4, 8, 9, 1, 2, 2, 7, 9, 3, 8, 1, 8, 3, 0, 1, 1, 9, 4, 9, 1, 
                2, 9, 8, 3, 3, 6, 7, 3, 3, 6, 2, 4, 4, 0, 6, 5, 6, 6, 4, 3, 0, 8, 6, 0, 2, 
                1, 3, 9, 4, 9, 4, 6, 3, 9, 5, 2, 2, 4, 7, 3, 7, 1, 9, 0, 7, 0, 2, 1, 7, 9, 
                8, 6, 0, 9, 4, 3, 7, 0, 2, 7, 7, 0, 5, 3, 9, 2, 1, 7, 1, 7, 6, 2, 9, 3, 1, 
                7, 6, 7, 5, 2, 3, 8, 4, 6, 7, 4, 8, 1, 8, 4, 6, 7, 6, 6, 9, 4, 0, 5, 1, 3, 
                2, 0, 0, 0, 5, 6, 8, 1, 2, 7, 1, 4, 5, 2, 6, 3, 5, 6, 0, 8, 2, 7, 7, 8, 5, 
                7, 7, 1, 3, 4, 2, 7, 5, 7, 7, 8, 9, 6, 0, 9, 1, 7, 3, 6, 3, 7, 1, 7, 8, 7, 
                2, 1, 4, 6, 8, 4, 4, 0, 9, 0, 1, 2, 2, 4, 9, 5, 3, 4, 3, 0, 1, 4, 6, 5, 4, 
                9, 5, 8, 5, 3, 7, 1, 0, 5, 0, 7, 9, 2, 2, 7, 9, 6, 8, 9, 2, 5, 8, 9, 2, 3, 
                5, 4, 2, 0, 1, 9, 9, 5, 6, 1, 1, 2, 1, 2, 9, 0, 2, 1, 9, 6, 0, 8, 6, 4, 0, 
                3, 4, 4, 1, 8, 1, 5, 9, 8, 1, 3, 6, 2, 9, 7, 7, 4, 7, 7, 1, 3, 0, 9, 9, 6, 
                0, 5, 1, 8, 7, 0, 7, 2, 1, 1, 3, 4, 9, 9, 9, 9, 9, 9, 8, 3, 7, 2, 9, 7, 8, 
                0, 4, 9, 9, 5, 1, 0, 5, 9, 7, 3, 1, 7, 3, 2, 8, 1, 6, 0, 9, 6, 3, 1, 8, 5, 
                9, 5, 0, 2, 4, 4, 5, 9, 4, 5, 5, 3, 4, 6, 9, 0, 8, 3, 0, 2, 6, 4, 2, 5, 2, 
                2, 3, 0, 8, 2, 5, 3, 3, 4, 4, 6, 8, 5, 0, 3, 5, 2, 6, 1, 9, 3, 1, 1, 8, 8, 
                1, 7, 1, 0, 1, 0, 0, 0, 3, 1, 3, 7, 8, 3, 8, 7, 5, 2, 8, 8, 6, 5, 8, 7, 5, 
                3, 3, 2, 0, 8, 3, 8, 1, 4, 2, 0, 6, 1, 7, 1, 7, 7, 6, 6, 9, 1, 4, 7, 3, 0, 
                3, 5, 9, 8, 2, 5, 3, 4, 9, 0, 4, 2, 8, 7, 5, 5, 4, 6, 8, 7, 3, 1, 1, 5, 9, 
                5, 6, 2, 8, 6, 3, 8, 8, 2, 3, 5, 3, 7, 8, 7, 5, 9, 3, 7, 5, 1, 9, 5, 7, 7, 
                8, 1, 8, 5, 7, 7, 8, 0, 5, 3, 2, 1, 7, 1, 2, 2, 6, 8, 0, 6, 6, 1, 3, 0, 0, 
                1, 9, 2, 7, 8, 7, 6, 6, 1, 1, 1, 9, 5, 9, 0, 9, 2, 1, 6, 4, 2, 0, 1, 9, 8, 9
            ]);

            assert_eq!(parse_to_digits(NUM_PI_INT_1001), expected_1001);
            assert_eq!(parse_to_digits(BIG_NUM_200_DIGITS), expected_200);
        }
    }
}
