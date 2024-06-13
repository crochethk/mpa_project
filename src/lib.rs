pub mod mp_uint {
    use crate::utils::parse_to_digits;
    use std::{fmt::Display, ops::ShlAssign, slice::Iter};

    /// Type of elements representing individual digits and number of Bits per digit
    /// of the internal number system. \
    /// __DO NOT CHANGE__
    type DigitT = u64;
    const DIGIT_BITS: u32 = 64;

    #[derive(Debug, Clone)]
    pub struct MPuint {
        width: usize,
        data: Vec<DigitT>,
    }

    /// Iterator for the digits
    impl<'a> IntoIterator for &'a MPuint {
        type Item = &'a DigitT;
        type IntoIter = Iter<'a, DigitT>;

        /// Iterator yields individual digits starting with __least significant__
        fn into_iter(self) -> Self::IntoIter {
            self.data.iter()
        }
    }

    impl MPuint {
        /// Alias for `into_iter()`
        pub fn iter(&self) -> Iter<DigitT> {
            self.data.iter()
        }
        /// Gets number of digits of the interal representation.
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    // // impl Div<Self> for &MPuint {
    // //     type Output = MPuint;
    // //     fn div(self, rhs: Self) -> Self::Output {
    // //         todo!()
    // //     }
    // // }

    // // impl Div<u64> for &MPuint {
    // //     type Output = u64;

    // //     fn div(self, divisor: u64) -> Self::Output {
    // //         // TODO: Take into account possibly differing operand widths
    // //         // TODO Probably simply extend the shorter to match the wider
    // //         // // a = q*b + r
    // //         // let mut result = 0;
    // //         // let mut rem_sum = 0;
    // //         // for i in (self.data.len() - 1)..=0 {
    // //         //     let dividend = self.data[i];
    // //         //     let divisor = rhs.data[i];

    // //         //     println!("i={}: {}", i, dividend);
    // //         //     println!("i={}: {}\n", i, divisor);

    // //         //     let (q, rem) = div_with_rem(dividend, divisor);
    // //         //     result += q;
    // //         //     rem_sum += rem;
    // //         // }
    // //         // result += rem_sum/
    // //         // result
    // //         42
    // //     }

    // //     // fn div(&self, rhs: &Self) -> Self::Output {
    // //     //     let a = self.data;
    // //     //     return 123;
    // //     // }
    // // }

    /// inplace `<<=` operator
    impl ShlAssign<u32> for MPuint {
        fn shl_assign(&mut self, mut shift_distance: u32) {
            assert!((shift_distance as usize) < self.width);
            const MAX_STEP: u32 = DIGIT_BITS - 1;

            let mut sh_step;
            while shift_distance > 0 {
                sh_step = shift_distance.min(MAX_STEP);

                let mut overflow = 0 as DigitT;
                for i in 0..self.len() {
                    let v = self.data[i];
                    let v_shl = v << sh_step;
                    let v_rtl = v.rotate_left(sh_step);

                    // Append last overflow
                    self.data[i] = v_shl | overflow;
                    // Update overflow
                    overflow = v_shl ^ v_rtl;
                }

                shift_distance -= sh_step;
            }

            // here we could panic! if `overflow != 0`, which would mean that
            // the number as a whole overflowed.
            // Actually we could do this check in advance by checking where the last `1`
            // is in the last bin and compare to `rhs` accordingly.
        }
    }

    /// !untested
    /// We want to allow comparing the actual values of different widths
    /// (i.e. not directly returning false, when the widths differ)
    /// Otherwise it would suffice to auto-derive
    impl PartialEq for MPuint {
        fn eq(&self, other: &Self) -> bool {
            // Figure out the wider instance
            let (big_num, small_num) = if self.width >= other.width {
                (self, other)
            } else {
                (other, self)
            };

            // Following code *should* automagically cover case of same data lengths...
            let bins_delta = big_num.len() - small_num.len();

            // Check whether the non-overlapping bins are populated with vals != 0
            // On bins_delta → takes no elements → false
            let excess_is_zero = big_num.data.iter().rev().take(bins_delta).all(|d| *d == 0);

            {
                excess_is_zero
                    // compare overlapping part, if non-overlapping part is zero
                    && big_num.data[0..(big_num.len() - bins_delta)] == small_num.data
            }
        }
    }

    impl MPuint {
        /// Creates a new instance with the desired bit-width and initialized
        /// to `0`.
        ///
        /// Actual bit-width will be a multiple of `DIGIT_BITS` and *at least* `width`.
        pub fn new(width: usize) -> Self {
            let bin_count = width.div_ceil(DIGIT_BITS as usize);
            let actual_width = bin_count * DIGIT_BITS as usize;
            Self {
                width: actual_width,
                data: vec![0; bin_count],
            }
        }

        #[allow(warnings)]
        pub fn from_str(num_str: &str, width: usize) -> Self {
            let digits: Vec<u8> = parse_to_digits(num_str);
            /*
            TODO add code, that computes the data vec elements based on the digits-vec:

            Scenarion: num_str = "1234" ↔ 1*10^3 + 2*10^2 + 3*10^1 + 4*10^0
            → digits = [1, 2, 3, 4]
                        ↑  ↑  ↑  ↑
                    i=  0  1  2  3
            → len(digits) = 4
            →  Calculate using Horner Schema:
                result : MPuint = 0;

                for each d in digits:
                    /* Do: result = result * 10 + d; */
                    result = (result << 3) + (result << 1); // == 2*2*2*x + 2*x == 10*x
                    result = result + d;

            → // TODO: implement Operators "+" and "<<" for MPuint

             */
            todo!()
        }

        /// Binary string, starting with MSB, ending with LSB on the right
        pub fn to_binary_string(&self) -> String {
            let mut result = String::new();

            for d in self.data.iter().rev() {
                result += &crate::bit_utils::int_to_binary_str(*d);
            }
            result
        }
    }

    impl Display for MPuint {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.to_binary_string())
        }
    }

    /// Implementations for often used integer values
    macro_rules! impl_common_val {
        ($value_name:ident as $val:literal) => {
            impl MPuint {
                #[doc = concat!("Creates new instance representing `", $val, "`.")]
                pub fn $value_name(width: usize) -> Self {
                    let mut result = Self::new(width);
                    result.data[0] = $val;
                    result
                }
            }
        };
    }

    impl_common_val!(one as 1);
    impl_common_val!(two as 2);
    impl_common_val!(ten as 10);
}

pub mod utils {
    use std::f64::consts::{LOG10_2, LOG2_10};
    use std::ops::{Div, Rem};

    /// Simple Division with remainder, i.e. `a = q*b + r`, where `(q, r)` is the
    /// returned result.
    pub fn div_with_rem<T: Div<Output = T> + Rem<Output = T> + Copy>(a: T, b: T) -> (T, T) {
        // These two operations are optimized away into one assembly instruction
        (a / b, a % b)
    }

    /// ! untested
    /// Calculates the least amount of decimal digits required to represent a
    /// `bit_width` binary number in base 10.
    ///
    pub fn bit_to_dec_width(bit_width: u64) -> u64 {
        (bit_width as f64 * LOG10_2).ceil() as u64
    }

    /// ! untested
    /// Calculates the least amount of bits required to represent a `dec_width`
    /// decimal number as binary integer.
    ///
    pub fn dec_to_bit_width(dec_width: u64) -> u64 {
        (dec_width as f64 * LOG2_10).ceil() as u64
    }

    ///
    /// Parses given decimal digits string into a vector of digits.
    /// Invalid chars are *ignored silently*.
    ///
    pub fn parse_to_digits(num: &str) -> Vec<u8> {
        let mut digits: Vec<u8> = Vec::new();
        for b in num.bytes() {
            match digit_char_to_value(b) {
                Some(value) => digits.push(value),
                None => continue,
            };
        }
        digits
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

    #[cfg(test)]
    mod tests {
        use super::*;

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
                let expected = vec![1, 2, 3, 4];
                assert_eq!(parse_to_digits(num_str), expected);
            }

            #[test]
            /// Test empty input string
            fn empty_str() {
                let num_str = "";
                let expected = vec![];
                assert_eq!(parse_to_digits(num_str), expected);
            }

            #[test]
            /// Test input where all chars are invalid
            fn all_chars_invalid() {
                let num_str = "foo";
                let expected = vec![];
                assert_eq!(parse_to_digits(num_str), expected);
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
                let expected_200 = vec![
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 
                    6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 
                    6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 
                    6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 
                    6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0
                ];

                #[rustfmt::skip]
                let expected_1001 = vec![
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
                ];

                assert_eq!(parse_to_digits(NUM_PI_INT_1001), expected_1001);
                assert_eq!(parse_to_digits(BIG_NUM_200_DIGITS), expected_200);
            }
        }
    }
}

pub mod bit_utils {
    use num_traits::{int::PrimInt, One};
    use std::ops::ShrAssign;

    pub fn int_to_binary_str<T: PrimInt + One + ShrAssign + 'static>(mut data: T) -> String {
        const BLOCK_WIDTH: usize = 4;
        const DELIM: &str = " ";

        let bits: usize = get_bit_width::<T>();

        let mut result = String::new();
        let _0001: T = T::one(); // bit mask

        for i in 0..bits {
            if (i != 0) && (i % BLOCK_WIDTH == 0) {
                result += DELIM;
            }
            let bit_is_1 = (data & _0001) == _0001;
            result += if bit_is_1 { "1" } else { "0" };
            data >>= T::one();
        }

        result.chars().rev().collect()
    }

    ///
    /// Returns the bit count of type `T`.
    /// NOTE: The implementation is only realiable for simple primitive types!
    /// (see: "std::mem::size_of" for more info)
    ///
    fn get_bit_width<T: 'static>() -> usize {
        std::mem::size_of::<T>() * 8
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        #[rustfmt::skip]
        fn test_int_to_binary_str() {
            assert_eq!(int_to_binary_str(-2 as i32),  "11111111111111111111111111111110");
            assert_eq!(int_to_binary_str(-1 as i32),  "11111111111111111111111111111111");
            assert_eq!(int_to_binary_str(0 as i32),   "00000000000000000000000000000000");
            assert_eq!(int_to_binary_str(1 as i32),   "00000000000000000000000000000001");
            assert_eq!(int_to_binary_str(2 as i32),   "00000000000000000000000000000010");
            assert_eq!(int_to_binary_str(3 as i32),   "00000000000000000000000000000011");
            assert_eq!(int_to_binary_str(4 as i32),   "00000000000000000000000000000100");
            assert_eq!(int_to_binary_str(127 as i32), "00000000000000000000000001111111");
            assert_eq!(int_to_binary_str(u32::MAX),   "11111111111111111111111111111111");
            assert_eq!(int_to_binary_str(2_147_483_648 as u32), "10000000000000000000000000000000");
        }
    }
}
