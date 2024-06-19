pub mod bit_utils;
pub mod utils;

pub mod mp_int {
    use crate::utils::{
        add_with_carry, dec_to_bit_width, div_with_rem, parse_to_digits, ParseError,
    };
    use std::{
        fmt::Display,
        mem::size_of,
        ops::{Add, Div, Index, IndexMut, Rem, ShlAssign},
        slice::Iter,
    };

    /// Type of elements representing individual digits. Directly related to the
    /// `const DIGIT_BITS`.
    /// __DO NOT CHANGE WITHOUT CAUTION__
    type DigitT = u64;

    /// Number of bits used per digit in the internal number system.
    /// Must stay ≤64, else e.g. division will break, since we need "2*DIGIT_BITS"
    /// for those calculations, while only ≤128bit are available "natively".
    const DIGIT_BITS: u32 = size_of::<DigitT>() as u32;

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Sign {
        Pos,
        Neg,
    }

    impl TryFrom<char> for Sign {
        type Error = ParseError;

        /// Converts `char` to the appropriate enum value.
        /// Returns an error type, if the provided character is unknown.
        fn try_from(value: char) -> Result<Self, Self::Error> {
            match value {
                '+' => Ok(Self::Pos),
                '-' => Ok(Self::Neg),
                _ => Err("invalid sign character".into()),
            }
        }
    }

    impl Default for Sign {
        fn default() -> Self {
            Self::Pos
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct MPint {
        width: usize,
        data: Vec<DigitT>,
        sign: Sign,
    }

    pub trait CreateNewFrom<T> {
        /// Provides different constructor overloads
        fn new(src: T) -> Self;
    }

    impl CreateNewFrom<&Self> for MPint {
        /// Creates new instance of similar width. Shorthand for `new(src.width)`.
        fn new(src: &Self) -> Self {
            Self::new(src.width)
        }
    }

    impl CreateNewFrom<usize> for MPint {
        /// Creates a new instance with the desired bit-width and initialized to `0`.
        ///
        /// Actual bit-width will be a multiple of `DIGIT_BITS` and *at least* `width`.
        fn new(width: usize) -> Self {
            let bin_count = width.div_ceil(DIGIT_BITS as usize);
            let actual_width = bin_count * DIGIT_BITS as usize;
            Self {
                width: actual_width,
                data: vec![0; bin_count],
                sign: Sign::Pos,
            }
        }
    }

    impl MPint {
        /// !untested
        /// Calculates quotient and remainder using the given _native_ integer
        /// divisor.
        ///
        /// ### Footnote
        /// Since how division works, the remainder will always be `0 ≤ |rem| < divisor`.
        ///
        /// ### Algorithm Description
        /// - Division Term: `A = q*b + r`
        ///     - `A` = self
        ///     - `b` = divisor
        ///     - `r` = remainder
        ///     - `q` = quotient
        /// - To divide `self` by `divisor`:
        ///     - init `quotient` with same width
        ///     - divide each `d` in `self.data`, in reverse order (i.e. starting with "MSB")
        ///         - calculate:
        ///         ```
        ///         base = 2^digit_bits
        ///         dividend = last_r*base + d // "puts last remainder infront of next digit"
        ///         q, last_r = div_with_rem(divident, divisor)
        ///         ```
        ///         - write `q` to `quotient.data.reverse()[i]`
        /// - Last `last_r` is result remainder
        /// - When finished, adjust sign of quotient and remainder
        ///
        pub fn div_with_rem(&self, divisor: DigitT) -> (Self, i128) {
            let mut quotient = Self::new(self);

            let divisor = divisor as u128;
            let mut last_r = 0u128;
            // Start with most significant digit
            for (i, d) in self.iter().rev().enumerate() {
                let d = *d as u128;
                // "Prefix" d with last_r (multiplies last_r by `2^digit_bits`)
                let dividend = (last_r << DIGIT_BITS) + d;

                // TODO test this assumption
                // Important: "0 ≤ last_r ≤ DigitT::MAX" and "0 ≤ q ≤ DigitT::MAX" ← unsure 'bout the latter
                let q;
                (q, last_r) = div_with_rem(dividend, divisor);

                // Write digit
                quotient[(self.len() - 1) - i] = q as u64;
            }

            // Account for `self`s sign
            quotient.sign = self.sign;
            let last_r = match self.sign {
                Sign::Neg => -(last_r as i128),
                _ => last_r as i128,
            };

            (quotient, last_r)
        }

        /// Gets max number of digits (in regards to the internal radix).
        pub fn len(&self) -> usize {
            self.data.len()
        }

        pub fn from_digit(digit: DigitT, width: usize) -> Self {
            let mut num = Self::new(width);
            num[0] = digit;
            num
        }

        /// !untested
        /// Creates new number with _at least_ `width` bits (see `new()`) using the given
        /// decimal string `num_str`. First character may be a sign (`+`/`-`).
        ///
        /// ### Returns
        ///  - `Ok(Self)`: new MPint instance representing the number in `num_str`
        ///  - `Err(ParseError)` if:
        ///     - `width` was too short
        ///     - `num_str` was empty or contained invalid chars
        pub fn from_str(mut num_str: &str, width: usize) -> Result<Self, ParseError> {
            // Extract sign
            let first_char = match num_str.chars().next() {
                Some(ch) => ch,
                None => return Err("provided decimal string (`num_str`) must be non-empty".into()),
            };

            let sign: Sign = match first_char.try_into() {
                Ok(s) => {
                    // remove sign before processing digits
                    num_str = num_str.strip_prefix(first_char).unwrap();
                    s
                }
                _ => Sign::default(),
            };

            let digits: Vec<u8> = match parse_to_digits(num_str) {
                Ok(ds) => ds,
                Err(e) => return Err(e),
            };

            // Validate width
            {
                let req_width = dec_to_bit_width(digits.len());
                if req_width > width {
                    return Err("speficfied bit width is too short for the given number".into());
                }
            };

            let mut result = Self::new(width);

            // Build digits by applying Horner-Schema
            // e.g. digits = [1, 2, 3, 4]
            //     →  Calculate: ((((0)*10 + 1)*10+2)*10+3)*10+4
            //                               ↑     ↑     ↑     ↑
            for d in digits {
                let mut r1 = result.clone();
                let mut r2 = result.clone();

                // Multiply by 10:
                // (2*2*2*x + 2*x == 10*x)
                r1 <<= 3;
                r2 <<= 1;
                result = &r1 + &r2;

                result = result + (d as DigitT);
            }

            result.sign = sign;
            Ok(result)
        }

        /// Binary string, starting with MSB, ending with LSB on the right
        pub fn to_binary_string(&self) -> String {
            let mut result = String::new();

            for d in self.iter().rev() {
                result += &crate::bit_utils::int_to_binary_str(*d);
            }
            result
        }

        /// Alias for `into_iter()`
        pub fn iter(&self) -> Iter<DigitT> {
            self.into_iter()
        }
    }

    /// !untested
    impl Add<DigitT> for MPint {
        type Output = Self;

        fn add(self, rhs: DigitT) -> Self::Output {
            &self + &Self::from_digit(rhs as DigitT, self.width)
        }
    }

    /// !untested
    impl Add for &MPint {
        type Output = MPint;

        fn add(self, rhs: Self) -> Self::Output {
            assert_eq!(self.width, rhs.width, "operands must have equal widths");

            let mut sum = self.clone();
            let mut carry: bool = false;

            // Carry-Ripple add overlapping bins
            for i in 0..rhs.len() {
                let digit: DigitT;
                (digit, carry) = add_with_carry(self[i], rhs[i], carry);
                sum[i] = digit;
            }

            assert!(!carry, "Overlfow occured");

            sum
        }
    }

    /// ! untested
    /// `/` Operator for `DigitT` divisor
    impl Div<DigitT> for &MPint {
        type Output = MPint;

        fn div(self, divisor: DigitT) -> Self::Output {
            self.div_with_rem(divisor).0
        }
    }

    /// ! untested
    /// `%` Operator for `DigitT` divisor
    impl Rem<DigitT> for &MPint {
        type Output = i128;

        fn rem(self, divisor: DigitT) -> Self::Output {
            self.div_with_rem(divisor).1
        }
    }

    /// inplace `<<=` operator
    impl ShlAssign<u32> for MPint {
        fn shl_assign(&mut self, mut shift_distance: u32) {
            assert!((shift_distance as usize) < self.width);
            const MAX_STEP: u32 = DIGIT_BITS - 1;

            let mut sh_step;
            while shift_distance > 0 {
                sh_step = shift_distance.min(MAX_STEP);

                let mut overflow = 0 as DigitT;
                for i in 0..self.len() {
                    let v = self[i];
                    let v_shl = v << sh_step;
                    let v_rtl = v.rotate_left(sh_step);

                    // Append last overflow
                    self[i] = v_shl | overflow;
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

    /// Indexing type
    type Idx = usize;
    impl IndexMut<Idx> for MPint {
        /// Mutable access to digits (with base 2^DIGIT_BITS).
        fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
            &mut self.data[index]
        }
    }
    impl Index<Idx> for MPint {
        type Output = DigitT;

        /// Immutable access to digits (with base 2^DIGIT_BITS).
        fn index(&self, index: Idx) -> &Self::Output {
            &self.data[index]
        }
    }

    /// Iterator for the digits (enables `for`-loop)
    impl<'a> IntoIterator for &'a MPint {
        type Item = &'a DigitT;
        type IntoIter = Iter<'a, DigitT>;

        /// Iterator yields individual digits starting with __least significant__
        fn into_iter(self) -> Self::IntoIter {
            self.data.iter()
        }
    }

    // TODO change this to an actual decimal string
    impl Display for MPint {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.to_binary_string())
        }
    }

    /// Implementations for often used integer values
    macro_rules! impl_common_val {
        ($value_name:ident as $val:literal) => {
            impl MPint {
                #[doc = concat!("Creates new instance representing `", $val, "`.")]
                pub fn $value_name(width: usize) -> Self {
                    let mut result = Self::new(width);

                    let sign = if $val < 0 { Sign::Neg } else { Sign::Pos };
                    result.sign = sign;

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
