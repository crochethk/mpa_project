pub mod bit_utils;
pub mod utils;

pub mod mp_uint {
    use crate::utils::{add_with_carry, dec_to_bit_width, div_with_rem, parse_to_digits};
    use std::{
        error::Error,
        fmt::Display,
        ops::{Add, Div, Index, IndexMut, Rem, ShlAssign},
        slice::Iter,
    };

    /// Type of elements representing individual digits and number of Bits per digit
    /// of the internal number system. \
    /// __DO NOT CHANGE__
    type DigitT = u64;
    const DIGIT_BITS: u32 = 64;
    // Must stay ≤64, else e.g. division will break, since we need "2*DIGIT_BITS"
    // for those calculations while only ≤128bit are available "natively".

    #[derive(Debug, Clone, PartialEq)]
    pub struct MPuint {
        width: usize,
        data: Vec<DigitT>,
    }

    /// Indexing type
    type Idx = usize;
    impl IndexMut<Idx> for MPuint {
        /// Mutable access to digits (with base 2^DIGIT_BITS).
        fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
            &mut self.data[index]
        }
    }
    impl Index<Idx> for MPuint {
        type Output = DigitT;

        /// Immutable access to digits (with base 2^DIGIT_BITS).
        fn index(&self, index: Idx) -> &Self::Output {
            &self.data[index]
        }
    }

    /// Iterator for the digits (enables `for`-loop)
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
            self.into_iter()
        }

        /// !untested
        /// Calculates quotient and remainder using the given _native_ integer
        /// divisor.
        ///
        /// ### Footnote
        /// Since how division works, the remainder will always be `0 ≤ rem < divisor`.
        /// Therefore remainder and divisor can be represented by the same type.
        ///
        /// ### Algorithm Description
        /// - Division Term: `A = q*b + r`
        ///     - A = self
        ///     - b = divisor
        ///     - r = remainder
        ///     - q = quotient
        /// - to divide `self` by `divisor`:
        ///     - init `quotient` with appropriate width
        ///     - divide each d in self.data, in reverse order (starting with "MSB")
        ///         - calculate:
        ///         ```
        ///         base = 2^digit_bits
        ///         dividend = last_r*base + d // "puts last remainder infront of next digit"
        ///         q, last_r = div_with_rem(divident, divisor)
        ///         ```
        ///         - write q to quotient.data.reverse()[i]
        /// - last `last_r` is result remainder
        ///
        pub fn div_with_rem(&self, divisor: DigitT) -> (Self, DigitT) {
            let mut quotient = Self::new(self.width);

            let divisor = divisor as u128;
            let mut last_r = 0u128;
            // Start with most significant digit
            for (i, d) in self.iter().rev().enumerate() {
                let d = *d as u128;
                // "Prefix" d with last_r (multiplies last_r by `2^digit_bits`)
                let dividend = (last_r << DIGIT_BITS) + d;

                // Important: "0 ≤ r_i ≤ DigitT::MAX" and "0 ≤ q ≤ DigitT::MAX" ← unsure 'bout the latter
                let q;
                (q, last_r) = div_with_rem(dividend, divisor);

                // Write digit
                quotient[(self.len() - 1) - i] = q as u64;
            }

            (quotient, last_r as u64)
        }

        /// Gets max number of digits (in regards to the internal radix).
        pub fn len(&self) -> usize {
            self.data.len()
        }

    }

    /// !untested
    impl Add<DigitT> for MPuint {
        type Output = Self;

        fn add(self, rhs: DigitT) -> Self::Output {
            &self + &Self::from_digit(rhs as DigitT, self.width)
        }
    }

    /// !untested
    impl Add for &MPuint {
        type Output = MPuint;

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
    impl Div<DigitT> for &MPuint {
        type Output = MPuint;

        fn div(self, divisor: DigitT) -> Self::Output {
            self.div_with_rem(divisor).0
        }
    }

    /// ! untested
    /// `%` Operator for `DigitT` divisor
    impl Rem<DigitT> for &MPuint {
        type Output = DigitT;

        fn rem(self, divisor: DigitT) -> Self::Output {
            self.div_with_rem(divisor).1
        }
    }

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

    impl MPuint {
        /// Creates a new instance with the desired bit-width and initialized to `0`.
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

        pub fn from_digit(digit: DigitT, width: usize) -> Self {
            let mut num = Self::new(width);
            num[0] = digit;
            num
        }

        /// !untested
        /// Creates new number with _at least_ `width` bits (see `new()`) using the given
        /// `num_str`. Non-Decimal characters in `num_str` are ignored silently.
        /// Returns `Err(MPParseErr)` if width was too short.
        pub fn from_str(num_str: &str, width: usize) -> Result<Self, MPParseErr> {
            let digits: Vec<u8> = parse_to_digits(num_str);
            {
                let req_width = dec_to_bit_width(digits.len());
                if req_width > width {
                    return Err("speficfied bit width is too short for given number".into());
                }
            };

            /*
            TODO add code, that computes the data vec elements based on the digits-vec:

            Scenarion: num_str = "1234" ↔ 1*10^3 + 2*10^2 + 3*10^1 + 4*10^0
            → digits = [1, 2, 3, 4]       ↑        ↑        ↑        ↑
                        ↑  ↑  ↑  ↑
                    i=  0  1  2  3
            → len(digits) = 4
            →  Calculate using Horner Schema: ((((0)*10 + 1)*10+2)*10+3)*10+4
                result : MPuint = 0;                      ↑     ↑     ↑     ↑

                for each d in digits:
                    /* Do: result = result * 10 + d; */
                    result = (result << 3) + (result << 1); // == 2*2*2*x + 2*x == 10*x
                    result = result + d;

            → // TODO: implement Operators "+" and "<<" for MPuint
             */

            let mut result = Self::new(width);

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
    }

    // TODO change this to an actual decimal string
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

    #[derive(Debug, Clone)]
    pub struct MPParseErr {
        msg: &'static str,
    }
    impl Error for MPParseErr {}
    impl Display for MPParseErr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "error parsing to mp type: {}", self.msg)
        }
    }
    impl From<&'static str> for MPParseErr {
        fn from(value: &'static str) -> Self {
            Self { msg: value }
        }
    }

    impl_common_val!(one as 1);
    impl_common_val!(two as 2);
    impl_common_val!(ten as 10);
}
