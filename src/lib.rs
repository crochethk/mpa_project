pub mod bit_utils;
pub mod utils;

pub mod mp_int {
    use crate::utils::{
        add_with_carry, dec_to_bit_width, div_with_rem, parse_to_digits, ParseError, TrimInPlace,
    };
    use std::{
        cmp::Ordering,
        fmt::Display,
        mem::size_of,
        ops::{Add, AddAssign, Div, Index, IndexMut, Neg, Not, Rem, ShlAssign},
        slice::Iter,
    };

    /// Type of elements representing individual digits. Directly related to the
    /// `const DIGIT_BITS`.
    /// __DO NOT CHANGE WITHOUT CAUTION__
    type DigitT = u64;

    /// Number of bits used per digit in the internal number system.
    /// Must stay ≤64, else e.g. division will break, since we need "2*DIGIT_BITS"
    /// for those calculations, while only ≤128bit are available "natively".
    const DIGIT_BITS: u32 = (size_of::<DigitT>() as u32) * 8;

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    enum Sign {
        Pos = 1,
        Neg = -1,
    }

    impl Sign {
        const PLUS: char = '+';
        const MINUS: char = '-';
    }

    impl Not for Sign {
        type Output = Self;
        fn not(self) -> Self::Output {
            match self {
                Self::Pos => Self::Neg,
                Self::Neg => Self::Pos,
            }
        }
    }

    impl TryFrom<char> for Sign {
        type Error = ParseError;

        /// Converts `char` to the appropriate enum value.
        /// Returns an error type, if the provided character is unknown.
        fn try_from(value: char) -> Result<Self, Self::Error> {
            match value {
                Self::PLUS => Ok(Self::Pos),
                Self::MINUS => Ok(Self::Neg),
                _ => Err("invalid sign character".into()),
            }
        }
    }

    impl From<Sign> for char {
        fn from(value: Sign) -> Self {
            match value {
                Sign::Pos => Sign::PLUS,
                Sign::Neg => Sign::MINUS,
            }
        }
    }

    impl Default for Sign {
        fn default() -> Self {
            Self::Pos
        }
    }

    #[derive(Debug, Clone)]
    pub struct MPint {
        width: usize,
        data: Vec<DigitT>,
        sign: Sign,
    }

    impl PartialEq for MPint {
        fn eq(&self, other: &Self) -> bool {
            if self.width != other.width {
                false
            } else if self.sign != other.sign {
                // different signs only eq when both are 0
                self.is_zero() && other.is_zero()
            } else {
                self.data == other.data
            }
        }
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
            let data = vec![0; bin_count];
            Self::new(data)
        }
    }

    impl CreateNewFrom<Vec<DigitT>> for MPint {
        /// Creates new instance by consuming the given `digits` Vec. Each element of
        /// `digits` is expected...
        /// - with respect to base `2^DIGIT_BITS` and
        /// - have __little endian__ order, i.e. least significant digit at index 0.
        ///
        /// The resulting width will be calculated automatically based on `digits.len()`.
        /// So in order to create numbers of same widths, provide Vecs of same lengths.
        ///
        /// ## Returns
        /// - New instance of `MPint`, containing the given digits and an appropriate width.
        ///
        fn new(digits: Vec<DigitT>) -> Self {
            let width = digits.len() * DIGIT_BITS as usize;
            Self {
                width,
                data: digits,
                sign: Sign::default(),
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

                result += d as DigitT; // result = result + (d as DigitT);
            }

            result.sign = sign;
            Ok(result)
        }

        /// Binary string, starting with MSB, ending with LSB on the right.
        /// Given a negative number, it will be prefixed with its sign.
        pub fn to_binary_string(&self) -> String {
            let mut result = String::new();
            if self.is_negative() {
                result.push(self.sign.into());
            }

            for d in self.iter().rev() {
                result += &crate::bit_utils::int_to_binary_str(*d);
                result += " ";
            }
            result
        }

        /// Alias for `into_iter()`
        pub fn iter(&self) -> Iter<DigitT> {
            self.into_iter()
        }

        pub fn is_negative(&self) -> bool {
            self.sign == Sign::Neg
        }

        /// Calculates the two's complement of the given number.
        /// Note that the result will have an _inverted sign_.
        fn twos_complement(&self) -> MPint {
            let mut twos_comp = !(self);
            twos_comp += 1;
            twos_comp
        }

        fn assert_same_width(&self, rhs: &MPint) {
            assert_eq!(self.width, rhs.width, "operands must have equal widths");
        }

        fn to_hex_string(&self) -> String {
            const X_WIDTH: usize = (DIGIT_BITS / 4) as usize;
            let mut hex: String = String::new();
            if self.is_negative() {
                hex.push(self.sign.into());
            }
            hex = self
                .iter()
                .rev()
                .fold(hex, |acc, d| acc + &format!("{:0width$X} ", d, width = X_WIDTH));
            hex.trim_end_in_place();
            hex
        }

        /// Helper function. Adds two number's bins with carry.
        /// Note that this **ignores sign**.
        fn carry_ripple_add_bins(&self, other: &MPint) -> (MPint, bool) {
            let mut sum = MPint::new(self);
            let mut carry = false;

            for i in 0..other.len() {
                let digit: DigitT;
                (digit, carry) = add_with_carry(self[i], other[i], carry);
                sum[i] = digit;
            }

            (sum, carry)
        }

        /// Compares the number's __absolute__ values (i.e. ignoring sign).
        /// ## Returns
        /// - `Ordering` enum value, representing the relation of `self` to `other`.
        /// - `None` when operands are incompatible.
        fn cmp_abs(&self, other: &MPint) -> Option<Ordering> {
            if self.width != other.width {
                return None;
            }

            let mut self_other_cmp = Ordering::Equal;
            // Compare bins/digits
            for (self_d, other_d) in self.iter().zip(other).rev() {
                if self_d == other_d {
                    continue;
                } else {
                    // On difference, assign matching relation and exit loop.
                    if self_d > other_d {
                        self_other_cmp = Ordering::Greater;
                    } else {
                        self_other_cmp = Ordering::Less;
                    }
                    break;
                }
            }

            Some(self_other_cmp)
        }

        fn is_zero(&self) -> bool {
            self.data.iter().all(|d| *d == 0)
        }
    }

    impl AddAssign<DigitT> for MPint {
        // inplace `+=` operator
        fn add_assign(&mut self, rhs: DigitT) {
            *self = &*self + &Self::from_digit(rhs, self.width);
        }
    }

    /// !untested
    impl Add for &MPint {
        type Output = MPint;

        fn add(self, rhs: Self) -> Self::Output {
            self.assert_same_width(rhs);

            let mut sum;
            let mut carry: bool = false;

            let same_sign = self.sign == rhs.sign;
            if !same_sign {
                // Order operands
                let (pos, neg) = if self.sign >= rhs.sign {
                    (self, rhs)
                } else {
                    (rhs, self)
                };

                let pos_lt_neg = pos.cmp_abs(&neg).unwrap() == Ordering::Less;

                let neg = neg.twos_complement();

                // Meaningful carry only ever possible with same signs.
                (sum, _) = pos.carry_ripple_add_bins(&neg);

                if pos_lt_neg {
                    sum = sum.twos_complement(); // sign is switched here
                }
            } else {
                // operands have same sign
                (sum, carry) = self.carry_ripple_add_bins(rhs);
                if self.is_negative() {
                    sum.sign = Sign::Neg;
                }
            }

            // TODO Reevaluate whether panic on overflow is desirable
            // Overflow can only ever occur, when both signs were equal, since
            // on unequal signs the worst-case is: `0 - MPint::max()` <=> `-MPint::max()`
            //
            // I.e.: `(same_sign && carry) => overflow`
            assert!(!carry, "MPint::Add resulted in overflow");
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

            // // here we could panic! if `overflow != 0`, which would mean that
            // // the number as a whole overflowed.
            // // Actually we could do this check in advance by checking where the last `1`
            // // is in the last bin and compare to `rhs` accordingly.
        }
    }

    impl Not for &MPint {
        type Output = MPint;

        /// Performs bitwise "not" aka. `!`.
        fn not(self) -> Self::Output {
            let mut result = MPint::new(self);
            for (i, d) in self.iter().enumerate() {
                result[i] = !d;
            }
            result
        }
    }

    /// Implements comparisson operators `<`, `<=`, `>`, and `>=`.
    impl PartialOrd for MPint {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            if self.width != other.width {
                return None;
            }

            // Treat possible +/-0 as equal
            if self.is_zero() && other.is_zero() {
                return Some(Ordering::Equal);
            }

            // Compare signs
            match self.sign.partial_cmp(&other.sign)? {
                Ordering::Greater => return Some(Ordering::Greater),
                Ordering::Less => return Some(Ordering::Less),
                _ => {} // signs equal
            }

            // Compare absolute values
            let mut self_other_cmp = self.cmp_abs(other)?;

            // Invert relation for negative numbers.
            assert!(self.sign == other.sign, "expected equal signs but were different");
            if self.is_negative() {
                self_other_cmp = match self_other_cmp {
                    Ordering::Greater => Ordering::Less,
                    Ordering::Less => Ordering::Greater,
                    Ordering::Equal => Ordering::Equal,
                }
            }

            Some(self_other_cmp)
        }
    }

    impl Neg for &MPint {
        type Output = MPint;

        fn neg(self) -> Self::Output {
            let mut result = self.clone();
            result.sign = !result.sign;
            result
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
            write!(f, "{}", self.to_hex_string())
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

    /// Shorthand macro for `MPint::new(vec![...])` that creates `MPint` from a
    /// list of digits, similar to `vec![1,2,3]`.
    #[macro_export]
    macro_rules! mpint {
        ($($d:expr),*) => {
            {
                let digits = vec![$($d,)*];
                MPint::new(digits)
            }
        };
    }

    #[allow(warnings)]
    #[cfg(test)]
    mod mpint_tests {
        use super::*;

        mod test_partial_cmp {
            use super::*;
            use Ordering::{Equal, Greater, Less};

            #[test]
            fn same_sign_positive() {
                // abs(a) > abs(b) and abs(b) < abs(a)
                {
                    let (a, b) = (mpint![1, 42], mpint![9, 1]);
                    assert_eq!(a.partial_cmp(&b), Some(Greater));
                    assert_eq!(b.partial_cmp(&a), Some(Less));
                }
                {
                    let (a, b) = (mpint![0, 99, 5, 17], mpint![99, 16, 5, 17]);
                    assert_eq!(a.partial_cmp(&b), Some(Greater));
                    assert_eq!(b.partial_cmp(&a), Some(Less));
                }
                {
                    let (a, b) = (mpint![4, 3, 42, 0], mpint![1, 3, 42, 0]);
                    assert_eq!(a.partial_cmp(&b), Some(Greater));
                    assert_eq!(b.partial_cmp(&a), Some(Less));
                }
                // abs(a) == abs(b)
                {
                    let (a, b) = (mpint![9, 16, 5, 17], mpint![9, 16, 5, 17]);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                }
                {
                    let (a, b) = (mpint![0, 42, 0, 0, 0, 0, 1], mpint![0, 42, 0, 0, 0, 0, 1]);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                }
            }

            #[test]
            fn same_sign_negative() {
                // abs(a) > abs(b) and abs(b) < abs(a)
                {
                    let (a, b) = (mpint![1, 42], mpint![9, 1]);
                    let (a, b) = (-&a, -&b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                    assert_eq!(b.partial_cmp(&a), Some(Greater));
                }
                {
                    let (a, b) = (mpint![0, 99, 5, 17], mpint![99, 16, 5, 17]);
                    let (a, b) = (-&a, -&b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                    assert_eq!(b.partial_cmp(&a), Some(Greater));
                }
                {
                    let (a, b) = (mpint![4, 3, 42, 0], mpint![1, 3, 42, 0]);
                    let (a, b) = (-&a, -&b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                    assert_eq!(b.partial_cmp(&a), Some(Greater));
                }
                // abs(a) == abs(b)
                {
                    let (a, b) = (mpint![9, 16, 5, 17], mpint![9, 16, 5, 17]);
                    let (a, b) = (-&a, -&b);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                }
                {
                    let (a, b) = (mpint![0, 42, 0, 0, 0, 0, 1], mpint![0, 42, 0, 0, 0, 0, 1]);
                    let (a, b) = (-&a, -&b);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                }
            }

            #[test]
            fn different_signs() {
                // abs(a) > abs(b)
                {
                    let (a, b) = (mpint![0, 0, 42], mpint![1, 2, 3]);
                    let (a, b) = (a, -&b);
                    assert_eq!(a.partial_cmp(&b), Some(Greater));
                }
                {
                    let (a, b) = (mpint![0, 0, 42], mpint![1, 2, 3]);
                    let (a, b) = (-&a, b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                }
                // abs(a) < abs(b)
                {
                    let (a, b) = (mpint![1, 2, 3], mpint![0, 0, 42]);
                    let (a, b) = (a, -&b);
                    assert_eq!(a.partial_cmp(&b), Some(Greater));
                }
                {
                    let (a, b) = (mpint![1, 2, 3], mpint![0, 0, 42]);
                    let (a, b) = (-&a, b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                }
                // abs(a) == abs(b)
                {
                    let (a, b) = (mpint![42, 42, 42, 42], mpint![42, 42, 42, 42]);
                    let (a, b) = (a, -&b);
                    assert_eq!(a.partial_cmp(&b), Some(Greater));
                }
                {
                    let (a, b) = (mpint![42, 42, 42, 42], mpint![42, 42, 42, 42]);
                    let (a, b) = (-&a, b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                }
            }

            #[test]
            fn zero_equality() {
                let (z_pos1, z_pos2) = (mpint![0, 0], mpint![0, 0]);
                let (z_neg1, z_neg2) = (-&z_pos1, -&z_pos2);

                assert_eq!(z_pos1.partial_cmp(&z_pos2), Some(Equal));
                assert_eq!(z_pos1.partial_cmp(&z_neg1), Some(Equal));
                assert_eq!(z_neg1.partial_cmp(&z_neg2), Some(Equal));
                assert_eq!(z_neg1.partial_cmp(&z_pos2), Some(Equal));
            }
        }
    }
}
