pub mod utils;

pub mod mp_int {
    //! # Multiple Precision Integer Type
    //!
    //! This module implements the data structure and various operations for
    //! arbitrary sized integers.
    //!

    // allows importing the macro more conveniently
    pub use crate::mpint;

    use crate::utils::*;
    use std::ops::{
        Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Not, Rem, ShlAssign, Sub, SubAssign,
    };
    use std::{cmp::Ordering, fmt::Display, mem::size_of, slice::Iter, str::from_utf8};

    /// Type of elements representing individual digits. Directly related to the
    /// `const DIGIT_BITS`.
    /// __DO NOT CHANGE WITHOUT CAUTION__
    pub type DigitT = u64;
    /// Type double the width of `DigitT`, sometimes required for intermediary results
    type DoubleDigitT = u128;

    /// Number of bits used per digit in the internal number system.
    /// Must stay ≤64, since e.g. multiplication depends on "2*DIGIT_BITS" width
    /// intermdiate results, while only ≤128bit are available "natively".
    const DIGIT_BITS: u32 = (size_of::<DigitT>() as u32) * 8;

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    enum Sign {
        Pos = 1,
        Neg = -1,
    }

    impl Sign {
        const PLUS: char = '+';
        const MINUS: char = '-';

        /// Extracts the sign from the given `num_str`. If first char is not a sign,
        /// `Sign::default()` is inferred.
        ///
        /// Note that `num_str` is not validated in any way.
        ///
        /// # Returns
        /// A tuple containing the sign and a slice based on `num_str`
        /// without the sign.
        fn extract_from_str(mut num_str: &str) -> (Sign, &str) {
            let first_char = match num_str.chars().next() {
                Some(ch) => ch,
                None => ' ',
            };
            let sign: Sign = match first_char.try_into() {
                Ok(s) => {
                    num_str = num_str.strip_prefix(first_char).unwrap();
                    s
                }
                _ => Sign::default(),
            };
            (sign, num_str)
        }
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

    /// The core datatype for multiple precision integers.
    #[derive(Debug, Clone)]
    pub struct MPint {
        data: Vec<DigitT>,
        sign: Sign,
    }

    pub trait CreateNewFrom<T> {
        /// Provides different constructor overloads
        fn new(src: T) -> Self;
    }

    impl CreateNewFrom<u128> for MPint {
        /// Creates new instance representing the given native integer.
        fn new(mut num: u128) -> Self {
            let mut res_data = vec![0; Self::width_to_bins_count(128 as usize)];

            for i in 0..res_data.len() {
                res_data[i] = num as DigitT;
                num >>= DIGIT_BITS;
            }

            Self::new(res_data)
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
        /// # Returns
        /// - New instance of `MPint`, containing the given digits and an appropriate width.
        ///
        fn new(digits: Vec<DigitT>) -> Self {
            Self {
                data: digits,
                sign: Sign::default(),
            }
        }
    }

    impl MPint {
        /// Creates a new instance with the desired bit-width and initialized to `0`.
        ///
        /// Actual bit-width will be a multiple of `DIGIT_BITS` and *at least* `width`.
        pub fn new_with_width(w: usize) -> Self {
            let bin_count = Self::width_to_bins_count(w);
            let data = vec![0; bin_count];
            Self::new(data)
        }

        /// Retrieves this number's current bit-width.
        pub fn width(&self) -> usize {
            self.data.len() * (DIGIT_BITS as usize)
        }

        pub fn width_to_bins_count(width: usize) -> usize {
            width.div_ceil(DIGIT_BITS as usize)
        }

        /// Calculates quotient and remainder using the given _native_ integer divisor.
        ///
        /// # Footnote
        /// Since how division works, the remainder will always be `0 ≤ |rem| < divisor`.
        ///
        /// # Algorithm Description
        /// - Division Term: `A = q*b + r`
        ///     - `A` = self
        ///     - `b` = divisor
        ///     - `r` = remainder
        ///     - `q` = quotient
        /// - To divide `self` by `divisor`:
        ///     - init `quotient` with same width
        ///     - divide each `d` in `self.data`, in reverse order (i.e. starting with "MSB")
        ///         - calculate:
        ///         ```plain
        ///         base = 2^digit_bits
        ///         dividend = last_r*base + d // "puts last remainder infront of next digit"
        ///         q, last_r = div_with_rem(divident, divisor)
        ///         ```
        ///         - write `q` to `quotient.data.reverse()[i]`
        /// - Last `last_r` is result remainder
        /// - When finished, adjust sign of quotient and remainder
        ///
        pub fn div_with_rem(&self, divisor: DigitT) -> (Self, i128) {
            let mut quotient = MPint::new_with_width(self.width());

            let divisor = divisor as DoubleDigitT;
            let mut last_r = 0 as DoubleDigitT;
            // Start with most significant digit
            for (i, d) in self.iter().rev().enumerate() {
                let d = *d as DoubleDigitT;
                // "Prefix" d with last_r (multiplies last_r by `2^digit_bits`)
                let dividend = (last_r << DIGIT_BITS) + d;

                // Important detail: "0 ≤ last_r ≤ DigitT::MAX" and "0 ≤ q ≤ DigitT::MAX" // TODO <----- review this assumption...
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

        /// Gets number of digits in regards to the internal radix.
        /// Note that this includes trailing "0".
        pub fn len(&self) -> usize {
            self.data.len()
        }

        /// Creates new `MPint` from the given decimal string `num_str`.
        /// Other than the first character, which may be a sign (`+`/`-`), only
        /// decimal digits are allowed inside `num_str`.
        ///
        /// # Returns
        ///  - `Ok(Self)`: new MPint instance representing the number in `num_str`
        ///  - `Err(ParseError)`: if `num_str` was empty or contained invalid chars
        pub fn from_dec_str(num_str: &str) -> Result<Self, ParseError> {
            if num_str.is_empty() {
                return Err("`num_str` must be non-empty".into());
            }
            let (sign, num_str) = Sign::extract_from_str(num_str);
            if num_str.is_empty() {
                return Err("`num_str` must contain at least one digit after the sign".into());
            }

            let decimals: Vec<u8> = match parse_to_digits(num_str) {
                Ok(ds) => ds,
                Err(e) => return Err(e),
            };

            // Build digits by applying Horner-Schema
            // e.g. digits = [1, 2, 3, 4]
            //     →  Calculate: ((((0)*10 + 1)*10+2)*10+3)*10+4
            //                               ↑     ↑     ↑     ↑
            let mut result = Self::new_with_width(approx_bit_width(decimals.len()));
            for d in decimals {
                let mut pt_res1 = result.clone();
                let mut pt_res2 = result.clone();

                // Multiply last result by 10:
                // (2*2*2*x + 2*x == 10*x)
                pt_res1 <<= 3;
                pt_res2 <<= 1;

                // result = pt_res1 + pt_res2;
                pt_res1 += pt_res2;
                result = pt_res1;

                result += d as DigitT
            }

            result.sign = sign;
            Ok(result)
        }

        /// Creates a MPint from a hexadecimal string number representation.
        /// `num_str` should match the  pattern `(+|-)?[0-9A-Fa-f]+`.
        /// Furthermore it shall start with the MSB and end with LSB.
        ///
        /// # Returns
        ///  - `Ok(Self)`: new MPint instance representing the number in `num_str`
        ///  - `Err(ParseError)`: if `num_str` was empty or contained invalid chars
        pub fn from_hex_str(num_str: &str) -> Result<Self, ParseError> {
            const BITS_PER_HEX: usize = 4;
            const HEX_PER_DIGIT_T: usize = DIGIT_BITS as usize / BITS_PER_HEX;

            if num_str.is_empty() {
                return Err("`given num_str` must be non-empty".into());
            }
            let (sign, num_str) = Sign::extract_from_str(num_str);
            if num_str.is_empty() {
                return Err("`num_str` must contain at least one digit after the sign".into());
            }

            // Remove leading '0'
            let num_str = num_str.trim_start_matches('0');

            let req_width = num_str.len() * BITS_PER_HEX as usize;
            let mut result = MPint::new_with_width(req_width);

            let num_str_bytes = num_str.as_bytes();
            let step = HEX_PER_DIGIT_T;
            let mut end = num_str_bytes.len();

            // Parse `num_str` in "16-hex-digit" bites, starting with its LSB
            for i in 0..result.len() {
                if end == 0 {
                    break;
                }
                let start = end.saturating_sub(step);
                let digit_as_hex_str = from_utf8(&num_str_bytes[start..end]).unwrap();

                // Parse hex-slice and write as internal digit
                result[i] = match DigitT::from_str_radix(digit_as_hex_str, 16) {
                    Ok(d) => d,
                    Err(_) => return Err("`num_str` must be a valid hexadecimal".into()),
                };

                end = start;
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

            const BIN_WIDTH: usize = DIGIT_BITS as usize;
            result = self
                .iter()
                .rev()
                .fold(result, |acc, d| acc + &format!("{:0width$b}", d, width = BIN_WIDTH));
            result
        }

        /// Alias for `into_iter()`. Returns an iterator over the underlying digits
        /// in _little-endian_ order.
        pub fn iter(&self) -> Iter<DigitT> {
            self.into_iter()
        }

        pub fn is_negative(&self) -> bool {
            self.sign == Sign::Neg
        }

        fn assert_same_width(&self, rhs: &MPint) {
            assert_eq!(self.width(), rhs.width(), "operands must have equal widths");
        }

        /// Returns the hexadecimal representation of this `MPint`.
        pub fn to_hex_string(&self) -> String {
            let mut hex: String = String::new();
            if self.is_negative() {
                hex.push(self.sign.into());
            }
            const HEX_WIDTH: usize = (DIGIT_BITS / 4) as usize;
            hex = self
                .iter()
                .rev()
                .fold(hex, |acc, d| acc + &format!("{:0width$X}", d, width = HEX_WIDTH));
            hex
        }

        /// Derives a decimal string representation of `self`, utilizing the
        /// _Division-Remainder Method_.
        pub fn to_dec_string(&self) -> String {
            // passing this as param should suffice to enable arbitrary output bases
            const BASE: u8 = 10;
            let mut quotient = self.clone();
            let mut digits: Vec<u8> = vec![];

            while quotient != 0 || digits.is_empty() {
                let rem;
                (quotient, rem) = quotient.div_with_rem(BASE as DigitT);
                digits.push(rem.abs() as u8);
            }
            digits.reverse();

            // init with sign if any
            let mut result = if self.is_negative() {
                String::from(Into::<char>::into(self.sign))
            } else {
                String::new()
            };

            for d in digits {
                result.push_str(&d.to_string());
            }
            result
        }

        /// Compares the number's __absolute__ values (i.e. ignoring sign).
        /// # Returns
        /// - `Ordering` enum value, representing the relation of `self` to `other`.
        fn cmp_abs(&self, other: &MPint) -> Ordering {
            // Sort by "width"
            let (wide, short) = if self.len() >= other.len() {
                (self, other)
            } else {
                (other, self)
            };
            let wide_is_self = std::ptr::eq(self, wide);

            // Check extra part in wider num
            for i in short.len()..wide.len() {
                if wide[i] != 0 {
                    // → wide > short
                    return if wide_is_self {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    };
                }
            }

            // Compare overlapping digits (starting with most significant)
            let mut self_other_ord = Ordering::Equal;

            for i in (0..short.len()).rev() {
                let (self_d, other_d) = (self[i], other[i]);
                if self_d != other_d {
                    // On difference, assign matching relation and exit loop.
                    if self_d > other_d {
                        self_other_ord = Ordering::Greater;
                    } else {
                        self_other_ord = Ordering::Less;
                    }
                    break;
                }
            }
            self_other_ord
        }

        fn is_zero(&self) -> bool {
            self.data.iter().all(|d| *d == 0)
        }

        /// Calculates the two's complement of the given number. This operation depends
        /// on the underlying width and, by definition, the result is mathematicaly
        /// compatible only with `MPint` of __same widths__.
        ///
        /// Note that the result will have an __inverted sign__.
        fn twos_complement_inplace(&mut self) {
            _ = self.not();
            let result_sign = !self.sign;

            // Add `1` avoiding auto-extend: Overflow is intended in certain cases
            let mut one = Self::new(vec![0; self.data.len()]);
            one[0] = 1;
            (*self).carry_ripple_add_bins_inplace(&one);

            self.sign = result_sign;
        }

        /// Helper function.
        /// Adds two number's bins and returns whether the most significant bin produced a carry.
        /// # Note
        /// - Both numbers shall have the __same width__.
        /// - This method effectively adds the __absolute values__, ignoring signs of the operands.
        /// - `self` __keeps its sign__, regardless of `other`'s sign.
        ///
        /// # Returns
        /// - A `bool` representing the "Carry-Out" of the adder.
        ///
        /// # Panics
        /// - When given numbers have unequal widths.
        fn carry_ripple_add_bins_inplace(&mut self, other: &MPint) -> bool {
            self.assert_same_width(other);

            let mut carry = false;
            for i in 0..other.len() {
                let digit: DigitT;
                (digit, carry) = add_with_carry(self[i], other[i], carry);
                self[i] = digit;
            }
            carry
        }

        /// In place adds `self` to `rhs`.\
        /// Returns a boolean indicating whether an arithmetic overflow occured.
        fn overflowing_add(&mut self, mut rhs: Self) -> bool {
            self.normalize_widths(&mut rhs);
            let rhs = &mut rhs;

            let mut carry: bool = false;

            let same_sign = self.sign == rhs.sign;
            if !same_sign {
                // Order operands
                let pos_is_self;
                let (pos, neg) = if self.sign >= rhs.sign {
                    pos_is_self = true;
                    (self, rhs)
                } else {
                    pos_is_self = false;
                    (rhs, self)
                };

                let pos_lt_neg = pos.cmp_abs(&neg) == Ordering::Less;

                neg.twos_complement_inplace();

                // Add `rhs` to `self`.
                // Meaningful carry only ever possible with same signs.
                let sum = if pos_is_self {
                    _ = pos.carry_ripple_add_bins_inplace(&neg);
                    pos
                } else {
                    _ = neg.carry_ripple_add_bins_inplace(&pos);
                    neg
                };

                if pos_lt_neg {
                    sum.twos_complement_inplace(); // sign is switched here
                }
            } else {
                // operands have same sign
                carry = self.carry_ripple_add_bins_inplace(&rhs);
            }

            // Overflow can only ever occur, when both signs were equal, since
            // on unequal signs the worst-case is: `0 - MPint::max()` <=> `-MPint::max()`
            carry
        }

        /// Normalizes the widths of two numbers, ensuring both have equal widths.
        /// In order to achieve this, the width of the _shorter_ number is adjusted
        /// to match the _wider_ number's width.
        ///
        /// If the widths are already equal, no changes are made.
        ///
        /// # Examples
        /// ```
        /// use mpa_lib::mp_int::*;
        /// let (mut a, mut b) = (mpint![1,2,3,0], mpint![4]);
        /// assert_ne!(a.width(), b.width());
        /// a.normalize_widths(&mut b);
        /// assert_eq!((a, b), (mpint![1,2,3,0], mpint![4,0,0,0]));
        ///
        /// let (mut a, mut b) = (mpint![4], mpint![1,2,3,0]);
        /// assert_ne!(a.width(), b.width());
        /// a.normalize_widths(&mut b);
        /// assert_eq!((a, b), (mpint![4,0,0,0], mpint![1,2,3,0]));
        ///
        /// let (mut a, mut b) = (mpint![0], mpint![0,0,0]);
        /// assert_ne!(a.width(), b.width());
        /// a.normalize_widths(&mut b);
        /// assert_eq!((a, b), (mpint![0,0,0], mpint![0,0,0]));
        /// ```
        pub fn normalize_widths(&mut self, other: &mut Self) {
            let (self_width, other_width) = (self.width(), other.width());

            match self_width.partial_cmp(&other_width).unwrap() {
                Ordering::Less => self.try_set_width(other_width).unwrap(),
                Ordering::Equal => (),
                Ordering::Greater => other.try_set_width(self_width).unwrap(),
            };
        }

        /// Tries to in-place extend or truncate `self` to the specified bit-width.
        /// Useful e.g. to normalize widths of two instances.
        ///
        /// # Returns
        /// - `Some(())` on success. As usual, the actual new width will be a multiple
        /// of `DIGITS_WIDTH`.
        /// - `None`, if non-zero bins would've been dropped to meet the specified
        /// `target_width`. In this case `self` is not modified.
        ///
        pub fn try_set_width(&mut self, target_width: usize) -> Option<()> {
            let new_bins_count = Self::width_to_bins_count(target_width);
            let old_bins_count = self.data.len();

            if new_bins_count == old_bins_count {
                return Some(());
            } else if new_bins_count > old_bins_count {
                let mut tail = vec![0; new_bins_count - old_bins_count];
                self.data.append(&mut tail);
            } else {
                // Check whether tail is empty
                for i in (new_bins_count..old_bins_count).rev() {
                    if self.data[i] != 0 {
                        return None;
                    }
                }
                self.data.truncate(new_bins_count);
            }
            Some(())
        }
    }

    impl Not for &mut MPint {
        type Output = Self;
        /// Inverts all bits, i.e. performs bitwise "not" (`!`).
        fn not(self) -> Self::Output {
            for i in 0..self.len() {
                let d = self[i];
                self[i] = !d;
            }
            self
        }
    }

    impl Neg for &MPint {
        type Output = MPint;
        /// Performs the unary - operation.
        fn neg(self) -> Self::Output {
            -(self.clone())
        }
    }
    impl Neg for MPint {
        type Output = MPint;
        // "Consuming" negation operation (`-self`)
        fn neg(mut self) -> Self::Output {
            self.sign = !self.sign;
            self
        }
    }

    impl Sub for MPint {
        type Output = Self;
        fn sub(mut self, rhs: Self) -> Self::Output {
            self -= rhs;
            self
        }
    }

    impl Sub for &MPint {
        type Output = MPint;
        fn sub(self, rhs: Self) -> Self::Output {
            let mut diff = self.clone();
            diff -= rhs.clone();
            diff
        }
    }

    impl SubAssign for MPint {
        fn sub_assign(&mut self, rhs: Self) {
            *self += -rhs;
        }
    }

    impl Add for MPint {
        type Output = Self;
        /// Consuming, binary `+` operator.
        /// Uses `MPint::add_assign(&mut self, rhs: Self)` internally.
        fn add(mut self, rhs: Self) -> Self::Output {
            self += rhs;
            self
        }
    }

    impl Add for &MPint {
        type Output = MPint;
        fn add(self, rhs: Self) -> Self::Output {
            let mut sum = self.clone();
            sum += rhs.clone();
            sum
        }
    }

    impl AddAssign<DigitT> for MPint {
        /// Inplace `+=` operator for `DigitT` right-hand side.
        fn add_assign(&mut self, rhs: DigitT) {
            *self += mpint![rhs];
        }
    }

    impl AddAssign<DoubleDigitT> for MPint {
        /// Inplace `+=` operator for `DoubleDigitT` right-hand side.
        fn add_assign(&mut self, rhs: DoubleDigitT) {
            let mp_rhs = MPint::new(rhs);
            *self += mp_rhs;
        }
    }

    impl AddAssign for MPint {
        /// Performs the += operation. Arithmetic overflows are silently ignored.
        fn add_assign(&mut self, rhs: Self) {
            let overflow = self.overflowing_add(rhs);
            if overflow {
                self.data.push(1);
            }
        }
    }

    impl Mul for &MPint {
        type Output = MPint;

        /// Performs the `*` operation, potentially extending the bit-width. Therefore,
        /// the result's width `w` will be in the range `self.width() ≤ w ≤ 2*self.width()`.
        fn mul(self, rhs: Self) -> Self::Output {
            self.extending_prod_scan_mul(rhs)
        }
    }

    impl MPint {
        /// Multiplies two `MPint`, extending the width as necessary.
        /// This uses a "Product-Scanning" approach, which means we directly
        /// calculate each result digit one at a time.
        fn extending_prod_scan_mul(&self, rhs: &Self) -> Self {
            // ~~~~ Preamble ~~~~
            self.assert_same_width(rhs);

            // Zero short circuit
            if self.is_zero() {
                return self.clone();
            } else if rhs.is_zero() {
                return rhs.clone();
            }

            let result_sign = if self.sign == rhs.sign {
                Sign::Pos
            } else {
                Sign::Neg
            };

            // ~~~~ Main ~~~~
            let operand_len = self.len();
            let max_new_width = self.width() * 2;

            let mut end_product = MPint::new_with_width(max_new_width);

            // Calculate one result digit per iteration
            for i in 0..end_product.len() {
                // Sum partial products and propagate carries
                for j in 0..operand_len {
                    if j > i {
                        // Happens in LSB-half of the end product.
                        // k would get neg.
                        break;
                    }

                    let k: usize = i - j;
                    if k >= operand_len {
                        // Possible, when `i >= end_product.len() / 2`
                        // i.e. in MSB-half of the end product
                        continue;
                    }
                    let a_j = self[j] as DoubleDigitT;
                    let b_k = rhs[k] as DoubleDigitT;

                    let prod_i_jk: DoubleDigitT = a_j * b_k;

                    // Implicit carry propagation
                    end_product += prod_i_jk;
                }

                // Prepare for easier partial product addition in next iteration.
                // After last iteration, end_product has performed a full cycle.
                end_product.rotate_digits_right(1);
            }

            // ~~~~ Epilog ~~~~
            end_product.trim_empty_end(self.len());
            end_product.sign = result_sign;

            end_product
        }

        fn rotate_digits_right(&mut self, n: usize) {
            self.data.rotate_left(n);
        }

        /// Removes empty (i.e. `0`) bins from the end
        fn trim_empty_end(&mut self, min_len: usize) {
            // Get first non-zero digit index from the end
            let mut first_non_zero = 0;
            for (i, d) in self.iter().enumerate().rev() {
                if *d != 0 {
                    first_non_zero = i;
                    break;
                }
            }
            self.data.truncate((first_non_zero + 1).max(min_len));
        }
    }

    /// `/` Operator for `DigitT` divisor
    impl Div<DigitT> for &MPint {
        type Output = MPint;

        fn div(self, divisor: DigitT) -> Self::Output {
            self.div_with_rem(divisor).0
        }
    }

    /// `%` Operator for `DigitT` divisor
    impl Rem<DigitT> for &MPint {
        type Output = i128;

        fn rem(self, divisor: DigitT) -> Self::Output {
            self.div_with_rem(divisor).1
        }
    }

    /// inplace `<<=` operator
    impl ShlAssign<u32> for MPint {
        /// Performs the <<= operation.
        ///
        /// # Panics
        /// Alike native integer implementations, when trying to shift more than
        /// the current bit-width.
        fn shl_assign(&mut self, mut shift_distance: u32) {
            assert!((shift_distance as usize) < self.width());
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
        }
    }

    impl PartialEq<DigitT> for MPint {
        fn eq(&self, digit: &DigitT) -> bool {
            self.partial_cmp(digit) == Some(Ordering::Equal)
        }
    }

    impl PartialEq for MPint {
        fn eq(&self, other: &Self) -> bool {
            self.partial_cmp(other) == Some(Ordering::Equal)
        }
    }

    impl PartialOrd<DigitT> for MPint {
        fn partial_cmp(&self, digit: &DigitT) -> Option<Ordering> {
            self.partial_cmp(&mpint![*digit])
        }
    }

    /// Implements comparisson operators `<`, `<=`, `>`, and `>=`.
    /// `==` is implemented with `PartialEq` trait.
    impl PartialOrd for MPint {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
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
            let mut self_other_cmp = self.cmp_abs(other);

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

    /// Shorthand macro for `MPint::new(vec![...])` that creates `MPint` from a
    /// list of digits, similar to `vec![1,2,3]`.
    /// Digits are expected to start with the least significant.
    #[macro_export]
    macro_rules! mpint {
        ($($d:expr),*) => {
            {
                let digits = vec![$($d,)*];
                MPint::new(digits)
            }
        };
    }

    #[cfg(test)]
    mod mpint_tests {
        use pyo3::prelude::*;
        use pyo3::types::PyList;

        use super::*;

        const D_MAX: DigitT = DigitT::MAX;

        /// Generates a function that tests correctness of the specified arithmetic
        /// operation on `MPint` objects.
        /// # Rules
        /// - `create_op_correctness_tester($fn_name, $op)`
        ///     - `$fn_name` - The created function's name.
        ///     - `$op` - An operator token. Currently supports: `+`, `-`, `*`.
        /// # Examples
        /// ```rust
        /// create_op_correctness_tester!(test_addition_correctness, +);
        /// ```
        macro_rules! create_op_correctness_tester {
            ($fn_name:ident, $op:tt) => {
                fn $fn_name(a: MPint, b: MPint) {
                    let result = &a $op &b;
                    let test_result = verify_arithmetic_result(
                        &a, stringify!($op), &b, &result);
                    println!("{:?}", test_result);
                    assert!(test_result.0, "{}", test_result.1);
                }
            };
        }

        mod test_try_set_width {
            use super::*;

            #[test]
            fn no_change() {
                let num = mpint![1, 2, 3];
                let expect = (Some(()), num.clone());

                let mut test_num = num.clone();
                let res = test_num.try_set_width(192);
                assert_eq!((res, test_num), expect.clone());

                let mut test_num = num.clone();
                let res = test_num.try_set_width(177);
                assert_eq!((res, test_num), expect.clone());

                let mut test_num = num.clone();
                let res = test_num.try_set_width(129);
                assert_eq!((res, test_num), expect.clone());
            }

            #[test]
            fn trim_fail_1() {
                let num = mpint![1, 2, 3];
                let expect = (None, num.clone());

                let mut num_res = num.clone();
                let res = num_res.try_set_width(128);
                assert_eq!((res, num_res), expect.clone());

                let mut num_res = num.clone();
                let res = num_res.try_set_width(96);
                assert_eq!((res, num_res), expect.clone());
            }

            #[test]
            fn trim_fail_2() {
                let num = mpint![1, 2, 3, 0];
                let expect = (None, num.clone());

                let mut num_res = num.clone();
                let res = num_res.try_set_width(128);
                assert_eq!((res, num_res), expect.clone());

                let mut num_res = num.clone();
                let res = num_res.try_set_width(96);
                assert_eq!((res, num_res), expect.clone());
            }

            #[test]
            fn truncate_1() {
                let num = mpint![1, 2, 3, 0];
                let expect = (Some(()), mpint![1, 2, 3]);

                let mut num_res = num.clone();
                let res = num_res.try_set_width(192);
                assert_eq!((res, num_res), expect.clone());

                let mut num_res = num.clone();
                let res = num_res.try_set_width(129);
                assert_eq!((res, num_res), expect.clone());
            }

            #[test]
            fn truncate_2() {
                let num = mpint![1, 2, 3, 0, 0, 0];
                let expect = (Some(()), mpint![1, 2, 3, 0]);

                let mut num_res = num.clone();
                let res = num_res.try_set_width(256);
                assert_eq!((res, num_res), expect.clone());

                let mut num_res = num.clone();
                let res = num_res.try_set_width(222);
                assert_eq!((res, num_res), expect.clone());

                let mut num_res = num.clone();
                let res = num_res.try_set_width(193);
                assert_eq!((res, num_res), expect.clone());
            }

            #[test]
            fn extend_1() {
                let num = mpint![1, 2];
                let expect = (Some(()), mpint![1, 2, 0]);

                let mut num_res = num.clone();
                let res = num_res.try_set_width(129);
                assert_eq!((res, num_res), expect);
            }

            #[test]
            fn extend_2() {
                let num = mpint![1, 2, 0];
                let mut exp_num = MPint::new(vec![0; 64]);
                exp_num[0] = num[0];
                exp_num[1] = num[1];
                exp_num[2] = num[2];
                let expect = (Some(()), exp_num);

                let mut num_res = num.clone();
                let res = num_res.try_set_width(4096);
                assert_eq!((res, num_res), expect);
            }
        }

        mod test_to_hex_string {
            use super::*;
            #[test]
            fn positive_values() {
                {
                    let a = mpint![0, D_MAX, 2, 3];
                    let expected = concat!(
                        "0000000000000003",
                        "0000000000000002",
                        "FFFFFFFFFFFFFFFF",
                        "0000000000000000"
                    );
                    assert_eq!(a.to_hex_string(), expected);
                }
                {
                    let a = mpint![42, 1 << 13, (1 as DigitT).rotate_right(1)];
                    let expected =
                        concat!("8000000000000000", "0000000000002000", "000000000000002A",);
                    assert_eq!(a.to_hex_string(), expected);
                }
                {
                    let a = mpint![D_MAX, D_MAX, D_MAX];
                    let expected =
                        concat!("FFFFFFFFFFFFFFFF", "FFFFFFFFFFFFFFFF", "FFFFFFFFFFFFFFFF",);
                    assert_eq!(a.to_hex_string(), expected);
                }
            }

            #[test]
            fn negative_values() {
                {
                    let a = -mpint![0, D_MAX, 2, 3];
                    let expected = concat!(
                        "-",
                        "0000000000000003",
                        "0000000000000002",
                        "FFFFFFFFFFFFFFFF",
                        "0000000000000000"
                    );
                    assert_eq!(a.to_hex_string(), expected);
                }
                {
                    let a = -mpint![42, 1 << 13, (1 as DigitT).rotate_right(1)];
                    let expected =
                        concat!("-", "8000000000000000", "0000000000002000", "000000000000002A",);
                    assert_eq!(a.to_hex_string(), expected);
                }
                {
                    let a = -mpint![D_MAX, D_MAX, D_MAX];
                    let expected =
                        concat!("-", "FFFFFFFFFFFFFFFF", "FFFFFFFFFFFFFFFF", "FFFFFFFFFFFFFFFF",);
                    assert_eq!(a.to_hex_string(), expected);
                }
            }
        }

        /// Tests the different comparisson operators of `MPint`.
        /// `PartialEq` ("==") is tested implicitly tested here, since its impl
        /// simply passes calls to `partial_cmp`.
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
                    let (a, b) = (-a, -b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                    assert_eq!(b.partial_cmp(&a), Some(Greater));
                }
                {
                    let (a, b) = (mpint![0, 99, 5, 17], mpint![99, 16, 5, 17]);
                    let (a, b) = (-a, -b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                    assert_eq!(b.partial_cmp(&a), Some(Greater));
                }
                {
                    let (a, b) = (mpint![4, 3, 42, 0], mpint![1, 3, 42, 0]);
                    let (a, b) = (-a, -b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                    assert_eq!(b.partial_cmp(&a), Some(Greater));
                }
                // abs(a) == abs(b)
                {
                    let (a, b) = (mpint![9, 16, 5, 17], mpint![9, 16, 5, 17]);
                    let (a, b) = (-a, -b);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                }
                {
                    let (a, b) = (mpint![0, 42, 0, 0, 0, 0, 1], mpint![0, 42, 0, 0, 0, 0, 1]);
                    let (a, b) = (-a, -b);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                }
            }

            #[test]
            fn different_signs() {
                // abs(a) > abs(b)
                {
                    let (a, b) = (mpint![0, 0, 42], mpint![1, 2, 3]);
                    let (a, b) = (a, -b);
                    assert_eq!(a.partial_cmp(&b), Some(Greater));
                }
                {
                    let (a, b) = (mpint![0, 0, 42], mpint![1, 2, 3]);
                    let (a, b) = (-a, b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                }
                // abs(a) < abs(b)
                {
                    let (a, b) = (mpint![1, 2, 3], mpint![0, 0, 42]);
                    let (a, b) = (a, -b);
                    assert_eq!(a.partial_cmp(&b), Some(Greater));
                }
                {
                    let (a, b) = (mpint![1, 2, 3], mpint![0, 0, 42]);
                    let (a, b) = (-a, b);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                }
                // abs(a) == abs(b)
                {
                    let (a, b) = (mpint![42, 42, 42, 42], mpint![42, 42, 42, 42]);
                    let (a, b) = (a, -b);
                    assert_eq!(a.partial_cmp(&b), Some(Greater));
                }
                {
                    let (a, b) = (mpint![42, 42, 42, 42], mpint![42, 42, 42, 42]);
                    let (a, b) = (-a, b);
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

            #[test]
            fn diff_widths_equality() {
                {
                    // explicitly tests "PartialEq"
                    let (a, b) = (mpint![1, 2, 3], mpint![1, 2, 3, 0, 0, 0]);
                    assert!(a == b);
                    assert!(b == a);
                }
                {
                    let (a, b) = (mpint![1, 2, 3], mpint![1, 2, 3, 0, 0, 0]);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                    assert_eq!(b.partial_cmp(&a), Some(Equal));
                }
                {
                    let (a, b) = (mpint![0, 1, 2, 3], mpint![0, 1, 2, 3, 0, 0, 0]);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                    assert_eq!(b.partial_cmp(&a), Some(Equal));
                }
                {
                    let (a, b) = (mpint![1, 2, 3, 0], mpint![1, 2, 3, 0, 0, 0]);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                    assert_eq!(b.partial_cmp(&a), Some(Equal));
                }
                {
                    let (a, b) = (mpint![0, 1, 2, 3, 0], mpint![0, 1, 2, 3, 0, 0, 0]);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                    assert_eq!(b.partial_cmp(&a), Some(Equal));
                }
                {
                    let (a, b) = (-mpint![0, 1, 2, 3, 0], -mpint![0, 1, 2, 3, 0, 0, 0]);
                    assert_eq!(a.partial_cmp(&b), Some(Equal));
                    assert_eq!(b.partial_cmp(&a), Some(Equal));
                }
            }

            #[test]
            fn diff_widths_lt_gt() {
                {
                    let (a, b) = (mpint![1, 2, 3], mpint![1, 2, 3, 4]);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                    assert_eq!(b.partial_cmp(&a), Some(Greater));
                }
                {
                    let (a, b) = (mpint![0, 1, 2, 3, 0], mpint![0, 1, 2, 3, 0, 0, 1]);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                    assert_eq!(b.partial_cmp(&a), Some(Greater));
                }
                {
                    let (a, b) = (mpint![1, 2, 3], mpint![0, 0, 0, 3, 2, 1]);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                    assert_eq!(b.partial_cmp(&a), Some(Greater));
                }
                {
                    let (a, b) = (mpint![1, 2, 3], mpint![0, 1, 2, 3]);
                    assert_eq!(a.partial_cmp(&b), Some(Less));
                    assert_eq!(b.partial_cmp(&a), Some(Greater));
                }
            }

            #[test]
            fn cmp_single_digit() {
                let a = mpint![123, 0];
                let b = mpint![12, 0];
                let c = mpint![42, 0];
                let d = mpint![0, 42];
                let digit: DigitT = 42;

                assert_eq!(a.partial_cmp(&digit), Some(Greater));
                assert_eq!((-a).partial_cmp(&digit), Some(Less));
                assert_eq!(b.partial_cmp(&digit), Some(Less));
                assert_eq!((-b).partial_cmp(&digit), Some(Less));
                assert_eq!(c.partial_cmp(&digit), Some(Equal));
                assert_eq!(d.partial_cmp(&digit), Some(Greater));
            }
        }

        mod test_mul {
            use super::*;
            create_op_correctness_tester!(test_mul_correctness, *);

            #[test]
            fn both_factors_pos_1() {
                let a = mpint![9, 8, 7];
                let b = mpint![100, 10, 1];
                test_mul_correctness(a, b);
            }

            #[test]
            fn both_factors_pos_2() {
                let a = mpint![0, 9, 8, 7, 6, 0];
                let b = mpint![100, 10, 1, 0, 0, 0];
                test_mul_correctness(a, b);
            }

            #[test]
            fn both_factors_neg_1() {
                let a = -mpint![9, 8, 7];
                let b = -mpint![100, 10, 1];
                test_mul_correctness(a, b);
            }

            #[test]
            fn both_factors_neg_2() {
                let a = -mpint![0, 9, 8, 7, 6, 0];
                let b = -mpint![100, 10, 1, 0, 0, 0];
                test_mul_correctness(a, b);
            }

            #[test]
            fn pos_neg_factors_1() {
                let a = -mpint![9, 8, 7];
                let b = mpint![100, 10, 1];
                test_mul_correctness(a, b);
                let a = mpint![9, 8, 7];
                let b = -mpint![100, 10, 1];
                test_mul_correctness(a, b);
            }

            #[test]
            fn pos_neg_factors_2() {
                let a = -mpint![0, 9, 8, 7, 6, 0];
                let b = mpint![100, 10, 1, 0, 0, 0];
                test_mul_correctness(a, b);
                let a = mpint![0, 9, 8, 7, 6, 0];
                let b = -mpint![100, 10, 1, 0, 0, 0];
                test_mul_correctness(a, b);
            }

            #[test]
            fn zero_factor() {
                let b = mpint![100, 10, 1, 0, 0, 0];
                let a = mpint![0, 0, 0, 0, 0, 0];
                test_mul_correctness(a, b);
                let a = mpint![0, 9, 8, 7, 6, 0];
                let b = mpint![0, 0, 0, 0, 0, 0];
                test_mul_correctness(a, b);
            }

            #[test]
            fn one_factor() {
                let b = mpint![100, 10, 1, 0, 0, 0];
                let a = mpint![1, 0, 0, 0, 0, 0];
                test_mul_correctness(a.clone(), b.clone());

                assert_eq!(
                    a.clone().mul(&b),
                    b.clone(),
                    "Result should exactly match 'not-one' operand"
                );
                assert_eq!(
                    b.clone().mul(&a),
                    b.clone(),
                    "Result should exactly match 'not-one' operand"
                );

                let c = mpint![0, 9, 8, 7, 6, 0];
                let d = mpint![1, 0, 0, 0, 0, 0];
                test_mul_correctness(c.clone(), d.clone());
                test_mul_correctness(-a, b);
                test_mul_correctness(c, -d);
            }

            #[test]
            fn large_factors_1() {
                let a = MPint::new(vec![D_MAX; 32]);
                let b = a.clone();
                test_mul_correctness(a, b);
            }

            #[test]
            fn large_factors_2() {
                let a = MPint::new(<Vec<u64>>::from(LARGE_NUM_1));
                let b = MPint::new(<Vec<u64>>::from(LARGE_NUM_2));
                test_mul_correctness(a.clone(), b.clone());
            }
        }

        mod test_add {
            use super::*;

            create_op_correctness_tester!(test_addition_correctness, +);

            mod same_signs {
                use super::*;
                #[test]
                fn same_signs_normal_case() {
                    // a>b
                    let a = mpint![0, 0, 42, 1];
                    let b = mpint![42, 42, 42, 0];
                    test_addition_correctness(-&a, -&b);
                    test_addition_correctness(a, b);
                    // a<b
                    let a = mpint![42, 42, 42, 0];
                    let b = mpint![1, 2, 3, 4];
                    test_addition_correctness(-&a, -&b);
                    test_addition_correctness(a, b);
                }
                #[test]
                fn same_signs_internal_overflow() {
                    let a = mpint![0, 0, 0, 3, 1];
                    let b = mpint![0, 0, 0, D_MAX, 0];
                    test_addition_correctness(-&a, -&b);
                    test_addition_correctness(a, b);
                }
                #[test]
                fn same_signs_nearly_overflow_1() {
                    let a = mpint![D_MAX - 1, D_MAX - 42, D_MAX - 2];
                    let b = mpint![1, 42, 2];
                    test_addition_correctness(-&a, -&b);
                    test_addition_correctness(a, b);
                }
                #[test]
                fn same_signs_nearly_overflow_2() {
                    let a = mpint![0, 0, 0];
                    let b = mpint![D_MAX, D_MAX, D_MAX];
                    test_addition_correctness(-&a, -&b);
                    test_addition_correctness(a, b);
                }
            }

            mod diff_signs {
                use super::*;
                #[test]
                fn normal_case_lhs_gt_rhs() {
                    // a>b:
                    let a = mpint![10, 20, 30, 40];
                    let b = mpint![1, 2, 3, 4];
                    // a + (–b)
                    test_addition_correctness(a.clone(), -&b);
                    // (-a) + b
                    test_addition_correctness(-a, b);
                }
                #[test]
                fn internal_underflow_lhs_gt_rhs_1() {
                    // a>b:
                    let a = mpint![0, D_MAX, 2, 0, 1];
                    let b = mpint![0, 0, 42, 0, 0];
                    test_addition_correctness(a.clone(), -&b);
                    test_addition_correctness(-a, b);
                }
                #[test]
                fn internal_underflow_lhs_gt_rhs_2() {
                    // a>b:
                    let a = mpint![0, 0, 0, 0, 1];
                    let b = mpint![0, 42, 0, 0, 0];
                    test_addition_correctness(a.clone(), -&b);
                    test_addition_correctness(-a, b);
                }

                #[test]
                fn normal_case_lhs_lt_rhs() {
                    // a<b:
                    let a = mpint![1, 2, 3, 4];
                    let b = mpint![10, 20, 30, 40];
                    // a + (–b)
                    test_addition_correctness(a.clone(), -&b);
                    // (-a) + b
                    test_addition_correctness(-a, b);
                }
                #[test]
                fn internal_underflow_lhs_lt_rhs_1() {
                    // a<b:
                    let a = mpint![0, 0, 42, 0, 0];
                    let b = mpint![0, D_MAX, 2, 0, 1];
                    test_addition_correctness(a.clone(), -&b);
                    test_addition_correctness(-a, b);
                }
                #[test]
                fn internal_underflow_lhs_lt_rhs_2() {
                    // a<b:
                    let a = mpint![0, 42, 0, 0, 0];
                    let b = mpint![0, 0, 0, 0, 1];
                    test_addition_correctness(a.clone(), -&b);
                    test_addition_correctness(-a, b);
                }
            }

            mod zero_result {
                use super::*;
                #[test]
                fn plus_minus_max() {
                    let a = mpint![D_MAX, D_MAX, D_MAX];
                    let b = -a.clone();
                    let zero = MPint::new_with_width(a.width());
                    assert_eq!(&a + &b, zero);
                    assert_eq!(&b + &a, zero);
                }
                #[test]
                fn normal_values() {
                    let a = mpint![630, 801, 366, 345, 372];
                    let b = -a.clone();
                    let zero = MPint::new_with_width(a.width());
                    assert_eq!(&a + &b, zero);
                    assert_eq!(&b + &a, zero);
                }
                #[test]
                fn both_zero() {
                    let a = mpint![0, 0, 0];
                    let b = a.clone();
                    let zero = MPint::new_with_width(a.width());
                    assert_eq!(&a + &b, zero);
                    assert_eq!(&-&a + &-&b, zero);
                    assert_eq!(&a + &-&b, zero);
                    assert_eq!(&-&a + &-&b, zero);
                }
            }

            mod large_values_4096 {
                use super::*;

                #[test]
                fn same_signs() {
                    let a = MPint::new(<Vec<u64>>::from(LARGE_NUM_1));
                    let b = MPint::new(<Vec<u64>>::from(LARGE_NUM_2));
                    test_addition_correctness(a.clone(), b.clone());
                    test_addition_correctness(-a, -b);
                }
                #[test]
                fn diff_signs() {
                    let a = MPint::new(<Vec<u64>>::from(LARGE_NUM_1));
                    let b = MPint::new(<Vec<u64>>::from(LARGE_NUM_2));
                    test_addition_correctness(a.clone(), -&b); //a + -b
                    test_addition_correctness(-&a, b.clone()); //-a + b
                    test_addition_correctness(b.clone(), -&a); // b + -a
                    test_addition_correctness(-b, a); //-b + a
                }
            }
        }

        mod test_from_hex_str {
            use super::*;

            #[test]
            fn low_digit_max_1() {
                let hex_str = "FFFFFFFFFFFFFFFF";
                assert_eq!(MPint::from_hex_str(hex_str), Ok(mpint![DigitT::MAX]));
            }

            #[test]
            fn low_digit_max_2() {
                let hex_str = concat!("F", "FFFFFFFFFFFFFFFF");
                assert_eq!(MPint::from_hex_str(hex_str), Ok(mpint![DigitT::MAX, 0xF]));
            }

            #[test]
            fn high_digit_non_zero() {
                let hex_str = concat!("ABCdEf0123456789", "0000000000000000");
                assert_eq!(MPint::from_hex_str(hex_str), Ok(mpint![0, 0xABCDEF0123456789]));
            }

            #[test]
            fn normal_value_1() {
                let hex_str = concat!("01", "0000000000000A00");
                assert_eq!(MPint::from_hex_str(hex_str), Ok(mpint![0xA00, 0x1]));
            }

            #[test]
            fn with_sign_1() {
                let hex_str = concat!("+", "FFFFFFFFFFFFFFFF");
                assert_eq!(MPint::from_hex_str(hex_str), Ok(mpint![DigitT::MAX]));
                let hex_str = concat!("-", "FFFFFFFFFFFFFFFF");
                assert_eq!(MPint::from_hex_str(hex_str), Ok(-mpint![DigitT::MAX]));
            }

            #[test]
            fn with_sign_2() {
                let hex_str = concat!("-A", "FFFFFFFFFFFFFFFF");
                assert_eq!(MPint::from_hex_str(hex_str), Ok(-mpint![DigitT::MAX, 0xA]));
            }

            #[test]
            fn small_value() {
                let hex_str = "123";
                assert_eq!(MPint::from_hex_str(hex_str), Ok(mpint![0x123]));
            }

            #[test]
            fn zero_1() {
                let zero = Ok(mpint![0]);
                let hex_str = "-0";
                assert_eq!(MPint::from_hex_str(hex_str), zero);
                let hex_str = "+0";
                assert_eq!(MPint::from_hex_str(hex_str), zero);
                let hex_str = "0";
                assert_eq!(MPint::from_hex_str(hex_str), zero);
            }

            #[test]
            fn zero_2_long() {
                let hex_str = "0000000000000000000000000000000000";
                assert_eq!(MPint::from_hex_str(hex_str), Ok(mpint![0]));
            }

            #[test]
            fn fail_2_empty_after_sign() {
                let hex_str = "-";
                assert!(MPint::from_hex_str(hex_str).is_err());
            }

            #[test]
            fn fail_3_empty() {
                let hex_str = "";
                assert!(MPint::from_hex_str(hex_str).is_err());
            }

            #[test]
            fn fail_4_whsp() {
                let hex_str = "1230 ";
                assert!(MPint::from_hex_str(hex_str).is_err());
                let hex_str = " 1230";
                assert!(MPint::from_hex_str(hex_str).is_err());
                let hex_str = "1 230";
                assert!(MPint::from_hex_str(hex_str).is_err());
            }

            #[test]
            fn fail_5_inv_chars() {
                let hex_str = "1230-";
                assert!(MPint::from_hex_str(hex_str).is_err());
                let hex_str = "-x1230";
                assert!(MPint::from_hex_str(hex_str).is_err());
                let hex_str = "0x1230";
                assert!(MPint::from_hex_str(hex_str).is_err());
            }
        }

        /// Verifies the result of the arithmetic operation, defined by the given
        /// parameters, using an external python script (`mpint_test_helper`).
        ///
        /// # Arguments
        /// - `lhs` - Left-hand side operand.
        /// - `op` - Operator.
        /// - `rhs` - Right-hand side operand.
        /// - `res_to_verify` - The result to verify against python's calculations.
        fn verify_arithmetic_result(
            lhs: &MPint,
            op: &str,
            rhs: &MPint,
            res_to_verify: &MPint,
        ) -> (bool, String) {
            // We will import ".../src/mpint_test_helper.py"
            let project_root = std::env::current_dir().unwrap();
            let py_mod_path = project_root.join("src");
            let py_module_dir = py_mod_path.to_str();
            let py_module_name = "mpint_test_helper";

            // Init pyo3
            pyo3::prepare_freethreaded_python();
            let py_result = Python::with_gil(move |py| -> Result<(bool, String), PyErr> {
                // Add .py file's dir to sys.path list
                let sys_path = py.import_bound("sys")?.getattr("path")?;
                let sys_path: &Bound<'_, PyList> = sys_path.downcast()?;
                sys_path.append(py_module_dir)?;

                // For this to work build.rs is setup to copy the `.py` to the target dir
                let test_helper = py.import_bound(py_module_name)?;

                let fn_name = "test_operation_result";
                let args: (String, &str, String, String, i32) = (
                    lhs.to_hex_string(),
                    op,
                    rhs.to_hex_string(),
                    res_to_verify.to_hex_string(),
                    16, //base of the number strings
                );
                let test_result: (bool, String) =
                    test_helper.getattr(fn_name)?.call1(args)?.extract()?;

                Ok(test_result)
            });

            py_result.unwrap()
        }

        const LARGE_NUM_1: [DigitT; 64] = [
            26, 57, 93, 1, 70, 36, 14, 42, 77, 64, 29, 44, 65, 3, 56, 84, 66, 88, 38, 94, 52, 46,
            73, 72, 30, 16, 8, 51, 83, 41, 34, 28, 33, 24, 40, 22, 59, 19, 99, 21, 75, 13, 96, 25,
            62, 0, 23, 18, 27, 32, 20, 85, 37, 86, 54, 80, 50, 9, 71, 60, 55, 81, 87, 2,
        ];

        const LARGE_NUM_2: [DigitT; 64] = [
            48, 96, 67, 81, 52, 61, 27, 58, 6, 59, 73, 33, 95, 91, 77, 60, 94, 76, 86, 41, 0, 42,
            89, 93, 19, 45, 64, 47, 21, 39, 10, 13, 1, 62, 43, 68, 24, 97, 15, 36, 23, 90, 25, 74,
            57, 82, 53, 99, 30, 4, 37, 31, 16, 7, 98, 69, 14, 92, 49, D_MAX, 22, 80, 26, 18,
        ];
    }
}
