use num_traits::{int::PrimInt, One, WrappingAdd};
use std::ops::{Not, ShrAssign};

pub fn int_to_binary_str<T: PrimInt + One + ShrAssign + 'static>(mut data: T) -> String {
    const BLOCK_WIDTH: usize = 8;
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

/// Calculates twos complement.
/// Possible overflows are wrapped, since they are intentional.
pub fn twos_complement<T>(x: T) -> T
where
    T: Not<Output = T> + WrappingAdd + One,
{
    (!x).wrapping_add(&T::one())
}

#[cfg(test)]
mod tests {
    use super::*;

    mod test_twos_complement {
        use super::*;

        #[test]
        fn positive_value() {
            // 0001 0100 → 1110 1100 and vica versa
            assert_eq!(twos_complement(20_u8), 236);
            assert_eq!(twos_complement(236_u8), 20);
        }

        #[test]
        fn negative_value() {
            assert_eq!(twos_complement(-21_i8), 21);
            assert_eq!(twos_complement(21_i8), -21);
            assert_eq!(twos_complement(-127_i8), 127);
            assert_eq!(twos_complement(127_i8), -127);
        }

        #[test]
        fn boundries() {
            // 1111...1111 → 1000...0000
            assert_eq!(twos_complement(u16::MAX), 1);
            assert_eq!(twos_complement(1), u16::MAX);
            assert_eq!(twos_complement(-1_i16), 1);
            assert_eq!(twos_complement(1), -1_i16);
        }

        #[test]
        fn corner_case() {
            // 1000...0000 → 1000...0000
            assert_eq!(twos_complement(32_768_u16), 32_768);
            assert_eq!(twos_complement(128_u8), 128);

            // Special case. Signed int overflows (into MSB) after inverting
            assert_eq!(twos_complement(-128_i8), -128);
        }

        #[test]
        fn zero() {
            assert_eq!(twos_complement(0_u8), 0);
            assert_eq!(twos_complement(0_i16), 0);
        }
    }

    mod test_int_to_binary_str {
        use super::*;

        #[test]
        #[rustfmt::skip]
        fn normal_values() {
            assert_eq!(int_to_binary_str(0 as i8),     "00000000");
            assert_eq!(int_to_binary_str(1 as i16),    "00000000 00000001");
            assert_eq!(int_to_binary_str(2 as i32),    "00000000 00000000 00000000 00000010");
            assert_eq!(int_to_binary_str(3 as u32),    "00000000 00000000 00000000 00000011");
            assert_eq!(int_to_binary_str(4 as u32),    "00000000 00000000 00000000 00000100");
            assert_eq!(int_to_binary_str(127 as i32),  "00000000 00000000 00000000 01111111");
            assert_eq!(int_to_binary_str(-127 as i32), "11111111 11111111 11111111 10000001");
            assert_eq!(int_to_binary_str(2_147_483_648 as u32), "10000000 00000000 00000000 00000000");
        }

        #[test]
        #[rustfmt::skip]
        fn boundries() {
            assert_eq!(int_to_binary_str(i32::MIN),  "10000000 00000000 00000000 00000000");
            assert_eq!(int_to_binary_str(-2 as i32), "11111111 11111111 11111111 11111110");
            assert_eq!(int_to_binary_str(-1 as i32), "11111111 11111111 11111111 11111111");
            assert_eq!(int_to_binary_str(u32::MAX),  "11111111 11111111 11111111 11111111");
        }
    }
}
