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
        assert_eq!(int_to_binary_str(-2 as i32),  "1111 1111 1111 1111 1111 1111 1111 1110");
        assert_eq!(int_to_binary_str(-1 as i32),  "1111 1111 1111 1111 1111 1111 1111 1111");
        assert_eq!(int_to_binary_str(0 as i32),   "0000 0000 0000 0000 0000 0000 0000 0000");
        assert_eq!(int_to_binary_str(1 as i32),   "0000 0000 0000 0000 0000 0000 0000 0001");
        assert_eq!(int_to_binary_str(2 as i32),   "0000 0000 0000 0000 0000 0000 0000 0010");
        assert_eq!(int_to_binary_str(3 as i32),   "0000 0000 0000 0000 0000 0000 0000 0011");
        assert_eq!(int_to_binary_str(4 as i32),   "0000 0000 0000 0000 0000 0000 0000 0100");
        assert_eq!(int_to_binary_str(127 as i32), "0000 0000 0000 0000 0000 0000 0111 1111");
        assert_eq!(int_to_binary_str(u32::MAX),   "1111 1111 1111 1111 1111 1111 1111 1111");
        assert_eq!(int_to_binary_str(2_147_483_648 as u32), "1000 0000 0000 0000 0000 0000 0000 0000");
    }
}
