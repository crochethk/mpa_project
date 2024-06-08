use num_traits::{int::PrimInt, One};
use std::mem::size_of;
use std::ops::ShrAssign;

fn main() {
    let mut my_str: String = String::new();
    my_str += "Hello World!";
    println!("{}", my_str);

    let mut one = 1_u64;

    println!("{}", one);
    println!("{}", one >> 1);
    one <<= 1;
    one <<= 1;
    println!("{}", one >> 1);

    const BITS: usize = size_of::<u16>() * 8;
    println!("size_of(u8) = {}", BITS);

    let num = 255_u8;
    println!("({num})_10 = ({})_2", int_to_binary_str(num));
    let num = 255_u16;
    println!("({num})_10 = ({})_2", int_to_binary_str(num));
}

fn int_to_binary_str<T: PrimInt + One + ShrAssign + 'static>(mut data: T) -> String {
    let bits: usize = get_bit_width::<T>();

    let mut result = String::new();
    let _0001: T = T::one(); // bit mask

    for _i in 0..bits {
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
