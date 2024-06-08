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
    println!("({num})_10 = ({})_2", u8_to_binary_str(num));

    let num = 255_u8;
    println!("({num})_10 = ({})_2", int_to_binary_str(num));
    let num = 255_u16;
    println!("({num})_10 = ({})_2", int_to_binary_str(num));
}

///
/// Generates a string of 0 and 1 representing the bits of the provided integer.
///
fn u8_to_binary_str(mut data: u8) -> String {
    let bits: usize = get_bit_width::<u8>();

    let mut result = String::new();
    const _0001: u8 = 1; // bit mask

    for _i in 0..bits {
        let bit = data & _0001;
        result += &bit.to_string();
        data >>= 1;
    }

    result.chars().rev().collect()
}

// trait MyIntConstraints: PrimInt + One + ShrAssign + 'static {}
// impl<T: PrimInt + One + ShrAssign + 'static> MyIntConstraints for T {}
// fn int_to_binary_str<T: MyIntConstraints>(mut data: T) -> String {

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
///
fn get_bit_width<T: 'static>() -> usize {
    std::mem::size_of::<T>() * 8
    // Won't use this for now, as repeated TypId::of might actually mitigate
    // the advantage of not having to do the multiplication...
    // match TypeId::of::<T>() {
    //     // Unsigned
    //     t if t == TypeId::of::<u8>() => 8,
    //     t if t == TypeId::of::<u16>() => 16,
    //     t if t == TypeId::of::<u32>() => 32,
    //     t if t == TypeId::of::<u64>() => 64,
    //     t if t == TypeId::of::<u128>() => 128,
    //     // Signed
    //     t if t == TypeId::of::<i8>() => 8,
    //     t if t == TypeId::of::<i16>() => 16,
    //     t if t == TypeId::of::<i32>() => 32,
    //     t if t == TypeId::of::<i64>() => 64,
    //     t if t == TypeId::of::<i128>() => 128,
    //     // Floating Point
    //     t if t == TypeId::of::<f32>() => 32,
    //     t if t == TypeId::of::<f64>() => 64,
    //     // Other
    //     t if t == TypeId::of::<char>() => 4 * 8,

    //     _ => std::mem::size_of::<T>() * 8,
    // }
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
