use mpa_lib::mp_uint::MPuint;

fn main() {
    let two = MPuint::two(128);
    let div_two_by_three = &two / 3;
    let mod_two_by_three = &two % 3;
    println!("→→→→ two / 3 = {:?}", div_two_by_three); // >>> [0,0]
    println!("→→→→ two % 3 = {:?}", mod_two_by_three); // >>> 2

    let ten = MPuint::ten(128);
    let div_ten_by_three = &ten / 3;
    let mod_ten_by_three = &ten % 3;
    println!("→→→→ ten / 3 = {:?}", div_ten_by_three); // >>> [3,0]
    println!("→→→→ ten % 3 = {:?}", mod_ten_by_three); // >>> 1

    // from_str
    let num_str = "1234";
    let width = 123;
    let num = MPuint::from_str(num_str, width).unwrap();
    println!("{:?}", num); // >>> MPuint { width: 128, data: [1234, 0] }
    println!("{}", num); // >>> 0000 ... 0000 0100 1101 0010
}
