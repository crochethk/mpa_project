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
}
