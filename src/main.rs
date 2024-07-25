use mpa_lib::mp_int::*;

fn main() {
    let two = MPint::new(2_u128);
    let div_two_by_three = &two / 3;
    let mod_two_by_three = &two % 3;
    println!("→→→→ two / 3 = {:?}", div_two_by_three); // >>> [0,0]
    println!("→→→→ two % 3 = {:?}", mod_two_by_three); // >>> 2

    let ten = MPint::new(10_u128);
    let div_ten_by_three = &ten / 3;
    let mod_ten_by_three = &ten % 3;
    println!("→→→→ ten / 3 = {:?}", div_ten_by_three); // >>> [3,0]
    println!("→→→→ ten % 3 = {:?}", mod_ten_by_three); // >>> 1

    // from_str
    let num_str = "1234";
    let num = MPint::from_dec_str(num_str).unwrap();
    println!("{:?}", num); // >>> MPint { data: [1234, 0], sign: Pos }
    println!("{}", num); // >>> 000000000000000000000000000004D2

    let num_str = "-1234";
    let num = MPint::from_dec_str(num_str).unwrap();
    println!("{:?}", num); // >>> MPint { data: [1234, 0], sign: Neg }
    println!("{}", num); // >>> -000000000000000000000000000004D2
}
