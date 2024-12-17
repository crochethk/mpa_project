# Multiple Precision Arithmetics in Rust – an end-of-semester project

This repository is part of my semester project of the 
[lecture "Rust"](https://github.com/FaCAHell/rusty) at the
[RheinMain University of Applied Sciences](https://www.hs-rm.de/en/).

Given the limited intended workload/time and the lectures subject being the Rust
programming language, this library currently explicitly concentrates on proper
usage of Rust, rather than fancy algorithms, alike _Karatsuba Multiplication_ or
_Montgomery Multiplication_.

To accomodate for this I chose following core features:
- Operations on signed and unsigned integers:
    - addition
    - subtraction (which bascially is addition with negated operand)
    - multiplication
- Unittests for the core functionality
- Extra: demo CLI, that enables trying out the implemented arithmetics

## History
Originally the library was designed in a way, where instance creation of Multiple
Precision Integers (`MPint`) depended on an arbitrary but then more or less fixed width.
Also all operations that required two instances, like comparissons or addition, required
the operands to have the same widths.

While this can have some advantages in certain scenarios, when you need 'full'
control over widths, it turned out to feel cumbersome, user unfriendly and unintuitive
upon creating usage examples for the library.
Therefore, I decided to adjust the library, so it handles widths dynamically.
A 'wrapper library' still could be created on top of this, to assure 'fixed' widths.

Originally there should've been provided a way for the lecturer to simply copy
outcomes of an operation, for example into the Python REPL, for validation purposes.
While this manual validation by writing some code that performs the operation and 
prints the result as hex- or decimal-string is possible with the implemented functionality,
there are two things to point out:

1) The implemented demo CLI provides means of generating and performing batches 
of random test operations without writing a single line of rust code. Optionally
a simple "REPL" mode allows for writing userdefined operands and evaluating them in a
loop. This is meant to allow for easier manual correctness validation (and also
was an interesting "sidequest" to follow on the project's journey).

2) The principle of comparing with python's results is utilized in the respective
unittests automatically, i.e. arithmetic operations are performed using this 
library's implementation and then the outcome's correctness is validated by 
comparing with the outcome produced by a python interpreter (pyo3). If both outcomes
match, the corresponding test passes.
That is handy, since this way I didn't have to hardcode every single expected result.

---
> **Notice: Most of the following repo description was generated using AI based on the contents of the project report inside `doc/`.**

## Overview
The **Multiple Precision Arithmetics (MPA)** library allows for arithmetic operations on integers with arbitrary precision. By supporting operations on arbitrarily large integers, the library overcomes the limitations of standard fixed-width integer types (e.g., `u32`, `u64`), enabling precise calculations in fields such as cryptography, combinatorics, and finance. The library provides functions for addition, subtraction, and multiplication of large integers.

This repository hosts the Rust implementation of the library, complete with unit tests, documentation, and examples.

## Features
- **Arbitrary Precision Integer Operations**: Handles addition, subtraction, and multiplication of integers of any size.
- **Unit Tests**: Ensures correctness of arithmetic operations.
- **Hexadecimal and Decimal Support**: Converts integers to and from hexadecimal and decimal string representations.
- **CLI for Demonstration**: A simple command-line interface (CLI) to interact with the library and perform operations without writing code.
- **Documentation**: Detailed API documentation using Rust's `rustdoc`.

## Modules
The library is divided into the following key modules:
- **mp_int**: Contains the main data type and arithmetic functions for Multiple Precision Integers (MPI).
- **utils**: Includes helper functions and structures used by other parts of the library.
- **CLI**: A demo command-line interface to interact with the library.
- **Examples**: Various usage examples for better understanding of the library.

## Installation

To use this library in your own Rust project, follow these steps:

1. **Create a new Rust project**:
   ```bash
   cargo new my_new_project
   cd my_new_project
   ```

2. **Add the `mpa_project` as a dependency**:
   - Option 1: Add directly from the GitHub repository:
     ```bash
     cargo add --git https://github.com/crochethk/mpa_project.git
     ```
   - Option 2: Clone the repository locally and add it as a path dependency:
     ```bash
     git clone https://github.com/crochethk/mpa_project.git
     cargo add --path ../mpa_project
     ```

3. **Use the library**:
   Replace the contents of `src/main.rs` with:
   ```rust
   use mpa_lib::mp_int::*;

   fn main() {
       println!("{}", mpint!(1, 2));
   }
   ```
   Then, run the project:
   ```bash
   cargo run
   ```

## Library Usage

### Working with MPI
The `MPint` struct represents a Multiple Precision Integer (MPI) and supports the following operations:
- **Arithmetic operations**: `+`, `-`, `*`, and more.
- **Comparison**: `==`, `>`, `>=`, and others.
- **Construction**: You can create MPIs from native integers, hex or decimal strings.

Example:
```rust
let num1 = MPint::new(12345u128);
let num2 = MPint::from_hex_str("A1F4");
let num3 = MPint::from_dec_str("12345678901234567890123456789");
let result = num1 + num2 + num3;
```

### Arithmetic Operations
- **Addition & Subtraction**: Implemented using the carry ripple adder technique. Subtraction is handled via two's complement.
- **Multiplication**: Implemented using a *Product Scanning* approach.

### String Conversion
- **Hexadecimal**: `to_hex_string()`
- **Decimal**: `to_dec_string()`
- **From strings**: `from_hex_str()` and `from_dec_str()`

### Unit Testing
Unit tests for arithmetic operations are provided. To run all tests, use:
```bash
cargo test --all
```

The library utilizes **Python** for verifying arithmetic results through the `pyo3` crate. This allows for automated validation of results.

## Documentation
The code is well-documented using `rustdoc`. To generate HTML documentation, run:
```bash
cargo doc --no-deps --examples --bins
```
The generated documentation can be found under the `target/doc` directory.

## CLI
The project includes a **Demo CLI** for testing MPI operations without writing Rust code. You can use the following commands to run the demo:
```bash
cargo run
```

To see usage info simply run:
```bash
cargo run -- --help
```

## Future Work
- Extend support for additional arithmetic operations (division, modular arithmetic).
- Improve performance and optimize memory usage further.
- Explore advanced features like support for rational numbers or floating-point precision.



