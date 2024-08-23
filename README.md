# Multiple Precision Arithmetics in Rust – an end-of-semester project

This repository is part of my semester project of the lecture "Programming in Rust" 
at the [RheinMain University of Applied Sciences](https://www.hs-rm.de/en/).

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


