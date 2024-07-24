"""
# Verify integer arithmetic operations

This script checks if the given result of an operation on integer operands
matches the outcome produced by Python's implementation of the same operation.

Usage: mpa_test_helper.py [-h] [--base BASE=10] lhs {+,-,*,/} rhs res_to_verify
Example: mpa_test_helper.py 123 + 456 579
"""

import sys, argparse

def test_operation_result(
        lhs: str,
        op: str,
        rhs: str,
        res_to_verify: str,
        base: int = 10
    ) -> bool:
    """
    Call this function from outside, to compare an externally performed operation's
    result (`res_to_verify`) with python's calculation.
    """

    # removes whitespace
    def rm_whspc(s): return "".join(s.split())

    try:
        lhs = int(rm_whspc(lhs), base)
        rhs = int(rm_whspc(rhs), base)
        res_to_verify = int(rm_whspc(res_to_verify), base)
    except ValueError:
        print("Error: input numbers must all be integers of specified base.")
        sys.exit(1)

    res_python = perform_operation(lhs, op, rhs)
    (success, msg) = cmp_result_with_given(res_python, res_to_verify)
    return (success, msg)


def perform_operation(lhs: int, op: str, rhs: int) -> int:
    """
    Performs a basic arithmetic operation on two integers.
    """
    if op == "+":
        result = lhs + rhs
    elif op == "-":
        result = lhs - rhs
    elif op == "*":
        result = lhs * rhs
    elif op == "/":
        result = lhs // rhs
    return result


def cmp_result_with_given(res_python, res_to_verify) -> tuple[bool, str]:
    """
    Compares the given with the python-calculated result.
    """
    success = res_python == res_to_verify
    out_msg = (
        f"correct: {res_python}\n" +
        f"given:   {res_to_verify}" + "\n"
    )
    if success:
        out_msg += "→→→→   ok   ←←←←"
    else:
        out_msg += "→→→→   TEST FAILED   ←←←←: calculated != given"
    return (success, out_msg)


# --------------------------- Basic CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple arithmetics tester',)
    parser.add_argument('--base', type=int, default=10, help="Input numbers base")
    parser.add_argument("lhs", type=str, help="First operand")
    parser.add_argument("op", type=str, choices=["+", "-", "*", "/"], help="Operator")
    parser.add_argument("rhs", type=str, help="Second operand")
    parser.add_argument("res_to_verify", type=str, help="Result to be verified")
    args = parser.parse_args()

    # get values and operator
    base = args.base
    a = args.lhs
    op = args.op
    b = args.rhs
    res_to_verify = args.res_to_verify

    print("-----------------------------------------------------------------")
    print(test_operation_result(a, op, b, res_to_verify, base)[1])
    print("-----------------------------------------------------------------")
