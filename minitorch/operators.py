"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    """
    Multiplies two numbers.
    """
    return x * y


def id(x: float) -> float:
    """
    Returns the input unchanged.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Adds two numbers.
    """
    return x + y


def neg(x: float) -> float:
    """
    Negates a number.
    """
    return -x


def lt(x: float, y: float) -> bool:
    """
    Checks if one number is less than another.
    """
    return x < y


def eq(x: float, y: float) -> bool:
    """
    Checks if two numbers are equal.
    """
    return x == y


def max(x: float, y: float) -> float:
    """
    Returns the larger of two numbers.
    """
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> bool:
    """
    Checks if two numbers are close in value.
    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """
    Calculates the sigmoid function.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(-x) / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    """
    Applies the ReLU activation function
    """
    return max(0.0, x)


def log(x: float) -> float:
    """
    Calculates the natural logarithm.
    """
    return math.ln(x)


def exp(x: float) -> float:
    """
    Calculates the exponential function.
    """
    return math.exp(x)


def inv(x: float) -> float:
    """
    Calculates the reciprocal
    """
    return 1.0 / x


def log_back(x: float, b: float) -> float:
    """
    Computes the derivative of log times a second arg
    """
    return (1.0 / x) * b


def inv_back(x: float, b: float) -> float:
    """
    Computes the derivative of reciprocal times a second arg
    """
    return (-1.0 / math.pow(x, 2)) * b


def relu_back(x: float, b: float) -> float:
    if x >= 0:
        return 1.0 * b
    else:
        return 0.0




# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def negList():
    return NotImplemented

def addLists():
    return NotImplemented

def sum():
    return NotImplemented

def prod():
    return NotImplemented