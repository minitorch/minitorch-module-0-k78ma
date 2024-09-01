"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

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
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal."""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value."""
    return abs(x - y) < 1e-1


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function."""
    # $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return max(0.0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    return math.ln(x)


def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal."""
    return 1.0 / x


def log_back(x: float, b: float) -> float:
    """Computes the derivative of log times a second arg."""
    return (1.0 / x) * b


def inv_back(x: float, b: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return (-1.0 / math.pow(x, 2)) * b


def relu_back(x: float, b: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
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

def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """
    Applies a given function to each element of an iterable.
    
    Args:
        fn: A function that takes a float and returns a float.
        ls: An iterable of floats.
    
    Returns:
        An iterable containing the results of applying fn to each element in ls.
    """
    return (fn(x) for x in ls)

def zipWith(fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Applies a given function to pairs of elements from two iterables.
    
    Args:
        fn: A function that takes two floats and returns a float.
        ls1: An iterable of floats.
        ls2: An iterable of floats.
    
    Returns:
        An iterable containing the results of applying fn to pairs of elements from ls1 and ls2.
    """
    return (fn(x, y) for x, y in zip(ls1, ls2))

def reduce(fn: Callable[[float, float], float], start: float, ls: Iterable[float]) -> float:
    """
    Reduces an iterable to a single value using a given function.
    
    Args:
        fn: A function that takes two floats and returns a float.
        start: The initial value for the reduction.
        ls: An iterable of floats.
    
    Returns:
        The final result of applying fn cumulatively to the elements of ls.
    """
    result = start
    for x in ls:
        result = fn(result, x)
    return result


# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def negList(original: Iterable[float]) -> Iterable[float]:
    """
    Negates each element in the input iterable.
    
    Args:
        original: An iterable of floats.
    
    Returns:
        An iterable containing the negated values of the input.
    """
    return list(map(fn=neg, ls=original))


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add corresponding list elements from two lists.

    Args:
        ls1: An iterable of floats
        ls2: An iterable of floats

    Returns:
        An iterable containing corresponding values of ls1 and ls2 added together.
    """
    return list(zipWith(fn=add, ls1=ls1, ls2=ls2))

def sum(ls: Iterable[float]) -> float:
    """
    Sum all elements in a list using reduce

    Args:
        ls: An iterable of floats
    """
    return reduce(fn=add, start=0, ls=ls)

def prod(ls: Iterable[float]) -> float:
    """
    Calculate the product of all elements in a list using reduce

    Args:
        ls: An iterable of floats
    """
    return reduce(fn=mul, start=1, ls=ls)