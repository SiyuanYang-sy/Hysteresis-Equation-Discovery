"""Shared expression selection and numeric evaluation for symbolic regression."""

import numpy as np
import sympy


def has_independent_constant(expr, variables):
    """Detect symbolic numeric terms that are independent of input variables.

    Standalone numeric literals remain eligible for candidate selection.
    """
    symbols = set(map(sympy.Symbol, variables))
    for term in sympy.sympify(expr).as_ordered_terms():
        if not term.free_symbols & symbols and term.is_Number is False:
            if term.is_number:
                return True
    return False


def select_expression(model, variables):
    """Select the highest-scoring expression accepted by the candidate filter."""
    for _, row in model.equations_.sort_values(by="score", ascending=False).iterrows():
        expr = row["sympy_format"]
        if not has_independent_constant(expr, variables):
            return expr, row["score"]
    raise RuntimeError("No symbolic expression passed the candidate filter.")


def evaluate_expression(expr, variables, *columns):
    """Evaluate a symbolic expression as numeric arrays, including constants."""
    arrays = np.broadcast_arrays(*[np.asarray(column) for column in columns])
    function = sympy.lambdify(variables, expr, modules="numpy")
    values = np.asarray(function(*arrays), dtype=np.float64)
    values = np.broadcast_to(values, arrays[0].shape).copy()
    if not np.isfinite(values).all():
        raise ValueError("The selected expression produced non-finite predictions.")
    return values
