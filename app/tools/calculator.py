"""Safe arithmetic calculator tool for AgentForge."""
from __future__ import annotations

import ast
import math
import operator
from langchain_core.tools import tool

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_POWER = 100
_MAX_ABS_VALUE = 10**100


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        value = node.value
        if abs(value) > _MAX_ABS_VALUE:
            raise ValueError("Number is too large.")
        return value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER:
            raise ValueError(f"Exponent must be between -{_MAX_POWER} and {_MAX_POWER}.")
        result = _BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(result, (int, float)) and (not math.isfinite(result) or abs(result) > _MAX_ABS_VALUE):
            raise ValueError("Result is too large or not finite.")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Only numeric constants and +, -, *, /, //, %, **, and parentheses are allowed.")


@tool
def calculator(expression: str) -> str:
    """Safely calculate a basic arithmetic expression, for example '(25 * 16) / 2'."""
    try:
        if len(expression) > 200:
            return "Calculation error: expression is too long."
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)
        return str(result)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation error: {exc}"
