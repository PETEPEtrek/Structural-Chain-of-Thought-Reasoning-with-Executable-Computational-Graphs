#!/usr/bin/env python3
"""
Execute a computational graph: flatten its dict nodes into ordered assignment
lines, then evaluate them.

Supports:
  - const nodes and string refs
  - inline numeric literals {\"value\": ...} in op inputs
  - binary infix (+ - * / ** %) for arity-2 standard ops
  - unary ops: cos, sin, sqrt, ln, neg, ...
  - variadic chains: mul on 3+ args -> chain, add -> chain, min/max -> min(...)
"""

from __future__ import annotations

import ast
import math
from typing import Any

INFIX_OPS = frozenset({"add", "sub", "mul", "div", "pow", "mod"})


def _input_atom(inp: Any) -> str:
    if isinstance(inp, str):
        return inp.strip()
    if isinstance(inp, dict):
        if "value" in inp:
            val = inp["value"]
            if isinstance(val, bool):
                return "1.0" if val else "0.0"
            if isinstance(val, (int, float)):
                x = float(val)
                if x.is_integer() and abs(x) < 10**15:
                    return str(int(round(x)))
                return repr(val)
            return repr(val)
        raise ValueError(f"unsupported inline input dict: {inp}")
    raise ValueError(f"unsupported input: {inp!r}")


def computational_graph_to_lines(graph: list) -> list[str] | None:
    """Return lines like ``['a=3', 'b=(a)+(1)']`` or None if malformed."""
    if not isinstance(graph, list) or not graph:
        return None
    transformed: list[str] = []

    def py_op_name(raw: str) -> str:
        aliases = {"arcsin": "asin", "arccos": "acos", "arctan": "atan"}
        m = raw.strip().lower()
        return aliases.get(m, m)

    for node in graph:
        if not isinstance(node, dict):
            return None
        nid = node.get("id")
        if nid is None:
            return None
        nid = str(nid)
        nt = node.get("type")

        if nt == "const":
            transformed.append(f"{nid}={_input_atom({'value': node['value']})}")
            continue

        if nt != "op":
            return None

        op_raw = node.get("op", "")
        op = str(op_raw).strip().lower()
        inputs = node.get("inputs") or []
        atoms = [_input_atom(x) for x in inputs]

        if op == "neg" and len(atoms) == 1:
            atom = atoms[0]
            if atom.replace(".", "", 1).lstrip("-").isdigit():
                rhs = f"-{atom}"
            else:
                rhs = f"-{atom}"
        elif len(atoms) == 1:
            fname = py_op_name(op)
            rhs = f"{fname}({atoms[0]})"
        elif len(atoms) == 2 and op in INFIX_OPS:
            symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**", "mod": "%"}
            sym = symbols[op]
            rhs = f"{atoms[0]}{sym}{atoms[1]}"
        elif op == "mul" and len(atoms) >= 2:
            rhs = "*".join(atoms)
        elif op == "add" and len(atoms) >= 2:
            rhs = "+".join(atoms)
        elif op == "max" and len(atoms) >= 2:
            rhs = "max(" + ",".join(atoms) + ")"
        elif op == "min" and len(atoms) >= 2:
            rhs = "min(" + ",".join(atoms) + ")"
        else:
            fname = py_op_name(op)
            rhs = fname + "(" + ",".join(atoms) + ")"

        transformed.append(f"{nid}={rhs}")

    return transformed


_EVAL_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sqrt": math.sqrt,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log10,
    "min": min,
    "max": max,
}


def safe_eval_math_expr(expr: str, values: dict[str, float]) -> float:
    """Evaluate the right-hand side of one assignment line."""

    def _eval(node):  # noqa: PLR0911
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("unsupported constant")
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError("unsupported BinOp")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("unsupported UnaryOp")
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise KeyError(node.id)
            return values[node.id]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("unsupported call target")
            fn = _EVAL_FUNCS.get(node.func.id.lower())
            if fn is None:
                raise ValueError(f"unsupported function {node.func.id}")
            evaluated = [_eval(arg) for arg in node.args]
            return fn(*evaluated)
        raise ValueError("unsupported expression node")

    return _eval(ast.parse(expr, mode="eval"))


def compute_from_lines(graph_lines: list[str]) -> float | None:
    """Evaluate assignment lines in order; returns the last node value."""
    try:
        values: dict[str, float] = {}
        last_nid = None
        for line in graph_lines:
            if not isinstance(line, str) or "=" not in line:
                return None
            node_id, expr = line.split("=", 1)
            node_id = node_id.strip()
            expr = expr.strip().split("@", 1)[0].strip()
            if not node_id:
                return None
            def parse_literal(s: str) -> float | None:
                s = s.strip()
                try:
                    return float(s)
                except ValueError:
                    return None

            lit = parse_literal(expr)
            if lit is not None:
                val = lit
            else:
                env = dict(values)
                val = safe_eval_math_expr(expr, env)
            values[node_id] = float(val)
            last_nid = node_id
        if last_nid is None:
            return None
        return values.get(last_nid)
    except Exception:
        return None


def parse_value_float(expr: str) -> float | None:
    expr = expr.strip()
    try:
        return float(expr)
    except ValueError:
        return None

