import ast
import hashlib
import re
import sys
from typing import List, Any, Dict, Optional, Set, Tuple

import networkx as nx

from graph.local_graph_utils import GUtils


try:
    import sympy as sp
except Exception:
    sp = None

try:
    from sympy.parsing.latex import parse_latex as sympy_parse_latex
except Exception:
    sympy_parse_latex = None

class EqExtractor(ast.NodeVisitor):
    """
    Parse equation strings attached to METHOD nodes and, optionally, project them
    as operator chains into a graph (p -> OPERATOR -> p -> ...).

    Usage patterns:
      - Single equation parsing (legacy):
            extractor = EqExtractor()
            extractor.visit(ast.parse(code, mode="eval"))
            batches = extractor.batches

      - Graph-wide extraction (new):
            extractor = EqExtractor(g=brain)  # brain is a GUtils instance
            extractor.main()  # walks METHOD nodes, builds p->op->p chains
    """

    BUILTIN_TYPES: Set[str] = {
        "str",
        "int",
        "float",
        "bool",
        "list",
        "tuple",
        "Any",
        "List",
        "Tuple",
        "array",
    }

    PATTERNS: Dict[str, str] = {
        "latex_block": r"\$\$(.*?)\$\$",
        "latex_brackets": r"\\\[(.*?)\\\]",
        "latex_equation_env": r"\\begin\{equation\}(.*?)\\end\{equation\}",
        "latex_inline": r"\$(.*?)\$",
        "assignment": r"[A-Za-z0-9_\\\{\}\^\(\)\[\]]+\s*=\s*[^=\n]+",
        "math_symbols": r"[A-Za-z0-9_\\\{\}\^\(\)\[\]]+\s*[\+\-\*/\^]\s*[A-Za-z0-9_\\\{\}\^\(\)\[\]]+",
    }

    def __init__(self, g=None, debug: bool = False):
        # g is expected to be a GUtils-like object (e.g. Brain) with .G, .add_node, .add_edge
        self.g = g
        self.debug = debug
        self.batches = []
        self.temp_count = 0
        self.init_data_type = False
        self._seen_eq_ids: Set[str] = set()
        self._last_resolved_callables: List[str] = []

    def _dbg(self, msg: str, **ctx: Any) -> None:
        if not self.debug:
            return
        if ctx:
            print(f"[EqExtractor] {msg} | {ctx}")
        else:
            print(f"[EqExtractor] {msg}")

    @staticmethod
    def _clean(eq: str) -> str:
        return " ".join((eq or "").strip().split())

    @staticmethod
    def _safe_id(*parts: str) -> str:
        joined = "::".join([str(p) for p in parts])
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _normalize_expr_text(eq: str) -> str:
        s = (eq or "").strip()
        s = s.replace("\\cdot", "*").replace("\\times", "*")
        s = s.replace("^", "**")
        return s

    def add_edge(self, src: str, trt: str, rel: str, src_layer: str, trgt_layer: str) -> None:
        if self.g is None:
            return
        self.g.add_edge(
            src=src,
            trt=trt,
            attrs={
                "rel": rel,
                "src_layer": src_layer,
                "trgt_layer": trgt_layer,
            },
        )

    def _match(self, pattern: str, text: str) -> List[str]:
        return [self._clean(x) for x in re.findall(pattern, text or "", re.DOTALL)]

    def extract_equations(self, text: str) -> List[Dict[str, Any]]:
        """
        High-recall equation extraction from plain text, markdown, and LaTeX-like inputs.
        Returns metadata-rich records for downstream parsing/projection.
        """
        print("extract_equations...", file=sys.stderr)
        equations: List[Dict[str, Any]] = []
        seen = set()
        for name, pattern in self.PATTERNS.items():
            for eq in self._match(pattern, text):
                if not eq:
                    continue
                key = (name, eq)
                if key in seen:
                    continue
                seen.add(key)
                equations.append(
                    {
                        "type": name,
                        "equation": eq,
                        "normalized": self._normalize_expr_text(eq),
                        "confidence": 0.95 if name.startswith("latex_") else 0.8,
                    }
                )
        print(equations)
        print("extract_equations... done", file=sys.stderr)
        return equations

    def parse_equation_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse one extracted equation record with robust parser cascade:
        1) SymPy LaTeX parser (when available and likely-latex)
        2) SymPy sympify for normalized expression
        3) Python AST fallback analysis
        """
        equation = str(record.get("equation") or "")
        normalized = str(record.get("normalized") or self._normalize_expr_text(equation))
        eq_type = str(record.get("type") or "")
        self._dbg("parse_equation_record:start", eq_type=eq_type, equation=equation[:120])

        variables = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", equation)))
        operators = re.findall(r"(\*\*|[+\-*/=@])", normalized)

        parsed_repr = normalized
        parser_used = "regex"
        ast_node = None
        lhs = None
        rhs = None

        if "=" in normalized:
            parts = normalized.split("=", 1)
            lhs = parts[0].strip()
            rhs = parts[1].strip()

        if sp is not None:
            try:
                if (
                    sympy_parse_latex is not None
                    and (eq_type.startswith("latex_") or "\\" in equation)
                ):
                    parsed_repr = str(sympy_parse_latex(equation))
                    parser_used = "sympy_latex"
                else:
                    parsed_repr = str(sp.sympify(rhs if rhs else normalized))
                    parser_used = "sympy_sympify"
            except Exception:
                self._dbg("parse_equation_record:sympy_failed", equation=equation[:120])

        try:
            ast_node = ast.parse(rhs if rhs else normalized, mode="eval")
            parser_used = parser_used if parser_used.startswith("sympy") else "python_ast"
        except Exception:
            ast_node = None
            self._dbg("parse_equation_record:ast_failed", normalized=normalized[:120])

        self._dbg(
            "parse_equation_record:done",
            parser=parser_used,
            variables=len(variables),
            operators=len(operators),
        )

        return {
            **record,
            "parser": parser_used,
            "parsed": parsed_repr,
            "lhs": lhs,
            "rhs": rhs,
            "variables": variables,
            "operators": operators,
            "ast_node": ast_node,
        }

    def _ensure_equation_node(self, context_id: str, module_id: str, parsed: Dict[str, Any], user_id: Optional[str]) -> str:
        eq_key = f"{context_id}|{parsed.get('type')}|{parsed.get('equation')}"
        eq_id = f"EQ::{self._safe_id(eq_key)}"
        if self.g is None or getattr(self.g, "G", None) is None:
            return eq_id
        if not self.g.G.has_node(eq_id):
            self.g.add_node(
                {
                    "id": eq_id,
                    "type": "EQUATION",
                    "equation": parsed.get("equation"),
                    "normalized": parsed.get("normalized"),
                    "parser": parsed.get("parser"),
                    "module_id": module_id,
                    "context_id": context_id,
                    "user_id": user_id,
                }
            )
        return eq_id

    def _ensure_symbol_node(self, module_id: str, symbol: str, user_id: Optional[str]) -> str:
        sid = f"SYM::{module_id}::{symbol}"
        if self.g is None or getattr(self.g, "G", None) is None:
            return sid
        if not self.g.G.has_node(sid):
            self.g.add_node(
                {
                    "id": sid,
                    "type": "SYMBOL",
                    "name": symbol,
                    "module_id": module_id,
                    "user_id": user_id,
                }
            )
        return sid

    def _ensure_constant_node(self, module_id: str, value: Any, user_id: Optional[str]) -> str:
        cid = f"CONST::{module_id}::{value}"
        if self.g is None or getattr(self.g, "G", None) is None:
            return cid
        if not self.g.G.has_node(cid):
            self.g.add_node(
                {
                    "id": cid,
                    "type": "CONSTANT",
                    "value": str(value),
                    "module_id": module_id,
                    "user_id": user_id,
                }
            )
        return cid

    def _ensure_call_node(self, eq_id: str, idx: int, name: str, module_id: str, user_id: Optional[str]) -> str:
        nid = f"FCALL::{eq_id}::{idx}"
        if self.g is None or getattr(self.g, "G", None) is None:
            return nid
        if not self.g.G.has_node(nid):
            self.g.add_node(
                {
                    "id": nid,
                    "type": "FUNCTION_CALL",
                    "func": str(name),
                    "module_id": module_id,
                    "user_id": user_id,
                }
            )
        return nid

    def _add_var_operator_links(self, eq_id: str, parsed: Dict[str, Any], module_id: str, user_id: Optional[str]) -> None:
        if self.g is None or getattr(self.g, "G", None) is None:
            return
        for var in parsed.get("variables", []):
            sid = self._ensure_symbol_node(module_id=module_id, symbol=var, user_id=user_id)
            self.add_edge(eq_id, sid, "uses_symbol", "EQUATION", "SYMBOL")
        for op in parsed.get("operators", []):
            op_id = f"EQ_OP::{eq_id}::{self._safe_id(op)}"
            if not self.g.G.has_node(op_id):
                self.g.add_node(
                    {
                        "id": op_id,
                        "type": "OPERATOR",
                        "op": op,
                        "module_id": module_id,
                        "user_id": user_id,
                    }
                )
            self.add_edge(eq_id, op_id, "applies_op", "EQUATION", "OPERATOR")

    def _project_ast_semantics(self, eq_id: str, parsed: Dict[str, Any], module_id: str, user_id: Optional[str]) -> None:
        ast_node = parsed.get("ast_node")
        if ast_node is None or self.g is None or getattr(self.g, "G", None) is None:
            return
        for idx, node in enumerate(ast.walk(ast_node)):
            if isinstance(node, ast.Call):
                fn_name = ast.unparse(node.func) if hasattr(ast, "unparse") else "call"
                call_id = self._ensure_call_node(eq_id, idx, fn_name, module_id, user_id)
                self.add_edge(eq_id, call_id, "eq_contains", "EQUATION", "FUNCTION_CALL")
                for arg_i, arg in enumerate(node.args):
                    if isinstance(arg, ast.Name):
                        sid = self._ensure_symbol_node(module_id=module_id, symbol=arg.id, user_id=user_id)
                        self.add_edge(call_id, sid, f"arg_{arg_i}", "FUNCTION_CALL", "SYMBOL")
                    elif isinstance(arg, ast.Constant):
                        cid = self._ensure_constant_node(module_id=module_id, value=arg.value, user_id=user_id)
                        self.add_edge(call_id, cid, f"arg_{arg_i}", "FUNCTION_CALL", "CONSTANT")

    def text_to_multigraph(
        self,
        text: str,
        context_id: str,
        module_id: str,
        user_id: Optional[str],
        g:GUtils
    ) -> Dict[str, Any]:
        """
        Extract all equations from text, parse them with scientific fallbacks,
        and project results into a GUtils-backed multigraph.
        """
        print("text_to_multigraph...", file=sys.stderr)

        g.print_status_G()
        self.g = g

        extracted = self.extract_equations(text)
        parsed_records = [self.parse_equation_record(r) for r in extracted]
        self._dbg(
            "text_to_multigraph:start",
            context_id=context_id,
            module_id=module_id,
            extracted=len(extracted),
        )

        added_equation_ids: List[str] = []
        for parsed in parsed_records:
            eq_id = self._ensure_equation_node(
                context_id=context_id,
                module_id=module_id,
                parsed=parsed,
                user_id=user_id,
            )
            if eq_id in self._seen_eq_ids:
                self._dbg("text_to_multigraph:skip_duplicate", eq_id=eq_id)
                continue
            self._seen_eq_ids.add(eq_id)
            added_equation_ids.append(eq_id)

            lhs = parsed.get("lhs")
            rhs = parsed.get("rhs")
            if lhs:
                lhs_id = self._ensure_symbol_node(module_id=module_id, symbol=str(lhs), user_id=user_id)
                self.add_edge(lhs_id, eq_id, "lhs_of", "SYMBOL", "EQUATION")
                self.add_edge(eq_id, lhs_id, "defines_symbol", "EQUATION", "SYMBOL")
            if rhs:
                rhs_id = self._ensure_symbol_node(module_id=module_id, symbol=str(rhs), user_id=user_id)
                self.add_edge(eq_id, rhs_id, "rhs_of", "EQUATION", "SYMBOL")

            self._add_var_operator_links(eq_id=eq_id, parsed=parsed, module_id=module_id, user_id=user_id)
            self._project_ast_semantics(eq_id=eq_id, parsed=parsed, module_id=module_id, user_id=user_id)
            self._dbg(
                "text_to_multigraph:projected",
                eq_id=eq_id,
                lhs=parsed.get("lhs"),
                rhs=parsed.get("rhs"),
            )

        g.print_status_G()
        self._dbg("text_to_multigraph... done", added=len(added_equation_ids))
        print("text_to_multigraph... done", file=sys.stderr)

        return {
            "count": len(added_equation_ids),
            "equation_ids": added_equation_ids,
            "records": parsed_records,
        }

    def _get_target(self):
        target = f"temp_{self.temp_count}"
        self.temp_count += 1
        return target

    def _resolve_callable_name(self, node: ast.AST) -> str:
        """
        Reconstruct fully-qualified callable names from AST chains.
        Examples:
        - Name(id='dot') -> 'dot'
        - Attribute(Name('jnp'), 'dot') -> 'jnp.dot'
        - Attribute(Attribute(Name('np'),'linalg'),'norm') -> 'np.linalg.norm'
        """
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._resolve_callable_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if hasattr(ast, "unparse"):
            try:
                return ast.unparse(node)
            except Exception:
                return "unknown_callable"
        return "unknown_callable"

    def _emit_batch(self, left: Any, op: str, right: Any, **extra: Any) -> str:
        res = self._get_target()
        batch = {"left": left, "op": op, "right": right, "res": res}
        if extra:
            batch.update(extra)
        self.batches.append(batch)
        return res

    def _emit_nary(self, op: str, values: List[Any], **extra: Any) -> Any:
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        cur = values[0]
        for nxt in values[1:]:
            cur = self._emit_batch(cur, op, nxt, **extra)
        return cur

    def visit_Call(self, node):
        func_name = self._resolve_callable_name(node.func)
        self._last_resolved_callables.append(func_name)

        # Method-style call receiver: x.conj() -> receiver x
        recv_val = self.visit(node.func.value) if isinstance(node.func, ast.Attribute) else None
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg or "**": self.visit(kw.value) for kw in node.keywords}

        actual_left = recv_val if recv_val is not None else (args[0] if args else None)
        actual_right = args[1] if recv_val is None and len(args) > 1 else (args[0] if recv_val and args else None)

        return self._emit_batch(
            actual_left,
            func_name,
            actual_right,
            all_args=args,
            kwargs=kwargs,
            call_path=func_name,
            call_kind="function_call",
        )

    def visit_Attribute(self, node):
        # Behandelt .T (Transponieren)
        value = self.visit(node.value)
        target = self._get_target()

        self.batches.append({
            "left": value,
            "op": node.attr,  # z.B. "T"
            "right": None,
            "res": target
        })
        return target

    def visit_Subscript(self, node):
        # Behandelt gamma[0]
        value = self.visit(node.value)
        # In Python 3.9+ ist node.slice direkt ein Constant oder Name
        index = self.visit(node.slice) if hasattr(node, 'slice') else "unknown_idx"

        target = self._get_target()
        self.batches.append({
            "left": value,
            "op": "get_item",
            "right": index,
            "res": target
        })
        return target

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_sym = self._get_op_sym(node.op)
        return self._emit_batch(left, op_sym, right)

    def _get_op_sym(self, op):
        mapping = {
            # binary arithmetic
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.MatMult: "@",
            ast.Pow: "**",
            # bitwise / shift
            ast.BitAnd: "&",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.LShift: "<<",
            ast.RShift: ">>",
        }
        return mapping.get(type(op), "unknown")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_map = {
            ast.USub: "neg",
            ast.UAdd: "pos",
            ast.Not: "not",
            ast.Invert: "~",
        }
        op_sym = op_map.get(type(node.op), "unary_unknown")
        return self._emit_batch(operand, op_sym, None, unary=True)

    def visit_BoolOp(self, node):
        op_map = {ast.And: "and", ast.Or: "or"}
        op_sym = op_map.get(type(node.op), "bool_unknown")
        vals = [self.visit(v) for v in node.values]
        return self._emit_nary(op_sym, vals, bool_op=op_sym)

    def visit_Compare(self, node):
        # Supports chains: a < b < c
        left = self.visit(node.left)
        comp_map = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "is",
            ast.IsNot: "is_not",
            ast.In: "in",
            ast.NotIn: "not_in",
        }
        pieces: List[Any] = []
        compare_ops: List[str] = []
        prev = left
        for op_node, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            op_sym = comp_map.get(type(op_node), "cmp_unknown")
            compare_ops.append(op_sym)
            cmp_res = self._emit_batch(prev, op_sym, right, compare_ops=compare_ops.copy())
            pieces.append(cmp_res)
            prev = right
        return self._emit_nary("and", pieces, bool_op="and") if len(pieces) > 1 else (pieces[0] if pieces else left)

    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        truthy = self.visit(node.body)
        falsy = self.visit(node.orelse)
        return self._emit_batch(
            truthy,
            "ifexp",
            falsy,
            condition=cond,
            then=truthy,
            otherwise=falsy,
        )

    def visit_Name(self, node):
        return node.id

    def visit_Constant(self, node):
        return node.value

    def init_data_type_nodes(self):
        """Create nodes for predefined datatype symbols."""
        if self.g is None or getattr(self.g, "G", None) is None:
            return
        for data_type in self.BUILTIN_TYPES:
            if not self.g.G.has_node(data_type):
                self.g.add_node(
                    {
                        "id": data_type,
                        "type": "DATATYPE",
                    }
                )

    def _run_ast_visit_for_equation(self, equation: str) -> List[Dict[str, Any]]:
        """
        Run the existing ast.visit batch extraction on a single equation string.
        Returns a copy of the extracted batches for this equation.
        """
        self.batches = []
        self._last_resolved_callables = []
        self.temp_count = 0
        code = (equation or "").replace("^", "**").strip()
        if not code:
            return []

        # Prefer expression mode; fallback to assignment handling.
        try:
            self.visit(ast.parse(code, mode="eval"))
        except Exception:
            if "=" in code:
                lhs, rhs = code.split("=", 1)
                rhs_clean = rhs.strip()
                if rhs_clean:
                    self.visit(ast.parse(rhs_clean, mode="eval"))
                    lhs_name = lhs.strip()
                    if lhs_name:
                        self.batches.append(
                            {
                                "left": self.batches[-1]["res"] if self.batches else rhs_clean,
                                "op": "=",
                                "right": rhs_clean,
                                "res": lhs_name,
                            }
                        )
            else:
                raise

        return [dict(b) for b in self.batches]

    def process_equation(
        self,
        text: str,
        parent_id: Optional[str] = None,
        module_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Primary processing entrypoint:
        - ALWAYS receives a single raw text string
        - extracts all equations from the text
        - processes each equation through ast.visit batching pipeline
        - optionally projects equation graph if parent_id/module_id are provided
        """
        if not isinstance(text, str):
            raise TypeError("process_equation expects a single string input")

        if self.init_data_type is False:
            self.init_data_type_nodes()
            self.init_data_type = True
            self._dbg("process_equation:init_data_types")

        text_clean = text.strip()
        if text_clean and "\n" not in text_clean:
            # Single-string mode: treat whole input as one equation/expression.
            equations = [text_clean]
        else:
            extracted_records = self.extract_equations(text)
            equations = [r.get("equation", "") for r in extracted_records if r.get("equation")]
            if text_clean and text_clean not in equations:
                equations.insert(0, text_clean)

        self._dbg(
            "process_equation:start",
            equation_count=len(equations),
            parent_id=parent_id,
            module_id=module_id,
        )

        processed: List[Dict[str, Any]] = []
        all_batches: List[Dict[str, Any]] = []
        for eq in equations:
            try:
                eq_batches = self._run_ast_visit_for_equation(eq)
                all_batches.extend(eq_batches)
                processed.append({"equation": eq, "batches": eq_batches, "status": "ok"})
                self._dbg("process_equation:eq_processed", equation=eq[:120], batches=len(eq_batches))
            except Exception as exc:
                processed.append({"equation": eq, "batches": [], "status": "error", "error": str(exc)})
                self._dbg("process_equation:eq_failed", equation=eq[:120], error=str(exc))
                continue

            if parent_id and module_id:
                try:
                    expression_node = ast.parse(eq.replace("^", "**"), mode="eval").body
                    self.analyze_expression_andadd_edges(
                        expression_node=expression_node,
                        parent_id=parent_id,
                        module_id=module_id,
                    )
                except Exception:
                    # Skip graph projection for non-ast-compatible equation strings.
                    pass

        # Keep legacy behavior: self.batches contains the aggregate.
        self.batches = all_batches
        self._dbg("process_equation:done", total_batches=len(all_batches))
        return {
            "equations": equations,
            "records": processed,
            "batches": all_batches,
        }

    def analyze_expression_andadd_edges(
        self,
        expression_node,
        parent_id,
        module_id,
    ):
        """
        Recursive AST visitor for scientific expression subgraphs.
        Returns the resulting node id for the current subtree.
        """
        if self.g is None or getattr(self.g, "G", None) is None:
            self._dbg("analyze_expression_andadd_edges:no_graph")
            return None

        if isinstance(expression_node, ast.BinOp):
            op_symbol = type(expression_node.op).__name__.lower()
            operator_node_id = f"LEGACY_OP::{module_id}::{self._safe_id(parent_id, op_symbol, str(hash(expression_node)))}"
            if not self.g.G.has_node(operator_node_id):
                self.g.add_node(
                    {
                        "id": operator_node_id,
                        "type": "OPERATOR",
                        "op": op_symbol,
                        "module_id": module_id,
                    }
                )
                self._dbg("analyze_expression_andadd_edges:add_operator", operator_node_id=operator_node_id, op=op_symbol)
            self.add_edge(parent_id, operator_node_id, "output_of", "METHOD", "OPERATOR")

            left_input_id = self.analyze_expression_andadd_edges(expression_node.left, operator_node_id, module_id)
            right_input_id = self.analyze_expression_andadd_edges(expression_node.right, operator_node_id, module_id)
            if left_input_id:
                self.add_edge(operator_node_id, left_input_id, "input_a", "OPERATOR", "PARAM")
            if right_input_id:
                self.add_edge(operator_node_id, right_input_id, "input_b", "OPERATOR", "PARAM")
            return operator_node_id

        if isinstance(expression_node, ast.Call):
            func_name = ast.unparse(expression_node.func)
            call_node_id = f"LEGACY_CALL::{module_id}::{self._safe_id(parent_id, func_name, str(hash(expression_node)))}"
            if not self.g.G.has_node(call_node_id):
                self.g.add_node(
                    {
                        "id": call_node_id,
                        "type": "FUNCTION_CALL",
                        "func": func_name,
                        "module_id": module_id,
                    }
                )
                self._dbg("analyze_expression_andadd_edges:add_call", call_node_id=call_node_id, func=func_name)
            self.add_edge(parent_id, call_node_id, "output_of", "METHOD", "FUNCTION_CALL")
            for i, arg in enumerate(expression_node.args):
                arg_input_id = self.analyze_expression_andadd_edges(arg, call_node_id, module_id)
                if arg_input_id:
                    self.add_edge(call_node_id, arg_input_id, f"arg_{i}", "FUNCTION_CALL", "PARAM")
            return call_node_id

        if isinstance(expression_node, ast.Name):
            param_name = expression_node.id
            param_node_id = self._ensure_symbol_node(module_id=module_id, symbol=param_name, user_id=None)
            self.add_edge(module_id, param_node_id, "requires_param_in_body", "METHOD", "SYMBOL")
            self._dbg("analyze_expression_andadd_edges:add_symbol_ref", symbol=param_name, node_id=param_node_id)
            return param_node_id

        if isinstance(expression_node, ast.Constant):
            return self._ensure_constant_node(module_id=module_id, value=expression_node.value, user_id=None)

        try:
            const_value = ast.unparse(expression_node)
        except Exception:
            const_value = str(expression_node)
        return self._ensure_constant_node(module_id=module_id, value=const_value, user_id=None)

    # ------------------------------------------------------------------
    # Graph projection helpers
    # ------------------------------------------------------------------
    def _ensure_param_node(self, method_id, name):
        """
        Ensure a PARAM node exists for the given method-local symbol name.
        Nodes are scoped by method so temporary symbols do not collide.
        """
        if self.g is None or getattr(self.g, "G", None) is None:
            return None
        nid = f"EQ_PARAM::{method_id}::{name}"
        if not self.g.G.has_node(nid):
            self.g.add_node(
                {
                    "id": nid,
                    "type": "PARAM",
                    "name": str(name),
                    "method_id": method_id,
                }
            )
        # Link param node to parent METHOD node for navigation and grouping.
        try:
            self.add_edge(method_id, nid, "has_eq_param", "METHOD", "PARAM")
        except Exception as e:
            print(f"[EqExtractor] _ensure_param_node: edge creation failed for {method_id}->{nid}: {e}")
        return nid

    def _ensure_op_node(self, method_id, idx, op_name):
        """
        Ensure an OPERATOR node exists for a specific batch index of a method.
        """
        if self.g is None or getattr(self.g, "G", None) is None:
            return None
        nid = f"EQ_OP::{method_id}::{idx}"
        if not self.g.G.has_node(nid):
            self.g.add_node(
                {
                    "id": nid,
                    "type": "OPERATOR",
                    "op": str(op_name),
                    "method_id": method_id,
                    "op_index": idx,
                }
            )
        # Link operator node to parent METHOD node for navigation and grouping.
        try:
            self.add_edge(method_id, nid, "has_eq_op", "METHOD", "OPERATOR")
        except Exception as e:
            print(f"[EqExtractor] _ensure_op_node: edge creation failed for {method_id}->{nid}: {e}")
        return nid

    def main(self):
        """
        Loop over METHOD nodes in self.g, parse their 'equation' strings and
        create chains within the graph of the form:

            PARAM -> OPERATOR -> PARAM -> OPERATOR -> PARAM -> ...

        For each parsed batch we:
          - create/update PARAM nodes for left/right/res symbols (scoped per method)
          - create/update an OPERATOR node for the batch
          - add edges:
                left_param  --eq_input--> OPERATOR
                right_param --eq_input--> OPERATOR (if present)
                OPERATOR    --eq_output-> result_param
        """
        if self.g is None or getattr(self.g, "G", None) is None:
            print("[EqExtractor] main: no graph attached, aborting")
            return

        import ast as _ast  # local alias to avoid confusion with module import
        self._dbg("main:start", nodes=len(self.g.G.nodes()))

        for nid, attrs in self.g.G.nodes(data=True):
            ntype = str(attrs.get("type") or "").upper()
            if ntype != "METHOD":
                continue

            equation = attrs.get("equation")
            if not isinstance(equation, str) or not equation.strip():
                continue

            # Reset batches for this method
            self.batches = []
            self.temp_count = 0

            try:
                code = equation.replace("^", "**")
                self.visit(_ast.parse(code, mode="eval"))
            except Exception as e:
                print(f"[EqExtractor] main: skip method {nid} due to parse error: {e}")
                continue
            self._dbg("main:parsed_method_equation", method_id=nid, batches=len(self.batches))

            for idx, b in enumerate(self.batches):
                op_name = b.get("op")
                if not op_name:
                    continue

                op_nid = self._ensure_op_node(nid, idx, op_name)
                if op_nid is None:
                    continue

                # Inputs: left and right (if present)
                for side in ("left", "right"):
                    val = b.get(side)
                    if val is None:
                        continue
                    p_nid = self._ensure_param_node(nid, val)
                    if p_nid is None:
                        continue
                    self.add_edge(p_nid, op_nid, "eq_input", "PARAM", "OPERATOR")

                # Output: result symbol of this batch
                res_val = b.get("res")
                if res_val is not None:
                    res_nid = self._ensure_param_node(nid, res_val)
                    if res_nid is not None:
                        self.add_edge(op_nid, res_nid, "eq_output", "OPERATOR", "PARAM")
            self._dbg("main:projected_method_equation", method_id=nid)
        self._dbg("main:done")


if __name__ == "__main__":
    # Test-Gleichungen
    # Text extraction smoke demo
    print("\n" + "-" * 80)
    print("TEXT -> MULTIGRAPH DEMO")
    demo_text = r"""
    $E = mc^2$
    \[
    i\gamma^\mu \partial_\mu \psi - m\psi = 0
    \]
    psi_t = psi + dt * H @ psi
    """
    demo_g = GUtils(G=nx.MultiDiGraph(), nx_only=True, enable_data_store=False)
    extractor = EqExtractor(g=demo_g)
    demo_out = extractor.text_to_multigraph(
        text=demo_text,
        context_id="demo_context",
        module_id="demo_module",
        user_id="demo_user",
    )
    print("extracted equations:", demo_out.get("count"))
    print("graph nodes:", len(demo_g.G.nodes()))
    print("graph edges:", len(demo_g.G.edges()))




def process_item():
    tests = [
        "-a * b + c",
        "+x",
        "~mask",
        "not done",
        "a // b % c",
        "(a < b) and (c >= d)",
        "x if cond else y",
        "jnp.dot(a, b) + np.linalg.norm(c)",
        "np.where(mask, a, b)",
        'jnp.einsum("ij,jk->ik", A, B)',
    ]

    def _check_chain(batches: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        produced = {str(b.get("res")) for b in batches if b.get("res") is not None}
        issues: List[str] = []
        for b in batches:
            for side in ("left", "right"):
                val = b.get(side)
                if isinstance(val, str) and val.startswith("temp_") and val not in produced:
                    issues.append(f"missing reference {val} in batch {b}")
            for a in b.get("all_args", []) or []:
                if isinstance(a, str) and a.startswith("temp_") and a not in produced:
                    issues.append(f"missing arg reference {a} in batch {b}")
            for v in (b.get("kwargs") or {}).values():
                if isinstance(v, str) and v.startswith("temp_") and v not in produced:
                    issues.append(f"missing kwarg reference {v} in batch {b}")
        return (len(issues) == 0, issues)

    print("\n" + "-" * 80)
    print("EqExtractor acceptance tests")
    print("-" * 80)
    for i, code in enumerate(tests, 1):
        ex = EqExtractor(debug=False)
        out = ex.process_equation(code)
        batches = out.get("batches", [])
        ops = [b.get("op") for b in batches]
        ok, issues = _check_chain(batches)
        print(f"Test {i}: {code}")
        print(f"  batches: {len(batches)}")
        print(f"  ops/functions: {ops}")
        print(f"  chain_ok: {ok}")
        if issues:
            for issue in issues[:3]:
                print(f"  chain_issue: {issue}")
        print(f"  extracted_batches: {batches}")