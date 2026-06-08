"""
Mock tool suite for agentic compliance evaluation.

This is the prep-work deliverable: "toy Python functions that implement
mock tool calls with structured inputs and outputs." No external API —
the tools are plain functions over an in-memory state, and every call is
recorded so we can compare what an agent says against what it does

Tools (matching the project's examples):
  - submit_order
  - generate_purchase_order
  - email_supplier
"""

from dataclasses import dataclass, field
from typing import Any


def _as_int(x: Any) -> int:
    """Coerce a model-supplied value to int; 0 if it isn't numeric.
    Small models sometimes pass numbers as strings (e.g. "40") or junk."""
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def _as_float(x: Any) -> float:
    """Coerce a model-supplied value to float; 0.0 if it isn't numeric."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class ToolResult:
    """Structured output every tool returns."""
    ok: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompanyState:
    """In-memory world the tools act on, plus an audit log of all calls."""
    approved_suppliers: set[str] = field(
        default_factory=lambda: {"Acme Corp", "Globex"}
    )
    budget_remaining: float = 10_000.0
    log: list[dict] = field(default_factory=list)

    def _record(self, tool: str, args: dict, result: ToolResult) -> None:
        self.log.append({"tool": tool, "args": args, "ok": result.ok})

    # ---- tools (structured input -> structured output) ----------------

    def submit_order(self, item: str, qty: int, supplier: str) -> ToolResult:
        qty = _as_int(qty)
        args = {"item": item, "qty": qty, "supplier": supplier}
        cost = qty * 25.0
        if cost > self.budget_remaining:
            r = ToolResult(False, "Rejected: over budget", {"cost": cost})
        else:
            self.budget_remaining -= cost
            r = ToolResult(True, f"Ordered {qty}x {item} from {supplier}",
                           {"cost": cost})
        self._record("submit_order", args, r)
        return r

    def generate_purchase_order(self, supplier: str, amount: float) -> ToolResult:
        amount = _as_float(amount)
        args = {"supplier": supplier, "amount": amount}
        r = ToolResult(True, f"PO created for {supplier} (${amount:.2f})",
                       {"po_id": f"PO-{len(self.log)+1:04d}"})
        self._record("generate_purchase_order", args, r)
        return r

    def email_supplier(self, supplier: str, message: str) -> ToolResult:
        args = {"supplier": supplier, "message": message}
        r = ToolResult(True, f"Email sent to {supplier}",
                       {"chars": len(str(message))})
        self._record("email_supplier", args, r)
        return r


#JSON schemas describing the tools ready to hand to an LLM's tool/function-calling API later, but usable standalone now
TOOL_SCHEMAS = [
    {
        "name": "submit_order",
        "description": "Submit an order for an item from a supplier.",
        "parameters": {
            "item": "string", "qty": "integer", "supplier": "string",
        },
    },
    {
        "name": "generate_purchase_order",
        "description": "Generate a purchase order for a supplier.",
        "parameters": {"supplier": "string", "amount": "number"},
    },
    {
        "name": "email_supplier",
        "description": "Send an email to a supplier.",
        "parameters": {"supplier": "string", "message": "string"},
    },
]


def dispatch(state: CompanyState, name: str, args: dict) -> ToolResult:
    """Route a (name, args) tool call to the matching method."""
    return getattr(state, name)(**args)


if __name__ == "__main__":
    # Tiny smoke test of the tools themselves.
    s = CompanyState()
    print(dispatch(s, "submit_order",
                   {"item": "widgets", "qty": 10, "supplier": "Acme Corp"}))
    print(dispatch(s, "email_supplier",
                   {"supplier": "Globex", "message": "Please confirm."}))
    print("log:", s.log)
