"""Agent tool: financial-rigor checks with exact decimal arithmetic.

A thin ``BaseTool`` wrapper around seven pure-stdlib verification and valuation
routines that guard investment research against numerical error and
hallucinated metrics.
Auto-discovered and registered via ``BaseTool.__subclasses__()``.

All arithmetic uses ``decimal.Decimal`` under a shared 28-digit context, so
results are free of IEEE-754 drift and are reproducible and auditable. The
tool takes raw numbers only — it does not fetch data. Pair it with the
market-data / financial-statement tools: fetch there, verify here.

Sub-commands (selected via ``command``):

- ``verify_market_cap`` — price × shares vs a reported cap; verdict at 1%/5%.
- ``verify_valuation`` — PE / PB / ROE / P/FCF / FCF yield / dividend yield /
  PS derived from raw per-share inputs.
- ``cross_validate`` — one field across several sources, flag deviations over
  a tolerance (default 2%), expose the median consensus.
- ``benford`` — Benford's-law first-digit check on a list of values; needs
  ≥50 samples; reports MAD / chi-square / conformity.
- ``calc`` — safe exact evaluation of an arithmetic expression string
  (AST-whitelisted: numbers and +, -, *, / only).
- ``three_scenario`` — bull / base / bear target prices from EPS-growth and
  target-PE assumptions.
- ``damodaran_fcff`` — ten-year FCFF DCF matching the core formulas in Aswath
  Damodaran's official ``fcffsimpleginzu.xlsx`` workbook.

Read-only: returns JSON verdicts, writes nothing.
"""

from __future__ import annotations

import ast
import json
import math
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from typing import Any

from src.agent.tools import BaseTool

_CTX = Context(prec=28, rounding=ROUND_HALF_EVEN)

# Benford's-law expected first-digit frequencies.
_BENFORD = {d: math.log10(1 + 1 / d) for d in range(1, 10)}

# AST operator handlers for the ``calc`` evaluator. Only numbers and
# +, -, *, / (with optional unary sign) are honoured; anything else raises.
_AST_BINOPS = {
    ast.Add: _CTX.add,
    ast.Sub: _CTX.subtract,
    ast.Mult: _CTX.multiply,
    ast.Div: _CTX.divide,
}
_AST_UNARYOPS = {
    ast.UAdd: lambda d: d,
    ast.USub: lambda d: -d,
}


def _err(msg: str) -> str:
    """Build the standard error JSON envelope."""
    return json.dumps({"status": "error", "error": msg}, ensure_ascii=False)


def _exact(value: Any) -> Decimal:
    """Convert any numeric to an exact Decimal, avoiding float binary traps."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _fmt(value: float) -> str:
    """Render a large number with K / M / B / T suffixes for readability."""
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"{value / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs_v >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:,.2f}"


def _eval_arith_node(node: ast.AST) -> Decimal:
    """Recursively evaluate an arithmetic AST node in the Decimal domain.

    Args:
        node: An AST node from a parsed expression.

    Returns:
        The exact Decimal value of the node.

    Raises:
        ValueError: If the node is not a supported numeric/arithmetic form.
    """
    if isinstance(node, ast.Constant):
        # bool is a subclass of int — reject it explicitly.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        return _exact(node.value)
    if isinstance(node, ast.BinOp):
        op_fn = _AST_BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op_fn(_eval_arith_node(node.left), _eval_arith_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _AST_UNARYOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_eval_arith_node(node.operand))
    raise ValueError(f"disallowed element in expression: {type(node).__name__}")


def _safe_arith(expr: str) -> Decimal:
    """Evaluate a numeric arithmetic expression in the exact-Decimal domain.

    The expression is parsed and evaluated recursively with Decimal arithmetic,
    so ``0.1 + 0.2`` is exactly ``0.3`` — no IEEE-754 drift, and no ``eval``.
    Only numbers and the operators ``+ - * /`` (with optional unary sign) are
    permitted; any other AST node raises ``ValueError``.

    Args:
        expr: Arithmetic expression string, e.g. ``"510 * 9.11e9"``.

    Returns:
        The exact Decimal result.

    Raises:
        ValueError: If the expression is malformed or contains a disallowed
            element.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"malformed expression: {exc}") from exc
    return _eval_arith_node(tree.body)


# ---------------------------------------------------------------------------
# Core verification routines (pure, return structured dicts, no I/O)
# ---------------------------------------------------------------------------

def verify_market_cap(
    price: Any, shares: Any, reported_cap: Any, currency: str = "",
) -> dict[str, Any]:
    """Verify ``market cap = price × shares`` against a reported value.

    Args:
        price: Current share price.
        shares: Total share count.
        reported_cap: The market-cap figure being checked.
        currency: Optional currency label for display only.

    Returns:
        Verdict dict. ``verdict`` is ``pass`` (≤1%), ``warn`` (1–5%) or
        ``fail`` (>5%).
    """
    p, s, r = _exact(price), _exact(shares), _exact(reported_cap)
    calculated = _CTX.multiply(p, s)
    deviation = abs(float(calculated - r) / float(r)) * 100 if r != 0 else 0.0
    if deviation > 5:
        verdict = "fail"
    elif deviation > 1:
        verdict = "warn"
    else:
        verdict = "pass"
    return {
        "price": float(p),
        "shares": float(s),
        "currency": currency,
        "calculated_market_cap": float(calculated),
        "calculated_market_cap_display": _fmt(float(calculated)),
        "reported_market_cap": float(r),
        "deviation_pct": round(deviation, 4),
        "verdict": verdict,
    }


def verify_valuation(
    price: Any,
    eps: Any | None = None,
    bvps: Any | None = None,
    fcf_per_share: Any | None = None,
    dividend: Any | None = None,
    revenue_per_share: Any | None = None,
) -> dict[str, Any]:
    """Derive valuation ratios from raw per-share inputs (exact decimal).

    Each optional input, when supplied and non-zero, contributes its metric(s).
    ROE additionally requires both ``eps`` and ``bvps``.

    Args:
        price: Current share price.
        eps: Earnings per share (TTM).
        bvps: Book value per share.
        fcf_per_share: Free cash flow per share.
        dividend: Dividend per share.
        revenue_per_share: Revenue per share.

    Returns:
        Dict with ``price`` and a ``metrics`` map (PE, PB, ROE_pct, P_FCF,
        FCF_yield_pct, dividend_yield_pct, PS — whichever apply).
    """
    p = _exact(price)
    metrics: dict[str, float] = {}
    if eps is not None:
        e = _exact(eps)
        if e != 0:
            metrics["PE"] = float(_CTX.divide(p, e))
            metrics["earnings_yield_pct"] = float(_CTX.divide(e, p) * 100)
    if bvps is not None:
        b = _exact(bvps)
        if b != 0:
            metrics["PB"] = float(_CTX.divide(p, b))
            if eps is not None and _exact(eps) != 0:
                metrics["ROE_pct"] = float(_CTX.divide(_exact(eps), b) * 100)
    if fcf_per_share is not None:
        f = _exact(fcf_per_share)
        if f != 0:
            metrics["P_FCF"] = float(_CTX.divide(p, f))
            metrics["FCF_yield_pct"] = float(_CTX.divide(f, p) * 100)
    if dividend is not None:
        d = _exact(dividend)
        if p != 0:
            metrics["dividend_yield_pct"] = float(_CTX.divide(d, p) * 100)
    if revenue_per_share is not None:
        rps = _exact(revenue_per_share)
        if rps != 0:
            metrics["PS"] = float(_CTX.divide(p, rps))
    return {"price": float(p), "metrics": metrics}


def cross_validate(
    field_name: str,
    source_values: dict[str, Any],
    unit: str = "",
    tolerance_pct: float = 2.0,
) -> dict[str, Any]:
    """Compare one field across sources, flag deviations over a tolerance.

    The median of the supplied values is used as the reference, and each
    source's percent deviation from it is reported.

    Args:
        field_name: Field being compared (e.g. ``"revenue"``).
        source_values: Mapping of source name to numeric value.
        unit: Optional unit label for display.
        tolerance_pct: Percent deviation above which a source is inconsistent.

    Returns:
        Dict with ``median_reference``/``consensus``, ``all_consistent`` and a
        per-source breakdown.
    """
    values = {k: _exact(v) for k, v in source_values.items()}
    nums = sorted(float(v) for v in values.values())
    n = len(nums)
    if n == 0:
        median = 0.0
    elif n % 2 == 1:
        median = nums[n // 2]
    else:
        median = (nums[n // 2 - 1] + nums[n // 2]) / 2
    per_source: list[dict[str, Any]] = []
    all_consistent = True
    for src, val in values.items():
        dev = abs(float(val) - median) / median * 100 if median != 0 else 0.0
        consistent = dev <= tolerance_pct
        all_consistent = all_consistent and consistent
        per_source.append({
            "source": src,
            "value": float(val),
            "deviation_pct": round(dev, 4),
            "consistent": consistent,
        })
    return {
        "field": field_name,
        "unit": unit,
        "tolerance_pct": tolerance_pct,
        "median_reference": median,
        "consensus": median,
        "all_consistent": all_consistent,
        "per_source": per_source,
    }


def benford_check(values: list[Any]) -> dict[str, Any]:
    """First-digit Benford's-law check on a list of financial values.

    Args:
        values: Numeric values to inspect.

    Returns:
        Dict with ``sample_size``, ``reliable`` (False when ``n < 50``), and —
        when reliable — ``mad`` (Nigrini's MAD), ``chi2``, ``conformity`` and a
        per-digit ``distribution``.
    """
    digits: list[int] = []
    for raw in values:
        v = abs(float(raw))
        if v > 0 and math.isfinite(v):
            sig = 10 ** (math.log10(v) - math.floor(math.log10(v)))
            d = int(sig)
            if 1 <= d <= 9:
                digits.append(d)
    n = len(digits)
    if n < 50:
        return {
            "sample_size": n,
            "reliable": False,
            "note": "Benford analysis needs >= 50 samples to be meaningful",
        }
    counts = {d: 0 for d in range(1, 10)}
    for d in digits:
        counts[d] += 1
    observed = {d: counts[d] / n for d in range(1, 10)}
    mad = sum(abs(observed[d] - _BENFORD[d]) for d in range(1, 10)) / 9
    chi2 = sum(
        (counts[d] - _BENFORD[d] * n) ** 2 / (_BENFORD[d] * n) for d in range(1, 10)
    )
    if mad < 0.006:
        conformity = "close"
    elif mad < 0.012:
        conformity = "acceptable"
    elif mad < 0.015:
        conformity = "marginal"
    else:
        conformity = "nonconforming"
    distribution = [
        {
            "digit": d,
            "observed": round(observed[d], 4),
            "expected": round(_BENFORD[d], 4),
            "deviation": round(observed[d] - _BENFORD[d], 4),
        }
        for d in range(1, 10)
    ]
    return {
        "sample_size": n,
        "reliable": True,
        "mad": round(mad, 6),
        "chi2": round(chi2, 4),
        "conformity": conformity,
        "is_conforming": mad < 0.015,
        "distribution": distribution,
    }


def exact_calc(expr: str) -> dict[str, Any]:
    """Evaluate an arithmetic expression with exact decimal arithmetic.

    Args:
        expr: Arithmetic expression string (numbers and ``+ - * /`` only).

    Returns:
        Dict with ``result`` (float) and ``result_exact`` (Decimal string).

    Raises:
        ValueError: If the expression is malformed or contains a disallowed
            element (surfaced by the caller as a tool error).
    """
    d = _safe_arith(expr)
    return {"expr": expr, "result": float(d), "result_exact": str(d)}


def three_scenario_valuation(
    current_price: Any,
    current_eps: Any,
    shares_billion: Any,
    growth_optimistic: Any,
    growth_neutral: Any,
    growth_pessimistic: Any,
    pe_optimistic: Any,
    pe_neutral: Any,
    pe_pessimistic: Any,
    years: int = 3,
    currency: str = "",
) -> dict[str, Any]:
    """Bull / base / bear target prices from EPS-growth and target-PE assumptions.

    Future EPS = ``current_eps × (1 + growth) ** years``; target price = future
    EPS × target PE. All math is exact decimal.

    Args:
        current_price: Current share price.
        current_eps: Current EPS.
        shares_billion: Share count in billions.
        growth_optimistic / growth_neutral / growth_pessimistic: Annual EPS
            growth rate per scenario (e.g. ``0.15`` for 15%).
        pe_optimistic / pe_neutral / pe_pessimistic: Target PE per scenario.
        years: Forecast horizon in years.
        currency: Optional currency label.

    Returns:
        Dict with the assumptions and a ``scenarios`` list, each carrying its
        ``future_eps``, ``target_price`` and ``upside_pct``.
    """
    p, eps, shares = _exact(current_price), _exact(current_eps), _exact(shares_billion)
    spec = [
        ("bull", growth_optimistic, pe_optimistic),
        ("base", growth_neutral, pe_neutral),
        ("bear", growth_pessimistic, pe_pessimistic),
    ]
    scenarios: list[dict[str, Any]] = []
    normalized: list[str] = []
    for name, growth, pe in spec:
        g, target_pe = _exact(growth), _exact(pe)
        # Defensive: LLMs frequently pass "15%" as 15 instead of 0.15. Treat
        # |growth| > 1 (i.e. > 100%) as a percent and normalize, flagging it.
        if abs(float(g)) > 1:
            g = _CTX.divide(g, Decimal("100"))
            normalized.append(name)
        future_eps = eps
        for _ in range(int(years)):
            future_eps = _CTX.multiply(future_eps, _CTX.add(Decimal("1"), g))
        target_price = _CTX.multiply(future_eps, target_pe)
        upside = float(target_price - p) / float(p) * 100 if p != 0 else 0.0
        scenarios.append({
            "scenario": name,
            "annual_growth": float(g),
            "target_pe": float(target_pe),
            "future_eps": float(future_eps),
            "target_price": float(target_price),
            "upside_pct": round(upside, 2),
        })
    result: dict[str, Any] = {
        "current_price": float(p),
        "current_eps": float(eps),
        "shares_billion": float(shares),
        "years": int(years),
        "currency": currency,
        "scenarios": scenarios,
    }
    if normalized:
        result["growth_normalized_from_percent"] = normalized
        result["note"] = (
            f"growth values > 1 (100%) were treated as percentages and divided "
            f"by 100 for scenarios: {normalized}. Pass 0.15 for 15% to avoid this."
        )
    return result


_DAMODARAN_SOURCE = "https://pages.stern.nyu.edu/~adamodar/pc/fcffsimpleginzu.xlsx"


def _damodaran_fcff_valuation(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run Damodaran's simple ten-year FCFF valuation from explicit inputs.

    This ports the core ``Valuation output`` sheet: revenue growth, operating
    margin, tax rate and cost of capital stay explicit for years 1-5 and fade
    linearly to stable assumptions in years 6-10. Reinvestment is the next
    year's incremental revenue divided by the applicable sales-to-capital
    ratio. Stable reinvestment is ``g / ROIC * NOPAT``.

    Company-level monetary inputs and shares must use corresponding scales;
    ``current_price`` remains raw currency per share. R&D, lease and employee-
    option adjustments are accepted only as normalized EBIT/debt/option-value
    inputs. If year-1 margin is omitted, adjusted EBIT/revenue supplies it.
    """
    required = {
        "revenue", "ebit", "book_equity", "debt", "cash",
        "nonoperating_assets", "minority_interest", "shares",
        "effective_tax_rate", "marginal_tax_rate", "revenue_growth_next",
        "revenue_growth_years_2_5", "target_operating_margin",
        "margin_convergence_year",
        "sales_to_capital_years_1_5", "sales_to_capital_years_6_10",
        "initial_wacc", "terminal_wacc", "terminal_growth", "terminal_roic",
    }
    optional = {
        "operating_margin_next", "current_price", "option_value", "nol",
        "failure_probability", "distress_recovery_pct", "distress_basis",
    }
    unknown = sorted(set(inputs) - required - optional)
    if unknown:
        raise ValueError(f"unsupported Damodaran FCFF inputs: {unknown}")
    missing = sorted(required - set(inputs))
    if missing:
        raise ValueError(f"missing Damodaran FCFF inputs: {missing}")

    margin_next_explicit = "operating_margin_next" in inputs
    d = {key: _exact(value) for key, value in inputs.items() if key != "distress_basis"}
    zero, one = Decimal("0"), Decimal("1")
    for key, value in d.items():
        if not value.is_finite():
            raise ValueError(f"{key} must be finite")
    if d["revenue"] <= 0:
        raise ValueError("revenue must be positive")
    if "operating_margin_next" not in d:
        d["operating_margin_next"] = _CTX.divide(d["ebit"], d["revenue"])
    if d["shares"] <= 0:
        raise ValueError("shares must be positive")
    if d.get("nol", zero) < 0:
        raise ValueError("nol must be non-negative")
    for key in ("revenue_growth_next", "revenue_growth_years_2_5", "terminal_growth"):
        if d[key] < -one:
            raise ValueError("growth rates cannot be below -1")
    for key in ("sales_to_capital_years_1_5", "sales_to_capital_years_6_10"):
        if d[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("effective_tax_rate", "marginal_tax_rate"):
        if not zero <= d[key] <= one:
            raise ValueError(f"{key} must be between 0 and 1")
    convergence = int(d["margin_convergence_year"])
    if d["margin_convergence_year"] != convergence or not 1 <= convergence <= 10:
        raise ValueError("margin_convergence_year must be an integer from 1 to 10")
    if d["terminal_wacc"] <= d["terminal_growth"]:
        raise ValueError("terminal_wacc must exceed terminal_growth")
    if d["initial_wacc"] <= -one or d["terminal_wacc"] <= -one:
        raise ValueError("initial_wacc and terminal_wacc must exceed -1")
    if d["terminal_roic"] <= 0 and d["terminal_growth"] > 0:
        raise ValueError("terminal_roic must be positive when terminal_growth is positive")
    failure_probability = d.get("failure_probability", zero)
    distress_recovery_pct = d.get("distress_recovery_pct", zero)
    for key, value in (
        ("failure_probability", failure_probability),
        ("distress_recovery_pct", distress_recovery_pct),
    ):
        if not zero <= value <= one:
            raise ValueError(f"{key} must be between 0 and 1")
    distress_basis = str(inputs.get("distress_basis") or "fair_value").strip().lower()
    if distress_basis not in {"fair_value", "book_capital"}:
        raise ValueError("distress_basis must be fair_value or book_capital")

    def calculate(terminal_wacc: Decimal, terminal_growth: Decimal) -> dict[str, Any]:
        if terminal_wacc <= terminal_growth:
            raise ValueError("terminal_wacc must exceed terminal_growth")
        if terminal_growth > 0 and d["terminal_roic"] <= 0:
            raise ValueError("terminal_roic must be positive when terminal_growth is positive")

        growth: list[Decimal] = []
        margin: list[Decimal] = []
        tax_rate: list[Decimal] = []
        wacc: list[Decimal] = []
        for year in range(1, 11):
            if year == 1:
                growth.append(d["revenue_growth_next"])
            elif year <= 5:
                growth.append(d["revenue_growth_years_2_5"])
            else:
                fade = _CTX.divide(Decimal(year - 5), Decimal("5"))
                growth.append(
                    d["revenue_growth_years_2_5"]
                    + (terminal_growth - d["revenue_growth_years_2_5"]) * fade
                )

            if year == 1:
                margin.append(d["operating_margin_next"])
            elif year > convergence:
                margin.append(d["target_operating_margin"])
            else:
                margin.append(
                    d["target_operating_margin"]
                    - _CTX.divide(
                        (d["target_operating_margin"] - d["operating_margin_next"])
                        * Decimal(convergence - year),
                        Decimal(convergence),
                    )
                )

            if year <= 5:
                tax_rate.append(d["effective_tax_rate"])
                wacc.append(d["initial_wacc"])
            else:
                fade = _CTX.divide(Decimal(year - 5), Decimal("5"))
                tax_rate.append(
                    d["effective_tax_rate"]
                    + (d["marginal_tax_rate"] - d["effective_tax_rate"]) * fade
                )
                wacc.append(d["initial_wacc"] + (terminal_wacc - d["initial_wacc"]) * fade)

        revenues: list[Decimal] = []
        previous_revenue = d["revenue"]
        for rate in growth:
            previous_revenue = previous_revenue * (one + rate)
            revenues.append(previous_revenue)
        terminal_revenue = revenues[-1] * (one + terminal_growth)

        ebit = [revenues[i] * margin[i] for i in range(10)]
        nopat: list[Decimal] = []
        nol: list[Decimal] = []
        carried_nol = d.get("nol", zero)
        for i, value in enumerate(ebit):
            if value <= 0:
                nopat.append(value)
                carried_nol -= value
            elif carried_nol >= value:
                nopat.append(value)
                carried_nol -= value
            else:
                nopat.append(value - (value - carried_nol) * tax_rate[i])
                carried_nol = zero
            nol.append(carried_nol)
        reinvestment: list[Decimal] = []
        for i in range(10):
            next_revenue = revenues[i + 1] if i < 9 else terminal_revenue
            ratio = d["sales_to_capital_years_1_5"] if i < 5 else d["sales_to_capital_years_6_10"]
            reinvestment.append(_CTX.divide(next_revenue - revenues[i], ratio))
        fcff = [nopat[i] - reinvestment[i] for i in range(10)]

        discount_factors: list[Decimal] = []
        cumulative = one
        for rate in wacc:
            cumulative = _CTX.divide(cumulative, one + rate)
            discount_factors.append(cumulative)
        pv_fcff = [fcff[i] * discount_factors[i] for i in range(10)]

        terminal_margin = margin[-1]
        terminal_nopat = terminal_revenue * terminal_margin * (one - d["marginal_tax_rate"])
        terminal_reinvestment = (
            _CTX.divide(terminal_growth, d["terminal_roic"]) * terminal_nopat
            if terminal_growth > 0 else zero
        )
        terminal_fcff = terminal_nopat - terminal_reinvestment
        terminal_value = _CTX.divide(terminal_fcff, terminal_wacc - terminal_growth)
        pv_terminal = terminal_value * discount_factors[-1]
        operating_assets_before_failure = sum(pv_fcff, zero) + pv_terminal

        if distress_basis == "book_capital":
            distress_proceeds = (d["book_equity"] + d["debt"]) * distress_recovery_pct
        else:
            distress_proceeds = operating_assets_before_failure * distress_recovery_pct
        operating_assets = (
            operating_assets_before_failure * (one - failure_probability)
            + distress_proceeds * failure_probability
        )
        equity_value = (
            operating_assets - d["debt"] - d["minority_interest"]
            + d["cash"] + d["nonoperating_assets"] - d.get("option_value", zero)
        )
        value_per_share = _CTX.divide(equity_value, d["shares"])
        return {
            "growth": growth,
            "margin": margin,
            "tax_rate": tax_rate,
            "wacc": wacc,
            "revenues": revenues,
            "ebit": ebit,
            "nopat": nopat,
            "nol": nol,
            "reinvestment": reinvestment,
            "fcff": fcff,
            "discount_factors": discount_factors,
            "pv_fcff": pv_fcff,
            "terminal_revenue": terminal_revenue,
            "terminal_nopat": terminal_nopat,
            "terminal_reinvestment": terminal_reinvestment,
            "terminal_fcff": terminal_fcff,
            "terminal_value": terminal_value,
            "pv_terminal": pv_terminal,
            "operating_assets_before_failure": operating_assets_before_failure,
            "operating_assets": operating_assets,
            "distress_proceeds": distress_proceeds,
            "equity_value": equity_value,
            "value_per_share": value_per_share,
        }

    base = calculate(d["terminal_wacc"], d["terminal_growth"])
    forecast = [
        {
            "year": i + 1,
            "revenue_growth": float(base["growth"][i]),
            "revenue": float(base["revenues"][i]),
            "operating_margin": float(base["margin"][i]),
            "ebit": float(base["ebit"][i]),
            "tax_rate": float(base["tax_rate"][i]),
            "nopat": float(base["nopat"][i]),
            "ending_nol": float(base["nol"][i]),
            "sales_to_capital": float(
                d["sales_to_capital_years_1_5"] if i < 5
                else d["sales_to_capital_years_6_10"]
            ),
            "reinvestment": float(base["reinvestment"][i]),
            "fcff": float(base["fcff"][i]),
            "wacc": float(base["wacc"][i]),
            "discount_factor": float(base["discount_factors"][i]),
            "pv_fcff": float(base["pv_fcff"][i]),
        }
        for i in range(10)
    ]

    sensitivity_wacc = [d["terminal_wacc"] + x for x in map(Decimal, ("-0.01", "0", "0.01"))]
    sensitivity_growth = [d["terminal_growth"] + x for x in map(Decimal, ("-0.005", "0", "0.005"))]
    sensitivity_values: list[list[float | None]] = []
    for wacc_value in sensitivity_wacc:
        row: list[float | None] = []
        for growth_value in sensitivity_growth:
            row.append(
                None if (
                    wacc_value <= -one
                    or growth_value < -one
                    or wacc_value <= growth_value
                    or (growth_value > 0 and d["terminal_roic"] <= 0)
                )
                else float(calculate(wacc_value, growth_value)["value_per_share"])
            )
        sensitivity_values.append(row)

    warnings: list[str] = []
    terminal_reinvestment_rate = (
        _CTX.divide(d["terminal_growth"], d["terminal_roic"])
        if d["terminal_growth"] > 0 else zero
    )
    if terminal_reinvestment_rate > one:
        warnings.append("terminal reinvestment rate exceeds 100%; terminal growth exceeds terminal ROIC")
    terminal_share = (
        _CTX.divide(base["pv_terminal"], base["operating_assets_before_failure"])
        if base["operating_assets_before_failure"] != 0 else zero
    )
    if terminal_share > Decimal("0.75"):
        warnings.append("terminal value exceeds 75% of pre-failure operating-asset value")

    result: dict[str, Any] = {
        "method": "Damodaran simple FCFF",
        "source": _DAMODARAN_SOURCE,
        "base_year": {
            "revenue": float(d["revenue"]),
            "adjusted_ebit": float(d["ebit"]),
            "adjusted_operating_margin": float(_CTX.divide(d["ebit"], d["revenue"])),
            "year_1_margin_source": (
                "explicit_operating_margin_next"
                if margin_next_explicit else "adjusted_ebit_div_revenue"
            ),
        },
        "unit_note": (
            "Company-level monetary inputs and shares must use corresponding scales "
            "(for example, millions and millions); current_price is unscaled currency per share."
        ),
        "forecast": forecast,
        "terminal": {
            "growth": float(d["terminal_growth"]),
            "wacc": float(d["terminal_wacc"]),
            "roic": float(d["terminal_roic"]),
            "reinvestment_rate": float(terminal_reinvestment_rate),
            "revenue": float(base["terminal_revenue"]),
            "nopat": float(base["terminal_nopat"]),
            "reinvestment": float(base["terminal_reinvestment"]),
            "fcff": float(base["terminal_fcff"]),
            "terminal_value": float(base["terminal_value"]),
            "pv_terminal_value": float(base["pv_terminal"]),
            "pv_share_of_operating_assets": float(terminal_share),
        },
        "pv_explicit_fcff": float(sum(base["pv_fcff"], zero)),
        "enterprise_value_before_failure": float(base["operating_assets_before_failure"]),
        "failure_probability": float(failure_probability),
        "distress_proceeds": float(base["distress_proceeds"]),
        "enterprise_value": float(base["operating_assets"]),
        "equity_bridge": {
            "debt": float(d["debt"]),
            "minority_interest": float(d["minority_interest"]),
            "cash": float(d["cash"]),
            "nonoperating_assets": float(d["nonoperating_assets"]),
            "option_value": float(d.get("option_value", zero)),
        },
        "equity_value": float(base["equity_value"]),
        "shares": float(d["shares"]),
        "value_per_share": float(base["value_per_share"]),
        "sensitivity": {
            "terminal_wacc": [float(x) for x in sensitivity_wacc],
            "terminal_growth": [float(x) for x in sensitivity_growth],
            "values_per_share": sensitivity_values,
        },
        "warnings": warnings,
    }
    if "current_price" in d:
        result["current_price"] = float(d["current_price"])
        result["price_to_value"] = (
            float(_CTX.divide(d["current_price"], base["value_per_share"]))
            if base["value_per_share"] != 0 else None
        )
        result["upside_pct"] = float(
            _CTX.divide(base["value_per_share"] - d["current_price"], d["current_price"]) * 100
        ) if d["current_price"] != 0 else None
    return result


def damodaran_fcff_valuation(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the FCFF model under its own deterministic 28-digit Decimal context."""
    with localcontext(_CTX):
        return _damodaran_fcff_valuation(inputs)


class FinancialRigorTool(BaseTool):
    """Exact-decimal financial verification and valuation."""

    name = "financial_rigor"
    description = (
        "Verify financial-data accuracy with exact decimal arithmetic (no float "
        "drift). Takes raw numbers only — does not fetch data. Pair it with the "
        "market-data / financial-statement tools: fetch there, verify here. Six "
        "sub-commands selected via `command`: 'verify_market_cap' (price x shares "
        "vs reported cap, verdict at 1%/5%), 'verify_valuation' (PE/PB/ROE/P-FCF/"
        "yields/PS from raw per-share inputs), 'cross_validate' (one field across "
        "sources, flag deviations > tolerance_pct, expose median consensus), "
        "'benford' (Benford first-digit fabrication check, needs >=50 samples), "
        "'calc' (safe exact arithmetic on an expression string), 'three_scenario' "
        "(bull/base/bear target prices from EPS-growth and target-PE assumptions), "
        "'damodaran_fcff' (official-workbook-compatible 10-year FCFF DCF with "
        "growth/margin/WACC fade, reinvestment, terminal value, equity bridge, "
        "distress adjustment and sensitivity matrix)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": [
                    "verify_market_cap", "verify_valuation", "cross_validate",
                    "benford", "calc", "three_scenario", "damodaran_fcff",
                ],
                "description": "Which verification to run.",
            },
            "price": {"type": "number", "description": "Share price."},
            "shares": {
                "type": "number",
                "description": "verify_market_cap: total share count; "
                               "three_scenario: share count in billions.",
            },
            "reported_cap": {"type": "number", "description": "Reported market cap."},
            "currency": {"type": "string", "description": "Currency label (display only)."},
            "eps": {"type": "number", "description": "Earnings per share."},
            "bvps": {"type": "number", "description": "Book value per share."},
            "fcf_per_share": {"type": "number", "description": "Free cash flow per share."},
            "dividend": {"type": "number", "description": "Dividend per share."},
            "revenue_per_share": {"type": "number", "description": "Revenue per share."},
            "field": {"type": "string", "description": "cross_validate: field name."},
            "source_values": {
                "type": "object",
                "description": "cross_validate: mapping of source name to value.",
            },
            "unit": {"type": "string", "description": "cross_validate: unit label."},
            "tolerance_pct": {
                "type": "number", "default": 2.0,
                "description": "cross_validate: max acceptable percent deviation.",
            },
            "values": {
                "type": "array", "items": {"type": "number"},
                "description": "benford: list of financial values to inspect.",
            },
            "expr": {
                "type": "string",
                "description": "calc: arithmetic expression (numbers and + - * /).",
            },
            "growth": {
                "type": "array", "items": {"type": "number"},
                "minItems": 3, "maxItems": 3,
                "description": "three_scenario: annual EPS growth as a decimal [bull, base, bear], e.g. 0.15 for 15% (values > 1 are auto-treated as percent).",
            },
            "pe": {
                "type": "array", "items": {"type": "number"},
                "minItems": 3, "maxItems": 3,
                "description": "three_scenario: target PE [bull, base, bear].",
            },
            "years": {"type": "integer", "default": 3, "description": "three_scenario horizon."},
            "fcff_inputs": {
                "type": "object",
                "additionalProperties": False,
                "description": (
                    "damodaran_fcff inputs. Company totals and shares must use "
                    "corresponding scales; current_price is unscaled currency per "
                    "share. Rates are decimals."
                ),
                "properties": {
                    "revenue": {"type": "number", "description": "Base-year/LTM revenue."},
                    "ebit": {"type": "number", "description": "Adjusted base-year operating income; derives next-year margin when operating_margin_next is omitted."},
                    "book_equity": {"type": "number"},
                    "debt": {"type": "number", "description": "Adjusted debt, including lease debt if applicable."},
                    "cash": {"type": "number"},
                    "nonoperating_assets": {"type": "number"},
                    "minority_interest": {"type": "number"},
                    "shares": {"type": "number", "exclusiveMinimum": 0},
                    "current_price": {"type": "number"},
                    "effective_tax_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    "marginal_tax_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    "revenue_growth_next": {"type": "number", "minimum": -1},
                    "revenue_growth_years_2_5": {"type": "number", "minimum": -1},
                    "operating_margin_next": {"type": "number", "description": "Optional explicit year-1 margin; defaults to adjusted EBIT / revenue."},
                    "target_operating_margin": {"type": "number"},
                    "margin_convergence_year": {"type": "integer", "minimum": 1, "maximum": 10},
                    "sales_to_capital_years_1_5": {"type": "number", "exclusiveMinimum": 0},
                    "sales_to_capital_years_6_10": {"type": "number", "exclusiveMinimum": 0},
                    "initial_wacc": {"type": "number"},
                    "terminal_wacc": {"type": "number"},
                    "terminal_growth": {"type": "number", "minimum": -1},
                    "terminal_roic": {"type": "number"},
                    "option_value": {"type": "number", "default": 0},
                    "nol": {"type": "number", "minimum": 0, "default": 0, "description": "Tax-loss carryforward entering year 1."},
                    "failure_probability": {"type": "number", "minimum": 0, "maximum": 1, "default": 0},
                    "distress_recovery_pct": {"type": "number", "minimum": 0, "maximum": 1, "default": 0},
                    "distress_basis": {"type": "string", "enum": ["fair_value", "book_capital"], "default": "fair_value"},
                },
                "required": [
                    "revenue", "ebit", "book_equity", "debt", "cash",
                    "nonoperating_assets", "minority_interest", "shares",
                    "effective_tax_rate", "marginal_tax_rate",
                    "revenue_growth_next", "revenue_growth_years_2_5",
                    "target_operating_margin",
                    "margin_convergence_year", "sales_to_capital_years_1_5",
                    "sales_to_capital_years_6_10", "initial_wacc",
                    "terminal_wacc", "terminal_growth", "terminal_roic",
                ],
            },
        },
        "required": ["command"],
    }
    is_readonly = True
    repeatable = True  # loop.py dedups non-repeatable tools by name; users call
                       # different sub-commands / params in one session.

    def execute(self, **kwargs: Any) -> str:
        """Dispatch to the requested sub-command and return a JSON envelope.

        Args:
            **kwargs: ``command`` plus the inputs for that sub-command.

        Returns:
            JSON string — ``status="ok"`` with the verdict on success,
            ``status="error"`` with a message otherwise.
        """
        command = str(kwargs.get("command") or "").strip()
        try:
            if command == "verify_market_cap":
                for key in ("price", "shares", "reported_cap"):
                    if kwargs.get(key) is None:
                        return _err(f"{key} is required for verify_market_cap")
                result: dict[str, Any] = verify_market_cap(
                    kwargs["price"], kwargs["shares"], kwargs["reported_cap"],
                    currency=str(kwargs.get("currency") or ""),
                )
            elif command == "verify_valuation":
                if kwargs.get("price") is None:
                    return _err("price is required for verify_valuation")
                result = verify_valuation(
                    kwargs["price"], kwargs.get("eps"), kwargs.get("bvps"),
                    kwargs.get("fcf_per_share"), kwargs.get("dividend"),
                    kwargs.get("revenue_per_share"),
                )
            elif command == "cross_validate":
                if not kwargs.get("field") or not kwargs.get("source_values"):
                    return _err("field and source_values are required for cross_validate")
                result = cross_validate(
                    str(kwargs["field"]), kwargs["source_values"],
                    unit=str(kwargs.get("unit") or ""),
                    tolerance_pct=float(kwargs.get("tolerance_pct") or 2.0),
                )
            elif command == "benford":
                vals = kwargs.get("values")
                if not isinstance(vals, list) or not vals:
                    return _err("values (non-empty list) is required for benford")
                result = benford_check(vals)
            elif command == "calc":
                if not kwargs.get("expr"):
                    return _err("expr is required for calc")
                result = exact_calc(str(kwargs["expr"]))
            elif command == "three_scenario":
                for key in ("price", "eps", "shares", "growth", "pe"):
                    if kwargs.get(key) is None:
                        return _err(f"{key} is required for three_scenario")
                growth = kwargs["growth"]
                pe = kwargs["pe"]
                if not isinstance(growth, list) or len(growth) != 3:
                    return _err("growth must be a list of 3 numbers [bull, base, bear]")
                if not isinstance(pe, list) or len(pe) != 3:
                    return _err("pe must be a list of 3 numbers [bull, base, bear]")
                result = three_scenario_valuation(
                    kwargs["price"], kwargs["eps"], kwargs["shares"],
                    growth[0], growth[1], growth[2],
                    pe[0], pe[1], pe[2],
                    years=int(kwargs.get("years") or 3),
                    currency=str(kwargs.get("currency") or ""),
                )
            elif command == "damodaran_fcff":
                fcff_inputs = kwargs.get("fcff_inputs")
                if not isinstance(fcff_inputs, dict):
                    return _err("fcff_inputs object is required for damodaran_fcff")
                result = damodaran_fcff_valuation(fcff_inputs)
            else:
                return _err(f"unknown command: {command}")
        except Exception as exc:  # noqa: BLE001 - surface a clean tool error
            return json.dumps(
                {"status": "error", "command": command, "error": str(exc)},
                ensure_ascii=False,
                allow_nan=False,
            )
        return json.dumps(
            {"status": "ok", "command": command, **result},
            ensure_ascii=False,
            allow_nan=False,
        )
