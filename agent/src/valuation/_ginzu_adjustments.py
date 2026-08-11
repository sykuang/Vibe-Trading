"""Auxiliary calculation sheets for the standalone Ginzu model."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from math import erf, exp, isfinite, log, sqrt

from ._ginzu_types import (
    _LARGE_RATINGS,
    _SMALL_RATINGS,
    BusinessExposure,
    CostOfCapitalInput,
    CostOfCapitalResult,
    EarningsNormalizationInput,
    EarningsNormalizationResult,
    GeographicExposure,
    GinzuResult,
    LeaseInput,
    LeaseResult,
    OptionInput,
    OptionResult,
    ResearchAndDevelopmentInput,
    ResearchAndDevelopmentResult,
    StoryToNumbersResult,
    SyntheticRatingInput,
    SyntheticRatingResult,
)


def _require_finite(value: object, path: str = "inputs") -> None:
    if is_dataclass(value):
        for field in fields(value):
            _require_finite(getattr(value, field.name), field.name)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and (not isfinite(value)):
        raise ValueError(f"{path} must be finite")


def _annuity_debt(
    principal: float, interest: float, maturity: float, rate: float
) -> float:
    if maturity == 0:
        return principal
    if rate == 0:
        return principal + interest * maturity
    return (
        interest * (1 - (1 + rate) ** (-maturity)) / rate
        + principal / (1 + rate) ** maturity
    )


def normalize_earnings(
    inputs: EarningsNormalizationInput,
) -> EarningsNormalizationResult:
    _require_finite(inputs)
    if inputs.approach == "historical_ebit":
        value = inputs.historical_average_ebit
    elif inputs.approach == "historical_roc":
        value = inputs.historical_average_roc * (inputs.debt + inputs.equity)
    elif inputs.approach == "sector_margin":
        value = inputs.sector_margin * inputs.current_revenue
    else:
        raise ValueError(f"unsupported normalization approach: {inputs.approach}")
    result = EarningsNormalizationResult(value)
    _require_finite(result)
    return result


def capitalize_rd(inputs: ResearchAndDevelopmentInput) -> ResearchAndDevelopmentResult:
    _require_finite(inputs)
    if not 1 <= inputs.amortization_years <= 10:
        raise ValueError("amortization_years must be between 1 and 10")
    if len(inputs.historical_expenses) < inputs.amortization_years:
        raise ValueError("historical_expenses must cover the amortization period")
    if inputs.current_expense < 0 or any((x < 0 for x in inputs.historical_expenses)):
        raise ValueError("R&D expenses cannot be negative")
    if not 0 <= inputs.marginal_tax_rate <= 1:
        raise ValueError("marginal_tax_rate must be between 0 and 1")
    n = inputs.amortization_years
    history = inputs.historical_expenses[:n]
    asset = inputs.current_expense + sum(
        (expense * (n - age) / n for age, expense in enumerate(history, 1))
    )
    amortization = sum(history) / n
    adjustment = inputs.current_expense - amortization
    result = ResearchAndDevelopmentResult(
        asset, amortization, adjustment, adjustment * inputs.marginal_tax_rate
    )
    _require_finite(result)
    return result


def capitalize_leases(inputs: LeaseInput) -> LeaseResult:
    _require_finite(inputs)
    if inputs.pretax_cost_of_debt < 0:
        raise ValueError("pretax_cost_of_debt cannot be negative")
    if (
        inputs.current_lease_expense < 0
        or inputs.beyond_year_five < 0
        or any((x < 0 for x in inputs.commitments))
    ):
        raise ValueError("lease expenses and commitments cannot be negative")
    average = sum(inputs.commitments) / len(inputs.commitments)
    if inputs.beyond_year_five > 0 and average == 0:
        raise ValueError("positive beyond-year-five leases require earlier commitments")
    embedded = (
        int(inputs.beyond_year_five / average + 0.5)
        if inputs.beyond_year_five > 0 and average > 0
        else 0
    )
    annual_beyond = (
        inputs.beyond_year_five / embedded if embedded > 0 else inputs.beyond_year_five
    )
    rate = inputs.pretax_cost_of_debt
    pvs = [
        commitment / (1 + rate) ** year
        for year, commitment in enumerate(inputs.commitments, 1)
    ]
    if embedded > 0:
        beyond_pv = (
            annual_beyond * embedded
            if rate == 0
            else annual_beyond
            * (1 - (1 + rate) ** (-embedded))
            / rate
            / (1 + rate) ** 5
        )
    else:
        beyond_pv = annual_beyond / (1 + rate) ** 6
    pvs.append(beyond_pv)
    debt = sum(pvs)
    depreciation = debt / (5 + embedded)
    result = LeaseResult(
        embedded,
        annual_beyond,
        tuple(pvs),
        debt,
        depreciation,
        inputs.current_lease_expense - depreciation,
    )
    _require_finite(result)
    return result


def estimate_rating(inputs: SyntheticRatingInput) -> SyntheticRatingResult:
    _require_finite(inputs)
    if inputs.firm_size not in {"large", "small"}:
        raise ValueError("firm_size must be large or small")
    if inputs.interest_expense == 0:
        coverage = 1000000.0
    elif inputs.ebit < 0:
        coverage = -100000.0
    else:
        coverage = inputs.ebit / inputs.interest_expense
    table = _LARGE_RATINGS if inputs.firm_size == "large" else _SMALL_RATINGS
    _, rating, spread = max(
        (row for row in table if coverage >= row[0]), key=lambda row: row[0]
    )
    result = SyntheticRatingResult(
        coverage,
        rating,
        spread,
        inputs.country_default_spread,
        inputs.riskfree_rate + spread + inputs.country_default_spread,
    )
    _require_finite(result)
    return result


def value_options(inputs: OptionInput) -> OptionResult:
    _require_finite(inputs)
    if inputs.stock_price <= 0 or inputs.strike_price <= 0:
        raise ValueError("stock_price and strike_price must be positive")
    if inputs.maturity_years <= 0 or inputs.volatility <= 0 or inputs.share_count <= 0:
        raise ValueError("maturity_years, volatility and share_count must be positive")
    if inputs.option_count < 0:
        raise ValueError("option_count cannot be negative")
    option_value = 0.0
    adjusted = inputs.stock_price
    for iteration in range(1, 1001):
        adjusted = (
            inputs.stock_price * inputs.share_count + option_value * inputs.option_count
        ) / (inputs.share_count + inputs.option_count)
        sigma_t = inputs.volatility * sqrt(inputs.maturity_years)
        d1 = (
            log(adjusted / inputs.strike_price)
            + (inputs.riskfree_rate - inputs.dividend_yield + inputs.volatility**2 / 2)
            * inputs.maturity_years
        ) / sigma_t
        d2 = d1 - sigma_t
        cdf1 = 0.5 * (1 + erf(d1 / sqrt(2)))
        cdf2 = 0.5 * (1 + erf(d2 / sqrt(2)))
        new_value = (
            exp(-inputs.dividend_yield * inputs.maturity_years) * adjusted * cdf1
            - inputs.strike_price
            * exp(-inputs.riskfree_rate * inputs.maturity_years)
            * cdf2
        )
        if abs(new_value - option_value) <= 1e-12 * max(1.0, abs(new_value)):
            option_value = new_value
            break
        option_value = new_value
    else:
        raise ValueError("option dilution calculation did not converge")
    result = OptionResult(
        adjusted, d1, d2, option_value, option_value * inputs.option_count, iteration
    )
    _require_finite(result)
    return result


def cost_of_capital(inputs: CostOfCapitalInput) -> CostOfCapitalResult:
    _require_finite(inputs)
    if inputs.shares <= 0 or inputs.stock_price <= 0:
        raise ValueError("shares and stock_price must be positive")
    if not 0 <= inputs.marginal_tax_rate <= 1:
        raise ValueError("marginal_tax_rate must be between 0 and 1")
    if inputs.pretax_cost_of_debt < 0:
        raise ValueError("pretax_cost_of_debt cannot be negative")
    nonnegative = (
        inputs.straight_debt_book_value,
        inputs.straight_debt_interest,
        inputs.straight_debt_maturity,
        inputs.convertible_debt_book_value,
        inputs.convertible_interest,
        inputs.convertible_maturity,
        inputs.convertible_market_value,
        inputs.lease_debt,
        inputs.preferred_shares,
        inputs.preferred_price,
        inputs.preferred_dividend_per_share,
    )
    if any(value < 0 for value in nonnegative):
        raise ValueError(
            "debt, maturity, lease and preferred inputs cannot be negative"
        )
    equity = inputs.shares * inputs.stock_price
    straight = _annuity_debt(
        inputs.straight_debt_book_value,
        inputs.straight_debt_interest,
        inputs.straight_debt_maturity,
        inputs.pretax_cost_of_debt,
    )
    convertible_debt = _annuity_debt(
        inputs.convertible_debt_book_value,
        inputs.convertible_interest,
        inputs.convertible_maturity,
        inputs.pretax_cost_of_debt,
    )
    convertible_equity = inputs.convertible_market_value - convertible_debt
    debt = straight + convertible_debt + inputs.lease_debt
    preferred = inputs.preferred_shares * inputs.preferred_price
    capital = equity + debt + preferred
    if capital <= 0:
        raise ValueError("total market capital must be positive")
    levered_beta = inputs.unlevered_beta * (
        1 + (1 - inputs.marginal_tax_rate) * debt / equity
    )
    cost_equity = inputs.riskfree_rate + levered_beta * inputs.equity_risk_premium
    after_tax_debt = inputs.pretax_cost_of_debt * (1 - inputs.marginal_tax_rate)
    cost_preferred = (
        inputs.preferred_dividend_per_share / inputs.preferred_price
        if inputs.preferred_price > 0
        else 0.0
    )
    ew, dw, pw = (equity / capital, debt / capital, preferred / capital)
    result = CostOfCapitalResult(
        equity,
        straight,
        convertible_debt,
        convertible_equity,
        debt,
        preferred,
        levered_beta,
        cost_equity,
        after_tax_debt,
        cost_preferred,
        ew,
        dw,
        pw,
        ew * cost_equity + dw * after_tax_debt + pw * cost_preferred,
    )
    _require_finite(result)
    return result


def weighted_erp(exposures: tuple[GeographicExposure, ...]) -> float:
    _require_finite(exposures)
    total_revenue = sum((exposure.revenue for exposure in exposures))
    if total_revenue <= 0:
        raise ValueError("geographic exposure revenue must be positive")
    if any((exposure.revenue < 0 for exposure in exposures)):
        raise ValueError("geographic exposure revenue cannot be negative")
    result = (
        sum((exposure.revenue * exposure.equity_risk_premium for exposure in exposures))
        / total_revenue
    )
    _require_finite(result)
    return result


def bottom_up_beta(exposures: tuple[BusinessExposure, ...]) -> float:
    _require_finite(exposures)
    if any(
        exposure.revenue < 0 or exposure.enterprise_value_to_sales < 0
        for exposure in exposures
    ):
        raise ValueError(
            "business revenue and enterprise_value_to_sales must be non-negative"
        )
    values = tuple(
        (
            exposure.revenue * exposure.enterprise_value_to_sales
            for exposure in exposures
        )
    )
    total_value = sum(values)
    if total_value <= 0:
        raise ValueError("business exposure value must be positive")
    result = (
        sum(
            (
                exposure.unlevered_beta * value
                for exposure, value in zip(exposures, values, strict=True)
            )
        )
        / total_value
    )
    _require_finite(result)
    return result


def trailing_twelve_months(
    last_annual: float, prior_interim: float, current_interim: float
) -> float:
    _require_finite((last_annual, prior_interim, current_interim))
    result = last_annual - prior_interim + current_interim
    _require_finite(result)
    return result


def story_to_numbers(result: GinzuResult) -> StoryToNumbersResult:
    _require_finite(result)
    if result.stable_cost_of_capital == 0:
        raise ValueError("stable_cost_of_capital must be non-zero")
    if result.operating_asset_value == 0:
        raise ValueError("operating_asset_value must be non-zero")
    assets_in_place = result.current_forecast_ebit / result.stable_cost_of_capital
    value_of_growth = result.operating_asset_value - assets_in_place
    total = result.operating_asset_value
    story = StoryToNumbersResult(
        assets_in_place,
        value_of_growth,
        assets_in_place / total,
        value_of_growth / total,
    )
    _require_finite(story)
    return story
