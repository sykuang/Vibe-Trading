"""Standalone Python port of Damodaran's full FCFF Ginzu workbook.

The module contains no agent or MCP registration.  Its public facade mirrors the
workbook's calculation sheets while keeping market/reference data as explicit
inputs, so valuation results never depend on an Excel runtime or stale hidden
state.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from ._ginzu_adjustments import (
    _require_finite,
    bottom_up_beta,
    capitalize_leases,
    capitalize_rd,
    cost_of_capital,
    estimate_rating,
    normalize_earnings,
    story_to_numbers,
    trailing_twelve_months,
    value_options,
    weighted_erp,
)
from ._ginzu_types import (
    BusinessExposure,
    CostOfCapitalInput,
    CostOfCapitalResult,
    CountryReference,
    EarningsNormalizationInput,
    EarningsNormalizationResult,
    FCFF_GINZU_SOURCE_SHA256,
    FCFF_GINZU_SOURCE_URL,
    Financials,
    ForecastAssumptions,
    ForecastYear,
    GeographicExposure,
    GinzuInput,
    GinzuReferenceData,
    GinzuResult,
    LeaseInput,
    LeaseResult,
    MarketInputs,
    OptionInput,
    OptionResult,
    ResearchAndDevelopmentInput,
    ResearchAndDevelopmentResult,
    StoryToNumbersResult,
    SyntheticRatingInput,
    SyntheticRatingResult,
)


class DamodaranGinzu:
    """Pure-Python implementation of the workbook's calculation sheets."""

    def normalize_earnings(
        self, inputs: EarningsNormalizationInput
    ) -> EarningsNormalizationResult:
        return normalize_earnings(inputs)

    def capitalize_rd(
        self, inputs: ResearchAndDevelopmentInput
    ) -> ResearchAndDevelopmentResult:
        return capitalize_rd(inputs)

    def capitalize_leases(self, inputs: LeaseInput) -> LeaseResult:
        return capitalize_leases(inputs)

    def estimate_rating(self, inputs: SyntheticRatingInput) -> SyntheticRatingResult:
        return estimate_rating(inputs)

    def value_options(self, inputs: OptionInput) -> OptionResult:
        return value_options(inputs)

    def cost_of_capital(self, inputs: CostOfCapitalInput) -> CostOfCapitalResult:
        return cost_of_capital(inputs)

    @staticmethod
    def weighted_erp(exposures: tuple[GeographicExposure, ...]) -> float:
        return weighted_erp(exposures)

    @staticmethod
    def bottom_up_beta(exposures: tuple[BusinessExposure, ...]) -> float:
        return bottom_up_beta(exposures)

    @staticmethod
    def trailing_twelve_months(
        last_annual: float, prior_interim: float, current_interim: float
    ) -> float:
        return trailing_twelve_months(last_annual, prior_interim, current_interim)

    @staticmethod
    def story_to_numbers(result: GinzuResult) -> StoryToNumbersResult:
        return story_to_numbers(result)

    def _resolve_debt_cost_and_lease(
        self,
        inputs: GinzuInput,
    ) -> tuple[float, str | None, LeaseResult | None]:
        market, financials = inputs.market, inputs.financials
        cost = market.pretax_cost_of_debt
        rating_name: str | None = None
        for _ in range(100):
            lease = (
                self.capitalize_leases(replace(inputs.lease, pretax_cost_of_debt=cost))
                if inputs.lease
                else None
            )
            if not market.use_synthetic_rating:
                return cost, None, lease
            rating = self.estimate_rating(
                SyntheticRatingInput(
                    firm_size=market.rating_firm_size,
                    ebit=financials.operating_income
                    + (lease.operating_income_adjustment if lease else 0),
                    interest_expense=financials.interest_expense
                    + (lease.lease_debt * cost if lease else 0),
                    riskfree_rate=market.riskfree_rate,
                    country_default_spread=market.country_default_spread,
                )
            )
            rating_name = rating.rating
            if abs(rating.cost_of_debt - cost) <= 1e-12:
                final_lease = (
                    self.capitalize_leases(
                        replace(
                            inputs.lease,
                            pretax_cost_of_debt=rating.cost_of_debt,
                        )
                    )
                    if inputs.lease
                    else None
                )
                return rating.cost_of_debt, rating_name, final_lease
            cost = rating.cost_of_debt
        raise ValueError("synthetic rating and lease calculation did not converge")

    def value(self, inputs: GinzuInput) -> GinzuResult:
        _require_finite(inputs)
        financials, market, assumptions = (
            inputs.financials,
            inputs.market,
            inputs.forecast,
        )
        self._validate_value_inputs(inputs)
        pretax_cost, synthetic_rating, lease = self._resolve_debt_cost_and_lease(inputs)
        market = replace(market, pretax_cost_of_debt=pretax_cost)
        inputs = replace(inputs, market=market)
        normalization = (
            replace(
                inputs.normalization,
                current_revenue=financials.current_revenue,
                debt=financials.current_book_debt,
                equity=financials.current_book_equity,
            )
            if inputs.normalization
            else None
        )
        normalized = (
            self.normalize_earnings(normalization).normalized_ebit
            if normalization
            else financials.operating_income
        )
        rd_input = (
            replace(inputs.rd, marginal_tax_rate=financials.marginal_tax_rate)
            if inputs.rd
            else None
        )
        rd = self.capitalize_rd(rd_input) if rd_input else None
        adjusted_ebit = (
            normalized
            + (rd.operating_income_adjustment if rd else 0)
            + (lease.operating_income_adjustment if lease else 0)
        )
        adjusted_interest = financials.interest_expense + (
            lease.lease_debt * pretax_cost if lease else 0
        )
        adjusted_capex = (
            financials.capital_spending
            + (inputs.rd.current_expense if inputs.rd else 0)
            + (inputs.lease.current_lease_expense if inputs.lease else 0)
        )
        adjusted_depreciation = (
            financials.depreciation
            + (rd.current_amortization if rd else 0)
            + (lease.depreciation if lease else 0)
        )
        adjusted_previous_debt = financials.previous_book_debt + (
            lease.lease_debt if lease else 0
        )
        adjusted_previous_equity = financials.previous_book_equity + (
            rd.research_asset - inputs.rd.current_expense + rd.current_amortization
            if rd and inputs.rd
            else 0
        )
        change_wc = financials.change_in_working_capital
        if change_wc < 0:
            change_wc = (
                (financials.current_revenue - financials.previous_revenue)
                * financials.current_non_cash_working_capital
                / financials.current_revenue
            )
        current_forecast_ebit = adjusted_ebit + (rd.tax_effect if rd else 0)
        current_nopat = current_forecast_ebit * (1 - financials.effective_tax_rate)
        invested_capital = (
            adjusted_previous_debt
            + adjusted_previous_equity
            - financials.previous_cash
            - financials.previous_non_operating_assets
        )
        if assumptions.high_growth_return_on_capital is None:
            if invested_capital == 0:
                raise ValueError(
                    "invested_capital must be non-zero when deriving return on capital"
                )
            high_roc = (
                adjusted_ebit * (1 - financials.marginal_tax_rate)
                + (rd.tax_effect if rd else 0)
            ) / invested_capital
        else:
            high_roc = assumptions.high_growth_return_on_capital
        if assumptions.high_growth_reinvestment_rate is None:
            if current_nopat == 0:
                raise ValueError(
                    "current_nopat must be non-zero when deriving reinvestment rate"
                )
            high_reinvestment = (
                adjusted_capex - adjusted_depreciation + change_wc
            ) / current_nopat
        else:
            high_reinvestment = assumptions.high_growth_reinvestment_rate
        high_growth = (
            high_roc * high_reinvestment
            if assumptions.use_fundamental_growth
            else assumptions.explicit_high_growth_rate
        )

        option_after_tax = 0.0
        debt_ratio = market.high_growth_debt_ratio
        result: GinzuResult | None = None
        estimated_price = market.current_price or 1.0
        for _ in range(100):
            if inputs.option:
                option_price = (
                    estimated_price
                    if inputs.option.use_estimated_value
                    else market.current_price
                )
                assert option_price is not None
                option_input = replace(
                    inputs.option,
                    stock_price=option_price,
                    share_count=market.shares,
                    riskfree_rate=market.riskfree_rate,
                )
                option_after_tax = self.value_options(option_input).total_value * (
                    1 - financials.marginal_tax_rate
                )
            lease_debt = lease.lease_debt if lease else 0.0
            if market.keep_current_debt_ratio:
                if market.stock_traded:
                    assert market.current_price is not None
                    equity_for_ratio = market.current_price * market.shares + (
                        option_after_tax if lease else 0.0
                    )
                    debt_ratio = 1 - equity_for_ratio / (
                        market.market_value_debt + lease_debt + equity_for_ratio
                    )
                else:
                    book_capital = (
                        financials.current_book_debt + financials.current_book_equity
                    )
                    if book_capital <= 0:
                        raise ValueError(
                            "private-company book capital must be positive"
                        )
                    debt_ratio = financials.current_book_debt / book_capital
            if not 0 <= debt_ratio <= 1:
                raise ValueError(
                    "computed high_growth_debt_ratio must be between 0 and 1"
                )
            try:
                result = self._value_once(
                    inputs,
                    normalized,
                    adjusted_ebit,
                    adjusted_interest,
                    adjusted_capex,
                    adjusted_depreciation,
                    current_forecast_ebit,
                    current_nopat,
                    high_growth,
                    high_roc,
                    high_reinvestment,
                    debt_ratio,
                    option_after_tax,
                    lease_debt,
                    synthetic_rating,
                )
            except (OverflowError, ZeroDivisionError) as exc:
                raise ValueError(
                    "valuation result must be finite and all denominators non-zero"
                ) from exc
            _require_finite(result)
            if not inputs.option or not inputs.option.use_estimated_value:
                break
            if abs(result.value_per_share - estimated_price) <= 1e-11 * max(
                1.0, abs(result.value_per_share)
            ):
                break
            estimated_price = result.value_per_share
        else:  # pragma: no cover - defensive outer circularity guard
            raise ValueError("estimated-price option valuation did not converge")
        assert result is not None
        return result

    def _value_once(
        self,
        inputs: GinzuInput,
        normalized: float,
        adjusted_ebit: float,
        adjusted_interest: float,
        adjusted_capex: float,
        adjusted_depreciation: float,
        current_forecast_ebit: float,
        current_nopat: float,
        high_growth: float,
        high_roc: float,
        high_reinvestment: float,
        debt_ratio: float,
        option_after_tax: float,
        lease_debt: float,
        synthetic_rating: str | None,
    ) -> GinzuResult:
        f, m, a = inputs.financials, inputs.market, inputs.forecast
        high_cost_equity = m.riskfree_rate + m.high_growth_beta * m.equity_risk_premium
        high_after_tax_debt = m.pretax_cost_of_debt * (1 - f.marginal_tax_rate)
        high_wacc = (
            high_cost_equity * (1 - debt_ratio) + high_after_tax_debt * debt_ratio
        )
        if high_wacc <= -1:
            raise ValueError("high-growth cost of capital must exceed -1")
        stable_cost_equity = (
            m.riskfree_rate + a.stable_beta * a.stable_equity_risk_premium
        )
        stable_after_tax_debt = a.stable_pretax_cost_of_debt * (1 - a.stable_tax_rate)
        stable_wacc = (
            stable_cost_equity * (1 - a.stable_debt_ratio)
            + stable_after_tax_debt * a.stable_debt_ratio
        )
        if stable_wacc <= a.stable_growth_rate:
            raise ValueError("stable cost of capital must exceed stable growth")
        stable_reinvestment = (
            a.stable_growth_rate / a.stable_return_on_capital
            if a.use_fundamental_stable_reinvestment
            else 0.0
        )
        years = a.high_growth_years
        cumulative_growth = 1.0
        cumulative_cost = 1.0
        previous_cumulative_growth = 1.0
        forecast: list[ForecastYear] = []
        for year in range(1, years + 1):
            growth = self._transition(
                high_growth, a.stable_growth_rate, year, years, a.adjust_second_half
            )
            reinvestment = self._transition(
                high_reinvestment,
                stable_reinvestment,
                year,
                years,
                a.adjust_second_half,
            )
            wacc = self._transition(
                high_wacc, stable_wacc, year, years, a.adjust_second_half
            )
            cumulative_growth *= 1 + growth
            ebit = current_forecast_ebit * cumulative_growth
            tax = (
                a.stable_tax_rate
                - (years - year) * (a.stable_tax_rate - f.effective_tax_rate) / years
            )
            nopat = ebit * (1 - tax)
            change_wc = (
                f.current_revenue
                * (cumulative_growth - previous_cumulative_growth)
                * a.working_capital_ratio
            )
            net_capex = reinvestment * nopat - change_wc
            fcff = nopat - net_capex - change_wc
            cumulative_cost *= 1 + wacc
            pv = fcff / cumulative_cost
            forecast.append(
                ForecastYear(
                    year,
                    growth,
                    cumulative_growth,
                    reinvestment,
                    ebit,
                    tax,
                    nopat,
                    net_capex,
                    change_wc,
                    fcff,
                    wacc,
                    cumulative_cost,
                    pv,
                )
            )
            previous_cumulative_growth = cumulative_growth
        max_nopat = max([current_nopat, *(row.nopat for row in forecast)])
        terminal_nopat = (
            max_nopat
            / (1 - f.marginal_tax_rate)
            * (1 - a.stable_tax_rate)
            * (1 + a.stable_growth_rate)
        )
        terminal_revenue_base = f.current_revenue * (1 + high_growth) ** years
        terminal_wc = (
            terminal_revenue_base * a.stable_growth_rate * a.working_capital_ratio
        )
        if a.use_fundamental_stable_reinvestment:
            terminal_net_capex = stable_reinvestment * terminal_nopat - terminal_wc
        else:
            terminal_net_capex = (
                (a.stable_capex_to_depreciation - 1)
                * adjusted_depreciation
                * (1 + high_growth) ** years
                * (1 + a.stable_growth_rate)
            )
            stable_reinvestment = (terminal_net_capex + terminal_wc) / terminal_nopat
        terminal_fcff = terminal_nopat - terminal_net_capex - terminal_wc
        terminal_value = terminal_fcff / (stable_wacc - a.stable_growth_rate)
        pv_high = sum(row.present_value for row in forecast)
        pv_terminal = (
            terminal_value
            if not forecast
            else terminal_value
            / max(row.cumulative_cost_of_capital for row in forecast)
        )
        operating_assets = pv_high + pv_terminal
        cash_assets = f.current_cash + f.current_non_operating_assets
        reported_debt = m.market_value_debt if m.stock_traded else f.current_book_debt
        debt = reported_debt + lease_debt
        equity = operating_assets + cash_assets - debt - f.minority_interests
        common = equity - option_after_tax
        value_per_share = common / m.shares
        diagnostics = self._diagnostics(inputs, stable_wacc, debt_ratio)
        price_gap = (
            m.current_price / value_per_share - 1
            if m.current_price is not None
            else None
        )
        return GinzuResult(
            inputs.company_name,
            normalized,
            adjusted_ebit,
            adjusted_interest,
            adjusted_capex,
            adjusted_depreciation,
            current_forecast_ebit,
            high_growth,
            high_roc,
            high_reinvestment,
            high_wacc,
            m.pretax_cost_of_debt,
            synthetic_rating,
            stable_reinvestment,
            stable_wacc,
            tuple(forecast),
            terminal_nopat,
            terminal_net_capex,
            terminal_wc,
            terminal_fcff,
            terminal_value,
            pv_high,
            pv_terminal,
            operating_assets,
            cash_assets,
            debt,
            f.minority_interests,
            equity,
            option_after_tax,
            common,
            value_per_share,
            m.current_price,
            price_gap,
            diagnostics,
        )

    @staticmethod
    def _transition(
        high: float, stable: float, year: int, years: int, enabled: bool
    ) -> float:
        if not enabled or years == 0 or year < years / 2:
            return high
        return stable + (high - stable) / (years / 2) * (years - year)

    @staticmethod
    def _validate_value_inputs(inputs: GinzuInput) -> None:
        f, m, a = inputs.financials, inputs.market, inputs.forecast
        if not 0 <= a.high_growth_years <= 15:
            raise ValueError("high_growth_years must be between 0 and 15")
        if f.current_revenue <= 0 or m.shares <= 0:
            raise ValueError("current_revenue and shares must be positive")
        if m.current_price is not None and m.current_price <= 0:
            raise ValueError("current_price must be positive when supplied")
        if m.stock_traded and m.current_price is None:
            raise ValueError("a traded company requires current_price")
        if m.keep_current_debt_ratio and m.stock_traded and m.current_price is None:
            raise ValueError("keep_current_debt_ratio requires current_price")
        if (
            inputs.option
            and not inputs.option.use_estimated_value
            and m.current_price is None
        ):
            raise ValueError("current-price option valuation requires current_price")
        for name, rate in (
            ("effective_tax_rate", f.effective_tax_rate),
            ("marginal_tax_rate", f.marginal_tax_rate),
            ("stable_tax_rate", a.stable_tax_rate),
        ):
            if not 0 <= rate <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if f.marginal_tax_rate == 1:
            raise ValueError("marginal_tax_rate must be less than 1")
        if m.pretax_cost_of_debt < 0:
            raise ValueError("pretax_cost_of_debt must be non-negative")
        if a.stable_pretax_cost_of_debt < 0:
            raise ValueError("stable_pretax_cost_of_debt must be non-negative")
        for name, amount in (
            ("market_value_debt", m.market_value_debt),
            ("current_book_debt", f.current_book_debt),
            ("previous_book_debt", f.previous_book_debt),
            ("current_cash", f.current_cash),
            ("previous_cash", f.previous_cash),
            ("current_non_operating_assets", f.current_non_operating_assets),
            ("previous_non_operating_assets", f.previous_non_operating_assets),
            ("minority_interests", f.minority_interests),
        ):
            if amount < 0:
                raise ValueError(f"{name} must be non-negative")
        for name, ratio in (
            ("high_growth_debt_ratio", m.high_growth_debt_ratio),
            ("stable_debt_ratio", a.stable_debt_ratio),
        ):
            if not 0 <= ratio <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if a.stable_growth_rate <= -1 or a.explicit_high_growth_rate <= -1:
            raise ValueError("growth rates cannot be below -100%")
        if a.use_fundamental_stable_reinvestment and a.stable_return_on_capital <= 0:
            raise ValueError("stable_return_on_capital must be positive")
        if (
            not a.use_fundamental_stable_reinvestment
            and a.stable_capex_to_depreciation < 0
        ):
            raise ValueError("stable_capex_to_depreciation must be non-negative")

    @staticmethod
    def _diagnostics(
        inputs: GinzuInput,
        stable_wacc: float,
        high_growth_debt_ratio: float,
    ) -> tuple[str, ...]:
        a, m = inputs.forecast, inputs.market
        messages: list[str] = []
        if a.high_growth_years > 10:
            messages.append(
                "Growth periods longer than 10 years are indicative of extraordinary competitive advantages. What is your company's competitive advantage?"
            )
        if a.stable_growth_rate > m.riskfree_rate:
            messages.append(
                "Stable growth rate > Riskfree rate: Not a good idea. Cap this number at the risk free rate"
            )
        if a.stable_debt_ratio == high_growth_debt_ratio:
            messages.append(
                "Stable growth debt ratio is set to same value as high growth debt ratio. While this is possible, stable growth companies usually carry more debt. Look at the industry average"
            )
        if a.stable_beta > 1.2:
            messages.append(
                "Too high a beta for stable growth. Move down towards 1.00 (cap = 1.20)"
            )
        elif a.stable_beta < 0.8:
            messages.append(
                "Too low a beta for stable growth. Push up towards 1.00 (floor = 0.80)"
            )
        if a.stable_pretax_cost_of_debt > m.riskfree_rate + 0.03:
            messages.append(
                "This is a high cost of debt for stable growth. Consider moving this down."
            )
        elif a.stable_pretax_cost_of_debt < m.riskfree_rate:
            messages.append(
                "Cost of debt is less than the riskfree rate - Not possible"
            )
        if a.stable_tax_rate < 0.35:
            messages.append(
                "If this is a US company, this tax rate is below the marginal tax rate. Move towards a marginal tax rate"
            )
        if a.stable_return_on_capital > stable_wacc + 0.05:
            messages.append(
                "Your return on capital exceeds your cost of capital by more than 5%. That is unusually high. Does your company have that sustainable a competitive advantage?"
            )
        elif a.stable_return_on_capital < stable_wacc:
            messages.append(
                "Your return on capital is less than your cost of capital forever and you are forcing the firm to destroy value in perpetuity. Push up to the cost of capital"
            )
        return tuple(messages)


def value_ginzu(inputs: GinzuInput, /) -> GinzuResult:
    """Value a company through the standalone full-Ginzu calculation path."""
    return DamodaranGinzu().value(inputs)


def excel_1904_date(serial: int | float, /) -> date:
    """Convert the workbook's 1904-epoch date serial to a calendar date."""
    if serial < 0:
        raise ValueError("Excel 1904 date serial cannot be negative")
    return date(1904, 1, 1) + timedelta(days=serial)


__all__ = [
    "BusinessExposure",
    "CostOfCapitalInput",
    "CostOfCapitalResult",
    "CountryReference",
    "DamodaranGinzu",
    "EarningsNormalizationInput",
    "EarningsNormalizationResult",
    "FCFF_GINZU_SOURCE_SHA256",
    "FCFF_GINZU_SOURCE_URL",
    "Financials",
    "ForecastAssumptions",
    "ForecastYear",
    "GeographicExposure",
    "GinzuInput",
    "GinzuReferenceData",
    "GinzuResult",
    "LeaseInput",
    "LeaseResult",
    "MarketInputs",
    "OptionInput",
    "OptionResult",
    "ResearchAndDevelopmentInput",
    "ResearchAndDevelopmentResult",
    "StoryToNumbersResult",
    "SyntheticRatingInput",
    "SyntheticRatingResult",
    "excel_1904_date",
    "value_ginzu",
]
