"""Typed contracts and frozen reference data for the Ginzu model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

FCFF_GINZU_SOURCE_URL = "https://pages.stern.nyu.edu/~adamodar/pc/fcffginzu.xlsx"
FCFF_GINZU_SOURCE_SHA256 = (
    "df8f9e91cc53afe2bdb33cf39b8f2719206ac03b6a5031f1858f82235ea944c9"
)


@dataclass(frozen=True)
class EarningsNormalizationInput:
    approach: Literal["historical_ebit", "historical_roc", "sector_margin"]
    current_revenue: float
    debt: float
    equity: float
    historical_average_ebit: float = 0.0
    historical_average_roc: float = 0.0
    sector_margin: float = 0.0


@dataclass(frozen=True)
class EarningsNormalizationResult:
    normalized_ebit: float


@dataclass(frozen=True)
class ResearchAndDevelopmentInput:
    amortization_years: int
    current_expense: float
    historical_expenses: tuple[float, ...]
    marginal_tax_rate: float


@dataclass(frozen=True)
class ResearchAndDevelopmentResult:
    research_asset: float
    current_amortization: float
    operating_income_adjustment: float
    tax_effect: float


@dataclass(frozen=True)
class LeaseInput:
    current_lease_expense: float
    commitments: tuple[float, float, float, float, float]
    beyond_year_five: float
    pretax_cost_of_debt: float


@dataclass(frozen=True)
class LeaseResult:
    embedded_years: int
    annual_beyond_commitment: float
    present_values: tuple[float, ...]
    lease_debt: float
    depreciation: float
    operating_income_adjustment: float


@dataclass(frozen=True)
class SyntheticRatingInput:
    firm_size: Literal["large", "small"]
    ebit: float
    interest_expense: float
    riskfree_rate: float
    country_default_spread: float = 0.0


@dataclass(frozen=True)
class SyntheticRatingResult:
    interest_coverage_ratio: float
    rating: str
    company_default_spread: float
    country_default_spread: float
    cost_of_debt: float


@dataclass(frozen=True)
class OptionInput:
    stock_price: float
    strike_price: float
    maturity_years: float
    volatility: float
    dividend_yield: float
    riskfree_rate: float
    option_count: float
    share_count: float
    use_estimated_value: bool = False


@dataclass(frozen=True)
class OptionResult:
    adjusted_stock_price: float
    d1: float
    d2: float
    value_per_option: float
    total_value: float
    iterations: int


@dataclass(frozen=True)
class CostOfCapitalInput:
    shares: float
    stock_price: float
    unlevered_beta: float
    riskfree_rate: float
    equity_risk_premium: float
    straight_debt_book_value: float
    straight_debt_interest: float
    straight_debt_maturity: float
    pretax_cost_of_debt: float
    marginal_tax_rate: float
    convertible_debt_book_value: float = 0.0
    convertible_interest: float = 0.0
    convertible_maturity: float = 0.0
    convertible_market_value: float = 0.0
    lease_debt: float = 0.0
    preferred_shares: float = 0.0
    preferred_price: float = 0.0
    preferred_dividend_per_share: float = 0.0


@dataclass(frozen=True)
class CostOfCapitalResult:
    market_value_equity: float
    market_value_straight_debt: float
    debt_value_in_convertible: float
    equity_value_in_convertible: float
    total_debt: float
    preferred_value: float
    levered_beta: float
    cost_of_equity: float
    after_tax_cost_of_debt: float
    cost_of_preferred: float
    equity_weight: float
    debt_weight: float
    preferred_weight: float
    cost_of_capital: float


@dataclass(frozen=True)
class GeographicExposure:
    name: str
    revenue: float
    equity_risk_premium: float


@dataclass(frozen=True)
class BusinessExposure:
    name: str
    revenue: float
    enterprise_value_to_sales: float
    unlevered_beta: float


@dataclass(frozen=True)
class CountryReference:
    name: str
    rating: str
    adjusted_default_spread: float
    equity_risk_premium: float
    country_risk_premium: float
    corporate_tax_rate: float


@dataclass(frozen=True)
class GinzuReferenceData:
    as_of: str
    source_url: str
    source_sha256: str
    mature_market_erp: float
    countries: dict[str, CountryReference]
    industries: dict[str, dict[str, dict[str, float | int | str | None]]]
    rd_periods: dict[str, int]

    @classmethod
    def builtin(cls) -> GinzuReferenceData:
        path = files("src.valuation").joinpath("data/fcffginzu_reference_2026.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source_url") != FCFF_GINZU_SOURCE_URL:
            raise ValueError(
                "Ginzu reference source URL does not match the model contract"
            )
        if payload.get("source_sha256") != FCFF_GINZU_SOURCE_SHA256:
            raise ValueError(
                "Ginzu reference source SHA-256 does not match the model contract"
            )
        countries = {
            name: CountryReference(name=name, **values)
            for name, values in payload["countries"].items()
        }
        return cls(
            as_of=payload["as_of"],
            source_url=payload["source_url"],
            source_sha256=payload["source_sha256"],
            mature_market_erp=payload["mature_market_erp"],
            countries=countries,
            industries=payload["industries"],
            rd_periods=payload["rd_amortization_years"],
        )

    def country(self, name: str) -> CountryReference:
        try:
            return self.countries[name]
        except KeyError as exc:
            raise KeyError(
                f"country not in official workbook lookup range: {name}"
            ) from exc

    def industry(
        self,
        name: str,
        *,
        market: Literal["us", "global"],
    ) -> dict[str, float | int | str | None]:
        try:
            return self.industries[market][name]
        except KeyError as exc:
            raise KeyError(
                f"industry not in official workbook lookup range: {name}"
            ) from exc

    def rd_amortization_years(self, industry: str) -> int:
        try:
            return self.rd_periods[industry]
        except KeyError as exc:
            raise KeyError(
                f"industry not in R&D amortization lookup: {industry}"
            ) from exc


@dataclass(frozen=True)
class Financials:
    operating_income: float
    interest_expense: float
    capital_spending: float
    depreciation: float
    effective_tax_rate: float
    marginal_tax_rate: float
    current_revenue: float
    previous_revenue: float
    current_non_cash_working_capital: float
    change_in_working_capital: float
    current_book_debt: float
    previous_book_debt: float
    current_book_equity: float
    previous_book_equity: float
    current_cash: float
    previous_cash: float
    current_non_operating_assets: float
    previous_non_operating_assets: float
    minority_interests: float = 0.0


@dataclass(frozen=True)
class MarketInputs:
    current_price: float | None
    shares: float
    market_value_debt: float
    riskfree_rate: float
    equity_risk_premium: float
    high_growth_beta: float
    high_growth_debt_ratio: float
    pretax_cost_of_debt: float
    keep_current_debt_ratio: bool = False
    stock_traded: bool = True
    use_synthetic_rating: bool = False
    rating_firm_size: Literal["large", "small"] = "large"
    country_default_spread: float = 0.0


@dataclass(frozen=True)
class ForecastAssumptions:
    high_growth_years: int
    use_fundamental_growth: bool
    high_growth_return_on_capital: float | None
    high_growth_reinvestment_rate: float | None
    explicit_high_growth_rate: float
    adjust_second_half: bool
    working_capital_ratio: float
    stable_growth_rate: float
    stable_beta: float
    stable_equity_risk_premium: float
    stable_debt_ratio: float
    stable_pretax_cost_of_debt: float
    stable_tax_rate: float
    use_fundamental_stable_reinvestment: bool
    stable_return_on_capital: float
    stable_capex_to_depreciation: float


@dataclass(frozen=True)
class GinzuInput:
    company_name: str
    financials: Financials
    market: MarketInputs
    forecast: ForecastAssumptions
    normalization: EarningsNormalizationInput | None = None
    rd: ResearchAndDevelopmentInput | None = None
    lease: LeaseInput | None = None
    option: OptionInput | None = None


@dataclass(frozen=True)
class ForecastYear:
    year: int
    growth_rate: float
    cumulative_growth: float
    reinvestment_rate: float
    ebit: float
    tax_rate: float
    nopat: float
    net_capital_expenditures: float
    change_in_working_capital: float
    fcff: float
    cost_of_capital: float
    cumulative_cost_of_capital: float
    present_value: float


@dataclass(frozen=True)
class GinzuResult:
    company_name: str
    normalized_ebit: float
    adjusted_ebit: float
    adjusted_interest_expense: float
    adjusted_capital_spending: float
    adjusted_depreciation: float
    current_forecast_ebit: float
    high_growth_rate: float
    high_growth_return_on_capital: float
    high_growth_reinvestment_rate: float
    high_growth_cost_of_capital: float
    pretax_cost_of_debt: float
    synthetic_rating: str | None
    stable_reinvestment_rate: float
    stable_cost_of_capital: float
    forecast: tuple[ForecastYear, ...]
    terminal_nopat: float
    terminal_net_capital_expenditures: float
    terminal_change_in_working_capital: float
    terminal_fcff: float
    terminal_value: float
    pv_high_growth_fcff: float
    pv_terminal_value: float
    operating_asset_value: float
    cash_and_non_operating_assets: float
    debt_value: float
    minority_interests: float
    equity_value: float
    option_value_after_tax: float
    equity_value_common: float
    value_per_share: float
    current_price: float | None
    price_to_value_minus_one: float | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class StoryToNumbersResult:
    assets_in_place: float
    value_of_growth: float
    assets_in_place_share: float
    value_of_growth_share: float


_LARGE_RATINGS = (
    (-100_000.0, "D2/D", 0.19),
    (0.2, "Caa/CCC", 0.16),
    (0.65, "Ca2/CC", 0.126101),
    (0.8, "C2/C", 0.0885),
    (1.25, "B3/B-", 0.050899),
    (1.5, "B2/B", 0.032098),
    (1.75, "B1/B+", 0.027531),
    (2.0, "Ba2/BB", 0.018395),
    (2.25, "Ba1/BB+", 0.013828),
    (2.5, "Baa2/BBB", 0.011113),
    (3.0, "A3/A-", 0.008872),
    (4.25, "A2/A", 0.007751),
    (5.5, "A1/A+", 0.007003),
    (6.5, "Aa2/AA", 0.005506),
    (8.5, "Aaa/AAA", 0.004),
)
_SMALL_RATINGS = (
    (-100_000.0, "D2/D", 0.19),
    (0.5, "Caa/CCC", 0.16),
    (0.8, "Ca2/CC", 0.126101),
    (1.25, "C2/C", 0.0885),
    (1.5, "B3/B-", 0.050899),
    (2.0, "B2/B", 0.032098),
    (2.5, "B1/B+", 0.027531),
    (3.0, "Ba2/BB", 0.018395),
    (3.5, "Ba1/BB+", 0.013828),
    (4.0, "Baa2/BBB", 0.011113),
    (4.5, "A3/A-", 0.008872),
    (6.0, "A2/A", 0.007751),
    (7.5, "A1/A+", 0.007003),
    (9.5, "Aa2/AA", 0.005506),
    (12.5, "Aaa/AAA", 0.004),
)
