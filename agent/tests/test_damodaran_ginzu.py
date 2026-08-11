"""Golden parity for the standalone Damodaran FCFF Ginzu model."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from src.valuation import (
    BusinessExposure,
    CostOfCapitalInput,
    DamodaranGinzu,
    EarningsNormalizationInput,
    FCFF_GINZU_SOURCE_SHA256,
    FCFF_GINZU_SOURCE_URL,
    Financials,
    ForecastAssumptions,
    GeographicExposure,
    GinzuReferenceData,
    GinzuInput,
    LeaseInput,
    MarketInputs,
    OptionInput,
    ResearchAndDevelopmentInput,
    SyntheticRatingInput,
    value_ginzu,
    excel_1904_date,
)


def test_auxiliary_sheets_match_official_cached_outputs() -> None:
    model = DamodaranGinzu()

    normalized = model.normalize_earnings(
        EarningsNormalizationInput(
            approach="sector_margin",
            current_revenue=34_124,
            debt=25_226,
            equity=10_864,
            historical_average_ebit=3_500,
            historical_average_roc=0.22,
            sector_margin=2_000 / 13_590,
        )
    )
    assert normalized.normalized_ebit == pytest.approx(5_021.927888153054)

    rd = model.capitalize_rd(
        ResearchAndDevelopmentInput(
            amortization_years=10,
            current_expense=13_113,
            historical_expenses=(
                8_099,
                7_901,
                6_746,
                5_835,
                7_035,
                6_209,
                5_340,
                5_331,
                4_934,
                5_588,
            ),
            marginal_tax_rate=0.25,
        )
    )
    assert rd.research_asset == pytest.approx(44_108.8)
    assert rd.current_amortization == pytest.approx(6_301.8)
    assert rd.operating_income_adjustment == pytest.approx(6_811.2)
    assert rd.tax_effect == pytest.approx(1_702.8)

    lease = model.capitalize_leases(
        LeaseInput(
            current_lease_expense=504,
            commitments=(453, 371, 265, 180, 122),
            beyond_year_five=259,
            pretax_cost_of_debt=0.0509,
        )
    )
    assert lease.embedded_years == 1
    assert lease.lease_debt == pytest.approx(1_430.3604083814394)
    assert lease.depreciation == pytest.approx(238.39340139690657)
    assert lease.operating_income_adjustment == pytest.approx(265.6065986030934)


def test_rating_option_and_cost_of_capital_match_official_cached_outputs() -> None:
    model = DamodaranGinzu()

    rating = model.estimate_rating(
        SyntheticRatingInput(
            firm_size="large",
            ebit=7_040,
            interest_expense=486,
            riskfree_rate=0.045,
            country_default_spread=0.0023,
        )
    )
    assert rating.interest_coverage_ratio == pytest.approx(14.48559670781893)
    assert rating.rating == "Aaa/AAA"
    assert rating.company_default_spread == pytest.approx(0.004)
    assert rating.cost_of_debt == pytest.approx(0.0513)

    options = model.value_options(
        OptionInput(
            stock_price=804.14,
            strike_price=40.35,
            maturity_years=8.3,
            volatility=0.25,
            dividend_yield=0,
            riskfree_rate=0.045,
            option_count=50.998,
            share_count=950.41,
        )
    )
    assert options.adjusted_stock_price == pytest.approx(802.6496913350761)
    assert options.value_per_option == pytest.approx(774.8759696552005)
    assert options.total_value == pytest.approx(39_517.12470047591)

    coc = model.cost_of_capital(
        CostOfCapitalInput(
            shares=950.41,
            stock_price=804.14,
            unlevered_beta=0.915135973481584,
            riskfree_rate=0.045,
            equity_risk_premium=0.04921860682334804,
            straight_debt_book_value=25_226,
            straight_debt_interest=486,
            straight_debt_maturity=3,
            pretax_cost_of_debt=0.0365,
            marginal_tax_rate=0.25,
        )
    )
    assert coc.market_value_straight_debt == pytest.approx(24_011.472745307412)
    assert coc.levered_beta == pytest.approx(0.9366996608369831)
    assert coc.cost_of_equity == pytest.approx(0.09110305231829893)
    assert coc.cost_of_capital == pytest.approx(0.08916184398076038)

    with pytest.raises(ValueError, match="firm_size"):
        model.estimate_rating(
            replace(
                SyntheticRatingInput("large", 7_040, 486, 0.045),
                firm_size="unsupported",  # type: ignore[arg-type]
            )
        )


def test_reference_data_calculators_and_ttm_match_workbook() -> None:
    model = DamodaranGinzu()
    erp = model.weighted_erp(
        (
            GeographicExposure("Asia", 7_440, 0.057223235648452254),
            GeographicExposure("North America", 21_791, 0.04446474878479404),
            GeographicExposure("Western Europe", 6_175, 0.052665467678857734),
            GeographicExposure("Japan", 1_673, 0.0563),
            GeographicExposure("China", 1_540, 0.0563),
        )
    )
    assert erp == pytest.approx(0.04921860682334804)

    beta = model.bottom_up_beta(
        (
            BusinessExposure(
                "Drugs (Pharmaceutical)", 84, 6.239336284469536, 0.915135973481584
            ),
        )
    )
    assert beta == pytest.approx(0.915135973481584)
    assert model.trailing_twelve_months(5_089, 2_242, 3_271) == pytest.approx(6_118)


def test_builtin_reference_snapshot_matches_workbook_lookup_ranges() -> None:
    data = GinzuReferenceData.builtin()
    assert data.as_of == "2026-01-01"
    assert data.source_url == FCFF_GINZU_SOURCE_URL
    assert data.source_sha256 == FCFF_GINZU_SOURCE_SHA256
    assert data.mature_market_erp == pytest.approx(0.0423)
    assert data.country("China").equity_risk_premium == pytest.approx(0.0514)
    assert data.country("United States").adjusted_default_spread == pytest.approx(
        0.0023
    )
    drugs = data.industry("Drugs (Pharmaceutical)", market="us")
    assert drugs["Unlevered Beta"] == pytest.approx(0.915135973481584)
    assert drugs["EV/Sales"] == pytest.approx(6.239336284469536)
    assert data.rd_amortization_years("Drug") == 10
    with pytest.raises(KeyError, match="Zimbabwe"):
        data.country("Zimbabwe")


def test_excel_1904_date_and_private_company_branch() -> None:
    assert excel_1904_date(43_951) == date(2024, 5, 1)
    case = _eli_lilly_case()
    private_market = replace(
        case.market,
        stock_traded=False,
        current_price=None,
        keep_current_debt_ratio=False,
    )
    result = value_ginzu(replace(case, market=private_market))
    assert result.current_price is None
    assert result.price_to_value_minus_one is None
    assert result.value_per_share > 0

    book_ratio_market = replace(private_market, keep_current_debt_ratio=True)
    changed_unused_market_debt = replace(book_ratio_market, market_value_debt=100_000)
    book = value_ginzu(replace(case, market=book_ratio_market))
    changed = value_ginzu(replace(case, market=changed_unused_market_debt))
    assert book.value_per_share == pytest.approx(changed.value_per_share)
    assert book.debt_value == pytest.approx(case.financials.current_book_debt)

    invalid_ratio = replace(
        private_market,
        keep_current_debt_ratio=True,
    )
    invalid_case = replace(
        case,
        market=invalid_ratio,
        financials=replace(
            case.financials,
            current_book_debt=100,
            current_book_equity=-50,
        ),
    )
    with pytest.raises(ValueError, match="computed high_growth_debt_ratio"):
        value_ginzu(invalid_case)


def test_current_debt_ratio_follows_the_option_and_lease_branch() -> None:
    case = _eli_lilly_case()
    option = OptionInput(
        stock_price=1,
        strike_price=250,
        maturity_years=5,
        volatility=0.30,
        option_count=50,
        share_count=1,
        riskfree_rate=0.01,
        dividend_yield=0.01,
    )
    market = replace(case.market, keep_current_debt_ratio=True)
    result = value_ginzu(replace(case, market=market, option=option))
    assert market.current_price is not None
    debt_ratio = market.market_value_debt / (
        market.market_value_debt + market.current_price * market.shares
    )
    expected_wacc = (
        market.riskfree_rate + market.high_growth_beta * market.equity_risk_premium
    ) * (1 - debt_ratio) + market.pretax_cost_of_debt * (
        1 - case.financials.marginal_tax_rate
    ) * debt_ratio
    assert result.high_growth_cost_of_capital == pytest.approx(expected_wacc)


def _eli_lilly_case() -> GinzuInput:
    return GinzuInput(
        company_name="Eli Lilly",
        financials=Financials(
            operating_income=7_040,
            interest_expense=486,
            capital_spending=3_448,
            depreciation=1_527,
            effective_tax_rate=1_314.2 / 6_554.6,
            marginal_tax_rate=0.25,
            current_revenue=34_124,
            previous_revenue=28_541,
            current_non_cash_working_capital=2_411,
            change_in_working_capital=2_225,
            current_book_debt=25_226,
            previous_book_debt=16_239,
            current_book_equity=10_864,
            previous_book_equity=10_775,
            current_cash=2_928,
            previous_cash=2_212,
            current_non_operating_assets=3_052,
            previous_non_operating_assets=2_902,
            minority_interests=92,
        ),
        market=MarketInputs(
            current_price=804.14,
            shares=950.41,
            market_value_debt=25_226,
            riskfree_rate=0.045,
            equity_risk_premium=0.04921860682334804,
            high_growth_beta=0.9366996608369831,
            high_growth_debt_ratio=0.03195232570532813,
            pretax_cost_of_debt=0.0509,
        ),
        forecast=ForecastAssumptions(
            high_growth_years=10,
            use_fundamental_growth=True,
            high_growth_return_on_capital=0.2042515237104207,
            high_growth_reinvestment_rate=0.8811285024606358,
            explicit_high_growth_rate=0.15,
            adjust_second_half=True,
            working_capital_ratio=2_411 / 34_124,
            stable_growth_rate=0.04,
            stable_beta=1,
            stable_equity_risk_premium=0.04921860682334804,
            stable_debt_ratio=0.2,
            stable_pretax_cost_of_debt=0.0509,
            stable_tax_rate=0.25,
            use_fundamental_stable_reinvestment=True,
            stable_return_on_capital=0.15,
            stable_capex_to_depreciation=1.2,
        ),
        rd=ResearchAndDevelopmentInput(
            amortization_years=10,
            current_expense=13_113,
            historical_expenses=(
                8_099,
                7_901,
                6_746,
                5_835,
                7_035,
                6_209,
                5_340,
                5_331,
                4_934,
                5_588,
            ),
            marginal_tax_rate=0.25,
        ),
    )


def test_full_ginzu_eli_lilly_case_matches_official_workbook() -> None:
    case = _eli_lilly_case()
    result = value_ginzu(case)
    assert result == DamodaranGinzu().value(case)

    assert result.adjusted_ebit == pytest.approx(13_851.2)
    assert result.current_forecast_ebit == pytest.approx(15_554)
    assert result.high_growth_rate == pytest.approx(0.17997183921226603)
    assert result.high_growth_cost_of_capital == pytest.approx(0.08941187795167599)
    assert [year.growth_rate for year in result.forecast[:6]] == pytest.approx(
        [0.17997183921226603] * 5 + [0.15197747136981282]
    )
    assert result.forecast[0].fcff == pytest.approx(1_733.4547153119845)
    assert result.forecast[9].fcff == pytest.approx(30_844.31704596279)
    assert result.terminal_fcff == pytest.approx(32_078.089727801304)
    assert result.terminal_value == pytest.approx(745_830.6244181979)
    assert result.pv_high_growth_fcff == pytest.approx(55_377.95751555063)
    assert result.pv_terminal_value == pytest.approx(322_397.8761421915)
    assert result.operating_asset_value == pytest.approx(377_775.8336577421)
    assert result.equity_value_common == pytest.approx(358_437.8336577421)
    assert result.value_per_share == pytest.approx(377.1402170197516)
    assert result.price_to_value_minus_one == pytest.approx(1.1322043200656204)
    story = DamodaranGinzu().story_to_numbers(result)
    assert story.assets_in_place == pytest.approx(187_375.27360813718)
    assert story.value_of_growth == pytest.approx(190_400.56004960494)
    assert story.value_of_growth_share == pytest.approx(0.5040040761900728)


def test_full_ginzu_eli_lilly_case_matches_all_forecast_rows() -> None:
    result = DamodaranGinzu().value(_eli_lilly_case())
    expected = {
        "growth_rate": (
            0.17997183921226603,
            0.17997183921226603,
            0.17997183921226603,
            0.17997183921226603,
            0.17997183921226603,
            0.15197747136981282,
            0.12398310352735961,
            0.0959887356849064,
            0.0679943678424532,
            0.04,
        ),
        "cumulative_growth": (
            1.179971839212266,
            1.3923335413339777,
            1.6429143695647812,
            1.9385926903236155,
            2.2874847822846114,
            2.6351309352931533,
            2.961842646851752,
            3.246146177820689,
            3.4668658351058026,
            3.6055404685100347,
        ),
        "reinvestment_rate": (
            0.8811285024606358,
            0.8811285024606358,
            0.8811285024606358,
            0.8811285024606358,
            0.8811285024606357,
            0.7582361353018419,
            0.635343768143048,
            0.5124514009842542,
            0.3895590338254605,
            0.26666666666666666,
        ),
        "ebit": (
            18353.281987107584,
            21656.35590190869,
            25553.890104210608,
            30152.870705293513,
            35579.538303654845,
            40986.8265675497,
            46068.50052913215,
            50490.55764982299,
            53923.63119923565,
            56080.57644720508,
        ),
        "tax_rate": (
            0.205450370732005,
            0.21040032953956,
            0.215350288347115,
            0.22030024715467,
            0.225250205962225,
            0.23020016476978,
            0.235150123577335,
            0.24010008238489,
            0.245050041192445,
            0.25,
        ),
        "nopat": (
            14582.593398707302,
            17099.851483521106,
            20050.852501878366,
            23510.185836494544,
            27565.239972715717,
            31551.652338309363,
            35235.4869366842,
            38367.77059844146,
            40709.64315261674,
            42060.432335403806,
        ),
        "net_capital_expenditures": (
            12415.226579054544,
            14555.162466158788,
            17063.227261174707,
            20002.614407332137,
            23447.339783409094,
            23085.42805648342,
            21598.945106140305,
            18976.161982647696,
            15326.654160198243,
            10881.770748303412,
        ),
        "change_in_working_capital": (
            433.91210434077334,
            512.0040638154469,
            604.1503768644673,
            712.8804313495493,
            841.1788337179611,
            838.1748749035947,
            787.7019365677813,
            685.4558131661065,
            532.1550937144093,
            334.3445411376037,
        ),
        "fcff": (
            1733.4547153119845,
            2032.684953546871,
            2383.474863839192,
            2794.690997812857,
            3276.7213555886624,
            7628.049406922349,
            12848.839893976117,
            18706.152802627654,
            24850.83389870409,
            30844.31704596279,
        ),
        "cost_of_capital": (
            0.08941187795167599,
            0.08941187795167599,
            0.08941187795167599,
            0.08941187795167599,
            0.08941187795167599,
            0.08813147945307648,
            0.08685108095447697,
            0.08557068245587746,
            0.08429028395727794,
            0.08300988545867843,
        ),
        "cumulative_cost_of_capital": (
            1.089411877951676,
            1.1868182398221974,
            1.2929338874320024,
            1.4085375343746585,
            1.53447752048852,
            1.6697132945566617,
            1.8147296990729687,
            1.9700173578955917,
            2.1360706803933773,
            2.313385662904473,
        ),
        "present_value": (
            1591.18396852001,
            1712.7179928169933,
            1843.462289145502,
            1984.1082893496355,
            2135.3987346425756,
            4568.4785716147335,
            7080.3050727277905,
            9495.425371586525,
            11633.900566496039,
            13332.97665865082,
        ),
    }
    for field, values in expected.items():
        assert [getattr(year, field) for year in result.forecast] == pytest.approx(
            values
        )


def test_stable_growth_and_fifteen_year_paths_are_supported() -> None:
    case = _eli_lilly_case()
    zero_year = GinzuInput(
        company_name=case.company_name,
        financials=case.financials,
        market=case.market,
        forecast=ForecastAssumptions(
            **{**case.forecast.__dict__, "high_growth_years": 0}
        ),
        rd=case.rd,
    )
    zero = DamodaranGinzu().value(zero_year)
    assert zero.forecast == ()
    assert zero.pv_high_growth_fcff == 0
    assert zero.pv_terminal_value == pytest.approx(zero.terminal_value)

    fifteen_year = GinzuInput(
        company_name=case.company_name,
        financials=case.financials,
        market=case.market,
        forecast=ForecastAssumptions(
            **{**case.forecast.__dict__, "high_growth_years": 15}
        ),
        rd=case.rd,
    )
    assert len(DamodaranGinzu().value(fifteen_year).forecast) == 15


def test_zero_fundamental_drivers_are_not_replaced_by_calculated_defaults() -> None:
    case = _eli_lilly_case()
    assumptions = replace(
        case.forecast,
        high_growth_return_on_capital=0,
        high_growth_reinvestment_rate=0,
    )
    result = DamodaranGinzu().value(replace(case, forecast=assumptions))
    assert result.high_growth_return_on_capital == 0
    assert result.high_growth_reinvestment_rate == 0
    assert result.high_growth_rate == 0


def test_explicit_high_growth_drivers_do_not_evaluate_unused_denominators() -> None:
    case = _eli_lilly_case()
    zero_invested_capital = replace(
        case,
        rd=None,
        financials=replace(
            case.financials,
            previous_book_debt=0,
            previous_book_equity=0,
            previous_cash=0,
            previous_non_operating_assets=0,
        ),
    )
    assert value_ginzu(zero_invested_capital).value_per_share > 0

    zero_current_nopat = replace(
        case,
        financials=replace(case.financials, effective_tax_rate=1),
    )
    assert value_ginzu(zero_current_nopat).value_per_share > 0


def test_integrated_model_uses_master_financial_and_market_facts() -> None:
    case = _eli_lilly_case()
    normalization = EarningsNormalizationInput(
        approach="sector_margin",
        current_revenue=1,
        debt=2,
        equity=3,
        sector_margin=0.1,
    )
    normalized = value_ginzu(replace(case, normalization=normalization))
    assert normalized.normalized_ebit == pytest.approx(
        case.financials.current_revenue * 0.1
    )

    canonical_option = OptionInput(
        stock_price=case.market.current_price or 1,
        strike_price=40.35,
        maturity_years=8.3,
        volatility=0.25,
        dividend_yield=0,
        riskfree_rate=case.market.riskfree_rate,
        option_count=50.998,
        share_count=case.market.shares,
    )
    contradictory_option = replace(
        canonical_option,
        stock_price=1,
        riskfree_rate=0.99,
        share_count=1,
    )
    canonical = value_ginzu(replace(case, option=canonical_option))
    contradictory = value_ginzu(replace(case, option=contradictory_option))
    assert contradictory.option_value_after_tax == pytest.approx(
        canonical.option_value_after_tax
    )


def test_explicit_growth_capex_and_estimated_option_branches_run() -> None:
    case = _eli_lilly_case()
    assumptions = replace(
        case.forecast,
        use_fundamental_growth=False,
        explicit_high_growth_rate=0.1,
        adjust_second_half=False,
        use_fundamental_stable_reinvestment=False,
    )
    market = replace(case.market, keep_current_debt_ratio=True)
    option = OptionInput(
        stock_price=case.market.current_price,
        strike_price=40.35,
        maturity_years=8.3,
        volatility=0.25,
        dividend_yield=0,
        riskfree_rate=case.market.riskfree_rate,
        option_count=50.998,
        share_count=case.market.shares,
        use_estimated_value=True,
    )
    result = DamodaranGinzu().value(
        replace(case, forecast=assumptions, market=market, option=option)
    )
    assert [year.growth_rate for year in result.forecast] == pytest.approx([0.1] * 10)
    assert result.option_value_after_tax > 0
    assert result.equity_value_common < result.equity_value


def test_synthetic_rating_and_lease_circularity_converges() -> None:
    case = _eli_lilly_case()
    lease_input = LeaseInput(
        current_lease_expense=504,
        commitments=(453, 371, 265, 180, 122),
        beyond_year_five=259,
        pretax_cost_of_debt=0.0509,
    )
    market = replace(
        case.market,
        use_synthetic_rating=True,
        rating_firm_size="large",
        country_default_spread=0.0023,
    )
    result = value_ginzu(replace(case, market=market, lease=lease_input))
    lease = DamodaranGinzu().capitalize_leases(
        replace(lease_input, pretax_cost_of_debt=result.pretax_cost_of_debt)
    )
    rating = DamodaranGinzu().estimate_rating(
        SyntheticRatingInput(
            firm_size="large",
            ebit=case.financials.operating_income + lease.operating_income_adjustment,
            interest_expense=case.financials.interest_expense
            + lease.lease_debt * result.pretax_cost_of_debt,
            riskfree_rate=case.market.riskfree_rate,
            country_default_spread=0.0023,
        )
    )
    assert result.synthetic_rating == rating.rating
    assert result.pretax_cost_of_debt == pytest.approx(rating.cost_of_debt)


def test_model_rejects_nonfinite_financial_inputs() -> None:
    case = _eli_lilly_case()
    bad = replace(case, financials=replace(case.financials, current_cash=float("nan")))
    with pytest.raises(ValueError, match="current_cash must be finite"):
        value_ginzu(bad)

    with pytest.raises(ValueError, match="current_expense must be finite"):
        DamodaranGinzu().capitalize_rd(
            ResearchAndDevelopmentInput(1, float("nan"), (1,), 0.25)
        )


def test_model_rejects_invalid_ratios_and_nonfinite_derived_outputs() -> None:
    case = _eli_lilly_case()
    with pytest.raises(ValueError, match="high_growth_debt_ratio"):
        value_ginzu(
            replace(case, market=replace(case.market, high_growth_debt_ratio=1.2))
        )
    with pytest.raises(ValueError, match="stable_debt_ratio"):
        value_ginzu(
            replace(
                case,
                forecast=replace(case.forecast, stable_debt_ratio=-0.1),
            )
        )
    with pytest.raises(ValueError, match="marginal_tax_rate"):
        value_ginzu(
            replace(
                case,
                financials=replace(case.financials, marginal_tax_rate=1),
            )
        )
    with pytest.raises(ValueError, match="market_value_debt"):
        value_ginzu(replace(case, market=replace(case.market, market_value_debt=-1)))
    with pytest.raises(ValueError, match="pretax_cost_of_debt"):
        value_ginzu(
            replace(case, market=replace(case.market, pretax_cost_of_debt=-0.01))
        )
    with pytest.raises(ValueError, match="stable_pretax_cost_of_debt"):
        value_ginzu(
            replace(
                case,
                forecast=replace(case.forecast, stable_pretax_cost_of_debt=-0.01),
            )
        )
    with pytest.raises(ValueError, match="stable_capex_to_depreciation"):
        value_ginzu(
            replace(
                case,
                forecast=replace(
                    case.forecast,
                    use_fundamental_stable_reinvestment=False,
                    stable_capex_to_depreciation=-1,
                ),
            )
        )
    overflowing = replace(
        case,
        forecast=replace(
            case.forecast,
            use_fundamental_growth=False,
            explicit_high_growth_rate=1e308,
            adjust_second_half=False,
        ),
    )
    with pytest.raises(ValueError, match="must be finite"):
        value_ginzu(overflowing)


def test_diagnostics_follow_official_f14_f23_contract() -> None:
    case = _eli_lilly_case()
    result = value_ginzu(case)
    assert result.diagnostics == (
        "If this is a US company, this tax rate is below the marginal tax rate. Move towards a marginal tax rate",
        "Your return on capital exceeds your cost of capital by more than 5%. That is unusually high. Does your company have that sustainable a competitive advantage?",
    )
    assert not any("Price/value" in message for message in result.diagnostics)

    near = value_ginzu(
        replace(
            case,
            forecast=replace(
                case.forecast,
                stable_debt_ratio=case.market.high_growth_debt_ratio + 5e-13,
            ),
        )
    )
    assert not any(
        "same value as the high growth debt ratio" in message
        for message in near.diagnostics
    )


def test_auxiliary_denominators_and_business_weights_fail_closed() -> None:
    model = DamodaranGinzu()
    cost = model.cost_of_capital(
        CostOfCapitalInput(
            shares=10,
            stock_price=100,
            unlevered_beta=1,
            riskfree_rate=0.03,
            equity_risk_premium=0.05,
            marginal_tax_rate=0.25,
            straight_debt_book_value=500,
            straight_debt_interest=50,
            straight_debt_maturity=0,
            pretax_cost_of_debt=0.05,
        )
    )
    assert cost.market_value_straight_debt == 500

    with pytest.raises(ValueError, match="revenue and enterprise_value_to_sales"):
        model.bottom_up_beta(
            (
                BusinessExposure(
                    "invalid",
                    revenue=-100,
                    enterprise_value_to_sales=-2,
                    unlevered_beta=1.25,
                ),
            )
        )

    case = _eli_lilly_case()
    result = value_ginzu(case)
    with pytest.raises(ValueError, match="stable_cost_of_capital"):
        model.story_to_numbers(replace(result, stable_cost_of_capital=0))
    with pytest.raises(ValueError, match="operating_asset_value"):
        model.story_to_numbers(replace(result, operating_asset_value=0))

    rounded = model.story_to_numbers(
        replace(
            result,
            current_forecast_ebit=1e308,
            stable_cost_of_capital=1,
            operating_asset_value=1,
        )
    )
    assert rounded.assets_in_place_share == 1e308
    assert rounded.value_of_growth_share == -1e308


def test_valuation_modules_stay_below_repository_line_cap() -> None:
    for relative in (
        "agent/src/valuation/damodaran_ginzu.py",
        "agent/src/valuation/_ginzu_types.py",
        "agent/src/valuation/_ginzu_adjustments.py",
    ):
        assert len(Path(relative).read_text().splitlines()) <= 800, relative


def test_model_is_standalone_and_not_an_mcp_or_agent_tool() -> None:
    from src.agent.tools import BaseTool

    assert not issubclass(DamodaranGinzu, BaseTool)
    assert FCFF_GINZU_SOURCE_URL.endswith("/fcffginzu.xlsx")
    assert (
        FCFF_GINZU_SOURCE_SHA256
        == "df8f9e91cc53afe2bdb33cf39b8f2719206ac03b6a5031f1858f82235ea944c9"
    )
    assert "damodaran" not in Path("agent/mcp_server.py").read_text().lower()
    assert (
        "damodaran_fcff"
        not in Path("agent/src/tools/financial_rigor_tool.py").read_text()
    )
