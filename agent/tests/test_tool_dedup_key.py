"""Duplicate-suppression must key on tool name AND arguments.

The agent loop blocks a repeat call to a non-repeatable tool that already
succeeded. Keying that block on the tool name alone meant the first success
suppressed every later call regardless of arguments, so one run could only
ever fetch a single symbol: get_stock_profile("AAPL") silently blocked
get_stock_profile("NVDA"), and the model was told to "use the previous
result" for data it had never received.
"""

from __future__ import annotations

from src.agent.loop import _dedup_key


class _Call:
    """Minimal stand-in for ToolCallRequest (id/name/arguments)."""

    def __init__(self, name: str, arguments: dict) -> None:
        self.name = name
        self.arguments = arguments


def test_same_tool_different_arguments_are_distinct() -> None:
    """The regression: different symbols must not collide."""
    aapl = _dedup_key(_Call("get_stock_profile", {"symbol": "AAPL"}))
    nvda = _dedup_key(_Call("get_stock_profile", {"symbol": "NVDA"}))

    assert aapl != nvda

    # Simulate the loop's guard: AAPL already succeeded.
    called_ok = {aapl}
    assert nvda not in called_ok, "second symbol would be wrongly blocked"


def test_identical_call_is_suppressed() -> None:
    """The original intent must survive: an exact repeat is still blocked."""
    first = _dedup_key(_Call("get_stock_profile", {"symbol": "AAPL"}))
    repeat = _dedup_key(_Call("get_stock_profile", {"symbol": "AAPL"}))

    assert first == repeat
    assert repeat in {first}


def test_key_is_insensitive_to_argument_order() -> None:
    """Dict ordering must not create a spurious cache miss."""
    a = _dedup_key(_Call("get_market_data", {"symbol": "AAPL", "period": "1d"}))
    b = _dedup_key(_Call("get_market_data", {"period": "1d", "symbol": "AAPL"}))

    assert a == b


def test_different_tools_never_collide() -> None:
    profile = _dedup_key(_Call("get_stock_profile", {"symbol": "AAPL"}))
    news = _dedup_key(_Call("get_stock_news", {"symbol": "AAPL"}))

    assert profile != news


def test_key_is_hashable_and_survives_unserializable_arguments() -> None:
    """A weird argument must not raise inside the agent loop."""
    key = _dedup_key(_Call("some_tool", {"when": object()}))

    assert isinstance(key, tuple)
    assert {key}  # hashable


if __name__ == "__main__":  # pragma: no cover - manual self-check
    test_same_tool_different_arguments_are_distinct()
    test_identical_call_is_suppressed()
    test_key_is_insensitive_to_argument_order()
    test_different_tools_never_collide()
    test_key_is_hashable_and_survives_unserializable_arguments()
    print("all dedup key checks passed")
