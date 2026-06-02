# Optional: Live Market Data via MCP (forward context only)

> **TL;DR** — You can connect [Financial Datasets MCP](https://docs.financialdatasets.ai/mcp-server#claude-code)
> to Claude Code for **live** stock/crypto prices and financial statements. It is
> **optional** and lives entirely *outside* the backtest. It must **never** feed
> `strategy.py`, the engine, or train/test data — that would break the project's
> reproducibility and no-look-ahead guarantees.

## Why this is opt-in and quarantined

This project's research loop is built on **offline, reproducible Binance data**
(parquet via [`prepare.py`](../prepare.py) / [`update_data.py`](../update_data.py)),
with a strict 2021–2024 train / 2025+ test split and a no-look-ahead engine. That
discipline is the entire point — see [Karpathy alignment](karpathy-autoresearch-alignment.md)
and [backtesting rules](backtesting-rules.md).

A live-data MCP is the opposite of reproducible: every call returns *now*, and
"now" leaks the future into any backtest that touches it. So the rule is simple:

| Use it for | Never use it for |
|------------|------------------|
| Ad-hoc analyst questions in a Claude Code session | Generating or tuning `strategy.py` |
| Forward / paper-trading context (what is the market doing *today*) | Any input to `backtest.py` / `evaluate.py` |
| Cross-asset / macro sanity checks while reading results | The 2021–2024 train or 2025+ test data |
| Sourcing an *idea*, then validating it offline on Binance parquet | Anything that lands in `results.db` / `results.tsv` |

Note also the coverage gap: this project trades **BTC/ETH/SOL USDT-M perpetual
futures** with funding rates at 5m–1w klines. The MCP provides **spot** crypto
prices and equities fundamentals — useful for *context*, not a substitute for the
Binance perp data the engine consumes.

## Setup (≈60 seconds)

The MCP is configured in **your** Claude Code, not committed to this repo (so it
never silently becomes a dependency for contributors or CI).

```bash
# 1. Add the server
claude mcp add --transport http financial-datasets https://mcp.financialdatasets.ai

# 2. Authenticate — run /mcp inside Claude Code and complete OAuth in the browser

# 3. Verify
claude mcp list
```

## Example prompts (analyst / forward context)

```
"How has BTC spot moved this week vs. the equity risk backdrop (SPY, QQQ)?"
"Pull the last 4 quarters of COIN's income statement — is exchange volume rolling over?"
"What's the current crypto-equity correlation regime?"
```

Treat the answers as **context for a human decision**, never as backtest input.

## If you want it shared with collaborators

Prefer documenting the opt-in `claude mcp add` command (as above) over committing a
project-scoped `.mcp.json`, because `.mcp.json` auto-prompts every contributor and
CI to enable an external, authenticated, paid service. Keep the research loop
self-contained and offline by default.

## Hard boundary (restating, because it matters)

The compliance validator and the immutable engine cannot see how a strategy *idea*
was sourced — that responsibility is yours. If a hypothesis came from live data,
it still only counts once it survives an **offline, reproducible, no-look-ahead,
cost-aware** backtest on the Binance parquet. The MCP changes how you *think*; it
must not change how the system *measures*.
