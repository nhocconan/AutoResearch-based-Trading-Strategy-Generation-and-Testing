# Backtesting Rules

## Execution Model

### Fill Delay
- Signal at bar `t` → position change at bar `t+1` open price
- This is enforced by the engine: signal array is shifted by `fill_delay_bars` (default: 1)
- The strategy code does NOT need to handle this - just generate clean signals

### Signal Convention

**CRITICAL: Signal value = position size as fraction of capital.**

| Value | Meaning |
|-------|---------|
| +0.35 | 35% of capital long (recommended max) |
| -0.35 | 35% of capital short |
| +0.20 | 20% of capital long (conservative) |
| 0.0 | Flat (no position) |

**Rules:**
- MAX signal magnitude: **0.40** (40% of capital). Never use 1.0.
- Use **discrete levels** (0.0, ±0.20, ±0.35) to minimize fee churn
- Every signal change triggers a trade with 0.10% round-trip cost
- Signal=1.0 with BTC's 77% crash in 2022 → -77% equity. Signal=0.35 → only -27%.

### Position Changes
- Position changes happen at bar open price
- Going from +1 to -1 = close long + open short (2x cost)
- Going from +1 to 0 = close long (1x cost)

## Cost Model

### Fees
- Taker fee: 0.04% per side (Binance USDT-M futures standard)
- Slippage: 0.01% per side (conservative for liquid pairs)
- **Total: 0.05% per side, 0.10% round trip**

### Funding Rate
- Applied every 8 hours (00:00, 08:00, 16:00 UTC)
- Long position pays positive rate, receives negative rate
- Short position receives positive rate, pays negative rate
- Rate applied to: `position_size * rate * leverage`
- Source: Binance historical funding rate data

### Leverage
- Configurable per strategy via `strategy.leverage`
- Default: 1x (no leverage)
- Maximum: 20x (hard cap in engine)
- Affects both PnL and costs proportionally

## No Look-Ahead Bias

### What's Enforced by Engine
- Signal shift (t+1 fill)
- Funding rate timing (only applies at actual funding times)

### What the Strategy Must Ensure
- `generate_signals()` at index `i` must only use data from `0:i+1`
- No `.shift(-n)` (negative shifts look into future)
- All rolling windows use only past data
- No future information in feature engineering

## Capital and Risk

- Initial capital: $10,000 (configurable)
- Bankruptcy: if equity hits 0, simulation stops
- No margin calls modeled (simplified)
- Position size is fraction of current equity

## Evaluation Metrics

### Primary: Sharpe Ratio
- Annualized, excess return over risk-free rate (5%)
- Must be > 0 to be kept (target > 1.0)

### Secondary
- Sortino Ratio (downside risk only)
- Calmar Ratio (return / max drawdown)
- Return/DD ratio (exceptional if > 10 with Sharpe > 2)

### Auto-Rejection Thresholds
| Metric | Threshold | Reason |
|--------|-----------|--------|
| Max drawdown | > -50% avg across symbols | Too risky — capital preservation |
| Trades | ≥ 10 | Trivial strategies don't count |
| Sharpe | > 0 | Must beat doing nothing |
| SOL-only | Must work across BTC/ETH/SOL | SOL rally 2021-2024 creates bias |
| 1m timeframe | Banned | Too noisy, excessive costs |
| Backtest time | < 120s per symbol | Prevents hung strategies |

## Compliance Validation

The [validator](../validator.py) checks every strategy for:
1. **No lookahead** — scans for `.shift(-n)`, future index access
2. **Required fields** — name, timeframe, leverage, generate_signals
3. **Leverage bounds** — max 5.0x (warning above 3.0x)
4. **Valid timeframe** — must be one of 5m, 15m, 1h, 4h, 1d
5. **Syntax validity** — AST parsing

Strategies failing compliance are auto-rejected before backtest.

## Last Updated
2026-03-21
