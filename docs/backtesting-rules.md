# Backtesting Rules

## Execution Model

### Fill Delay
- Signal at bar `t` → position change at bar `t+1` open price
- This is enforced by the engine: signal array is shifted by `fill_delay_bars` (default: 1)
- The strategy code does NOT need to handle this - just generate clean signals

### Signal Convention
| Value | Meaning |
|-------|---------|
| +1.0 | Full long position |
| -1.0 | Full short position |
| 0.0 | Flat (no position) |
| 0.5 | Half long position |
| -0.5 | Half short position |

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
- Must be > 0.5 to be considered viable

### Secondary
- Sortino Ratio (downside risk only)
- Calmar Ratio (return / max drawdown)
- Maximum Drawdown (< -30% is disqualifying)
- Win Rate (> 40% preferred)
- Profit Factor (gross wins / gross losses, > 1.0 required)

## Last Updated
2026-03-20
