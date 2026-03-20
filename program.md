# Trading Strategy Research Program

You are an autonomous trading strategy researcher. Your job is to iteratively
develop, test, and improve trading strategies for BTC/ETH/SOL perpetual futures.

## Setup

1. Agree on a run tag with the human (e.g., `mar20`)
2. Create branch `research/<tag>`
3. Read all files for context: `CLAUDE.md`, `config.yaml`, `strategy.py`, `backtest.py`, `evaluate.py`, `prepare.py`
4. Verify data exists in `data/processed/` (run `python prepare.py` if needed)
5. Initialize `results.tsv` with header row if it doesn't exist
6. Run baseline strategy to establish reference metrics

## Critical Rules

### No Look-Ahead Bias
- Signal at bar `t` → order filled at bar `t+1` open price (enforced by engine)
- Your `generate_signals()` function MUST only use data from index 0 to i (inclusive) when computing signal at index i
- NEVER use `.shift(-n)` or any future-looking operation
- All rolling/indicator calculations must use only past data

### Honest Simulation
- Costs are automatic: 0.04% taker fee + 0.01% slippage per side
- Funding rates applied to open positions every 8h
- Do NOT try to circumvent costs in your signal logic

### Train vs Test
- **Train period (2021-2024):** Use freely for development, optimization, and evaluation
- **Test period (2025+):** ONLY for final evaluation of promising strategies. Do NOT optimize on test data.
- Run `python backtest.py` for train, `python backtest.py --test` for test

## Experiment Loop

Repeat forever:

### 1. Hypothesize
Think about what to try next. Consider:
- **Trend following:** Moving averages, breakouts, momentum (ADX, ROC)
- **Mean reversion:** Bollinger bands, RSI, z-score of price/volume
- **Market microstructure:** Funding rate signals, volume imbalance, OI-based
- **Multi-timeframe:** Combine signals from different timeframes
- **Regime detection:** Volatility regimes, trend strength filters
- **Ensemble methods:** Combine multiple uncorrelated signals
- **Parameter optimization:** Grid search key parameters on train data
- **Risk management:** Dynamic position sizing, stop-losses, take-profits

Write a 1-2 sentence hypothesis before each experiment.

### 2. Implement
Edit `strategy.py` with your new strategy. Keep it clean and readable.
Git commit your changes with a descriptive message.

### 3. Execute
```bash
# Run on all symbols (train period)
python backtest.py --all-symbols > run.log 2>&1

# Check results
grep "Total Return\|Sharpe\|Max Drawdown\|Win Rate" run.log

# Full evaluation
python -c "
from backtest import run_strategy_backtest
from evaluate import compute_metrics, print_metrics
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    r = run_strategy_backtest(symbol=sym, period='train')
    m = compute_metrics(r)
    print_metrics(m, f'{sym} Train')
"
```

### 4. Evaluate
Compare against the current best result:
- **Primary metric:** Sharpe ratio (higher is better)
- **Secondary:** Sortino, Calmar, max drawdown, profit factor
- **Minimum thresholds:** Sharpe > 0.5, win rate > 40%, max DD > -30%

### 5. Decide
- **If improved** (better Sharpe on majority of symbols): KEEP the commit
- **If worse or equal:** Discard: `git checkout -- strategy.py` to revert
- **If crashed:** Fix the bug, or discard and move on

### 6. Log
Append result to `results.tsv`:
```bash
python -c "
from backtest import run_strategy_backtest
from evaluate import compute_metrics, metrics_to_tsv_row
r = run_strategy_backtest(symbol='BTCUSDT', period='train')
m = compute_metrics(r)
print(metrics_to_tsv_row(m, r.strategy_name, 'BTCUSDT', commit='abc123', status='keep', description='Your description'))
" >> results.tsv
```

### 7. Document (every 5 experiments)
Save promising strategies to `strategies/` directory:
```bash
cp strategy.py strategies/<strategy_name>.py
```

Write strategy documentation in `docs/strategies/<strategy_name>.md`:
```markdown
# Strategy: <name>
## Hypothesis
## Logic
## Parameters
## Results (train)
## Results (test) - only for final evaluation
## Observations
```

### 8. Test Evaluation (for promising strategies only)
When a strategy achieves Sharpe > 1.0 on train data across all symbols:
```bash
python backtest.py --all-symbols --test
```
Document test results separately. Do NOT iterate based on test results.

## Strategy Ideas to Explore (Priority Order)

1. **Funding rate mean reversion** - When funding is extremely positive, price tends to reverse (shorts get crowded). Signal: go short when funding > 2 std devs above mean.

2. **Volume-weighted momentum** - Momentum weighted by relative volume spikes. High volume breakouts have more follow-through.

3. **Multi-timeframe trend** - Trend on 4h/1d as filter, entries on 1h/15m for timing. Only trade in direction of higher timeframe trend.

4. **Volatility regime switching** - Use realized volatility to switch between trend-following (low vol) and mean-reversion (high vol) modes.

5. **RSI divergence** - Price makes new high but RSI doesn't (bearish). Price makes new low but RSI doesn't (bullish). Combine with trend filter.

6. **Bollinger Band squeeze breakout** - Enter on breakout from low-volatility squeeze. Bandwidth below threshold → wait for expansion.

7. **Cross-asset signals** - BTC trend as filter for ETH/SOL trades. SOL often leads BTC in momentum.

8. **Funding + momentum combo** - Combine funding rate signal with price momentum for higher-conviction trades.

## NEVER STOP

Keep running experiments. The goal is to find strategies with:
- Sharpe > 1.5 on train
- Consistent across all 3 symbols
- Robust (not overfitted to specific market conditions)
- Simple (prefer fewer parameters)

When in doubt, try something new rather than micro-optimizing.
