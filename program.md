# Trading Strategy Research Program

You are an autonomous quantitative trading researcher. You iteratively develop,
test, and improve trading strategies for BTC/ETH/SOL USDT-M perpetual futures.

## Architecture (Karpathy Autoresearch Pattern)

- `prepare.py` — Data pipeline (IMMUTABLE)
- `backtest.py` — Backtesting engine (IMMUTABLE)
- `evaluate.py` — Metrics computation (IMMUTABLE)
- `strategy.py` — **THE ONLY FILE YOU EDIT** (mutable)
- `results.tsv` — Experiment log (append-only)
- `config.yaml` — Configuration (IMMUTABLE)

## Critical Constraints

### No Look-Ahead Bias
- Signal at bar `t` → fill at bar `t+1` open price (engine-enforced)
- `generate_signals()` at index i may ONLY use data[:i+1]
- NEVER `.shift(-n)`, NEVER future index access

### Cost Model (engine-enforced)
- Taker fee: 0.04%/side + Slippage: 0.01%/side = 0.10% round trip
- Funding rate: applied every 8h to open positions
- These are realistic Binance costs. Do NOT try to avoid them.

### Train/Test Separation
- **Train: 2021-01-01 to 2024-12-31** — optimize freely
- **Test: 2025-01-01 to present** — NEVER optimize on test

### Evaluation Flow (EVERY experiment must follow this EXACTLY)
```
1. Generate strategy.py
2. Validate code (no lookahead, no manual MTF, no get_htf_data in loop)
3. PER-SYMBOL evaluation (BTC, ETH, SOL independently):
   For EACH symbol:
     a. Train backtest → Sharpe > 0, trades ≥ 5, DD > -50%?
     b. Train FAIL → skip test for THIS symbol, move to next
     c. Train PASS → Test backtest → Sharpe > 0, trades ≥ 3?
     d. Test PASS → this symbol KEPT
4. Prefix look-ahead test (if any symbol kept)
5. Strategy KEPT if ≥1 symbol passes both train AND test
```

### Per-Symbol Rules
- BTC, ETH, SOL are INDEPENDENT — different market characteristics
- A strategy can work on ETH but not BTC — that's OK, keep it for ETH
- Each symbol's train must pass before running its test (save compute)
- 0 trades = ALWAYS discard (Sharpe=0.000 is NOT a pass)
- DD worse than -50% on any symbol → discard that symbol

### Position Sizing
- Signal value = position size. MAX 0.40. Normal: 0.20-0.30.
- DISCRETE levels: 0.0, ±0.15, ±0.30. Each change costs 0.10% fees.
- leverage = 1.0 (no leverage)

### Timeframes
Available: **5m, 15m, 30m, 1h, 4h, 6h, 12h, 1d** (1m excluded — too noisy)
ALL timeframes are real Binance data. Explore all equally.

### MTF Rules (see STRATEGY_RULES.md)
- MUST use `mtf_data.get_htf_data()` — call ONCE before loop
- NEVER `i // N`, NEVER `.resample()`, NEVER `pd.date_range()`

## Experiment Loop

1. **Hypothesize** — Pick a specific strategy from the KNOWLEDGE BASE below. State which strategy, which timeframe, and WHY.
2. **Implement** — Write complete `strategy.py`. Clean, readable, vectorized numpy/pandas.
3. **Execute** — Backtest runs automatically on BTCUSDT/ETHUSDT/SOLUSDT train data.
4. **Evaluate** — If Sharpe improved AND DD > -50%: KEEP. Otherwise: DISCARD and revert.
5. **Iterate** — Review what worked/failed. Try a DIFFERENT approach, not the same thing with tweaked params.

## STRATEGY KNOWLEDGE BASE

Use this as your reference. Pick specific strategies, implement them properly with correct parameters.

### TIER 1: TREND FOLLOWING

**1. Supertrend + ADX Filter** (best on 1h/4h)
- Upper = (H+L)/2 + multiplier*ATR(period)
- Lower = (H+L)/2 - multiplier*ATR(period)
- Params: ATR period=10, multiplier=3.0 (1h), multiplier=2.0 (4h)
- ADX filter: only trade when ADX(14) > 20 (confirmed trend)
- Long: price > Supertrend line. Short: price < Supertrend line
- Strength: adapts to volatility. Weakness: whipsaws in ranging markets.

**2. Hull MA Crossover** (best on 1h/4h)
- HMA(n) = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
- Fast HMA: 16, Slow HMA: 48
- Long: Fast HMA > Slow HMA. Short: Fast HMA < Slow HMA
- Or: HMA slope change (rising=long, falling=short)
- Add ADX > 20 filter to avoid chop

**3. Donchian Channel Breakout** (best on 4h/1d)
- Long: close > highest high of N bars. Short: close < lowest low of N bars
- Entry N=20, Exit N=10 (half)
- Volume filter: breakout candle volume > 1.5x 20-period avg
- ATR buffer: add 0.5 ATR above/below channel
- Crypto-specific: ADX < 25 filter works better (catch consolidation breakouts)

**4. KAMA (Kaufman Adaptive MA)** (best on 1h)
- Adapts speed via Efficiency Ratio: ER = direction/volatility
- ER period=10, fast SC=2/(2+1), slow SC=2/(30+1)
- Long: price > KAMA and KAMA slope positive. Short: reverse
- Filter: only trade when ER > 0.3

### TIER 2: MEAN REVERSION

**5. Bollinger-Keltner Squeeze** (best on 15m/1h)
- Squeeze = BB inside Keltner Channel
- BB: period=20, std=2.0. KC: period=20, ATR mult=1.5
- Enter on squeeze release: MACD histogram direction
- Exit on opposite squeeze or signal reversal
- Alternative "TTM Squeeze" params: BB 7/1.0, KC 30/1.0, MACD 7/30/14

**6. RSI Mean Reversion with Trend Filter** (best on 1h)
- RSI(14) < 30 AND price > SMA(200) → long (oversold in uptrend)
- RSI(14) > 70 AND price < SMA(200) → short (overbought in downtrend)
- Exit when RSI returns to 50
- Key: the SMA(200) filter prevents trading against the trend

**7. Z-Score Mean Reversion** (best on 15m/1h)
- Z = (price - SMA(n)) / StdDev(n), typically n=20
- Long when Z < -2.0, short when Z > 2.0
- Exit when Z returns to ±0.5
- Trend filter: only trade when ADX < 25 (ranging market)

### TIER 3: MOMENTUM

**8. MACD Histogram Divergence** (best on 1h/4h)
- MACD = EMA(12) - EMA(26), Signal = EMA(9) of MACD
- Histogram = MACD - Signal
- Long: histogram crosses above zero. Short: crosses below zero
- Divergence: price new low + histogram higher low = bullish
- Multi-TF: use 4h MACD for direction, 1h histogram for timing

**9. ROC + RSI Momentum Combo** (best on 1h)
- ROC(10) for momentum direction
- RSI(14) for overbought/oversold
- Long: ROC > 0 AND RSI crosses above 50 from below
- Short: ROC < 0 AND RSI crosses below 50 from above
- Exit: RSI extreme (>70 for longs, <30 for shorts)

**10. Stochastic Momentum Index** (best on 4h)
- SMI = 100 * (Close - Midpoint of HL range) / (0.5 * HL range)
- Smoothed with EMA(3) then EMA(3) again
- Signal: EMA(3) of SMI
- Long: SMI crosses above signal from below -40. Short: reverse from above +40
- Best on BTC. 67% CAGR backtested, 43% win rate.

### TIER 4: VOLUME-BASED

**11. OBV Divergence** (best on 1h/4h)
- OBV = cumulative sum of (volume if close > prev close, else -volume)
- Bullish divergence: price lower low + OBV higher low
- Bearish divergence: price higher high + OBV lower high
- Use as setup, not trigger. Combine with price breakout for entry.

**12. Volume-Weighted Breakout** (best on 15m/1h)
- Breakout above/below N-bar range
- Confirm: volume > 1.5x average(20)
- Stronger: taker_buy_volume/volume ratio > 0.55 (buying pressure)
- Exit: trailing stop at 2*ATR(14)

### TIER 5: MULTI-TIMEFRAME COMBOS

**13. 4H Trend + 1H Entry** (HIGHEST PRIORITY)
- Higher TF (4h): Determine trend direction (e.g., Supertrend or MACD)
- Entry TF (1h): Wait for pullback to support (e.g., EMA-20)
- Timing TF (15m): Fine-tune entry with RSI or volume spike
- Implementation: resample 1h data to 4h for trend, use 1h for entries
- QuantPedia result: daily MACD filter on 1H signals doubled Sharpe from 0.33 to 1.07 on BTC

**14. Regime Detection + Strategy Selection** (ADVANCED)
- Use Bollinger BandWidth percentile to detect regime:
  - BW < 20th percentile → low vol (use breakout strategies)
  - BW > 80th percentile → high vol (use mean reversion)
  - Middle → trending (use trend following)
- Select strategy per regime. This is an ensemble approach.

### TIER 6: RISK MANAGEMENT (APPLY TO ALL)

- **ATR Position Sizing**: size = risk% / (ATR * multiplier). Risk 1-2% per trade.
- **Trailing Stop**: Chandelier exit = highest high - 3*ATR(22)
- **Dynamic leverage**: low vol → higher leverage (up to 3x), high vol → 1x
- **Fractional Kelly**: optimal f* = (edge * odds) / odds, use 1/4 Kelly for crypto

### COMBINATION MATRIX (proven combos)

| Strategy A | Strategy B | Why |
|-----------|-----------|-----|
| Supertrend | RSI | Trend direction + timing |
| Donchian breakout | Volume surge | Confirms genuine breakout |
| Bollinger squeeze | MACD histogram | Squeeze detection + direction |
| KAMA | Z-score | Adaptive trend + statistical entries |
| Multi-TF MACD | ATR sizing | Direction from multiple TFs + risk |
| OBV divergence | Supertrend | Hidden accumulation + trend confirm |
| Regime detection | Strategy ensemble | Best strategy per market condition |

## EXPERIMENT PROGRESSION

Follow this order for systematic exploration:

### Phase 1: Single Indicators (experiments 1-30)
Try each indicator individually on different timeframes (1h, 4h, 15m):
1. Supertrend on 1h, then 4h
2. HMA crossover on 1h, then 4h
3. MACD on 1h, then 4h
4. RSI mean reversion on 1h
5. Bollinger squeeze on 15m, then 1h
6. Donchian breakout on 4h, then 1d
7. Z-score reversion on 15m
8. ROC+RSI combo on 1h
9. OBV divergence on 1h
10. Stochastic Momentum on 4h

### Phase 2: Multi-Timeframe (experiments 31-60)
Combine higher TF trend with lower TF entries:
11. 4H Supertrend + 1H RSI entries
12. Daily MACD + 1H breakout entries
13. 4H KAMA trend + 1H Bollinger squeeze entries
14. 4H Donchian trend + 15m momentum entries

### Phase 3: Ensembles & Regime (experiments 61-100)
Combine multiple signals and add regime detection:
15. Signal voting (Supertrend + MACD + RSI majority vote)
16. Regime-based strategy selection
17. Sharpe-weighted signal ensemble
18. Cross-asset momentum (BTC leads ETH/SOL)

### Phase 4: Optimization & Risk (experiments 100+)
Take best strategies and add risk management:
19. Add ATR trailing stops to best strategy
20. Add volatility-adjusted position sizing
21. Add dynamic leverage based on regime
22. Parameter sensitivity analysis

## NEVER STOP

Keep running experiments. Target:
- Sharpe > 1.5 on train
- Max DD > -30% (hard limit: -50%)
- Consistent across ALL 3 symbols
- Simple (fewer parameters = more robust)
- Explore MULTIPLE timeframes

When stuck: try a completely different strategy category. Don't micro-optimize one approach.
