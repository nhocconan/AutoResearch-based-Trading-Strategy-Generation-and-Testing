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

### TIER 7: PIVOT BOSS PIVOT METHODS (Frank Ochoa)

**15. Central Pivot Range (CPR)** (best on 4h/1d, daily CPR)
- Pivot = (High + Low + Close) / 3
- BC (Bottom Central) = (High + Low) / 2
- TC (Top Central) = (Pivot - BC) + Pivot
- CPR width = TC - BC. Narrow CPR → trending day. Wide CPR → ranging day.
- Entry: price breaks above TC → long; breaks below BC → short
- Filter: if today's CPR is above yesterday's → bullish bias (non-overlapping CPR)
- **Virgin CPR**: an untested CPR level from a prior day acts as strong S/R
- Implementation: use 1d HTF data for CPR levels, trade on 4h/1h entry TF

**16. Weekly/Monthly Pivot Levels** (best on 4h/1d)
- Standard pivots: P=(H+L+C)/3; R1=2P-L; S1=2P-H; R2=P+(H-L); S2=P-(H-L)
- Camarilla levels: R3=C+1.1*(H-L)/4; S3=C-1.1*(H-L)/4 (very tight, mean-reversion)
- Woodie pivot: P=(H+L+2C)/4 (weights closing price more)
- Trade: fade at R1/S1 (mean reversion). Breakout above R2 → momentum long.
- Use WEEKLY pivots as major zones, DAILY pivots for entries
- Implementation: from 1w HTF data, compute each week's pivot for the following week

**17. CPR Trend Day vs Range Day** (best on 1h/4h)
- Non-overlapping CPR (today's CPR entirely above/below yesterday's) → trending day
- Overlapping CPR → ranging day
- Trend day: enter on CPR break, trail with ATR, ignore mean reversion
- Range day: fade moves at CPR edges, take profit at ±1 ATR
- Filter with ADX: ADX > 25 confirms trend day, ADX < 20 confirms range day

**18. Floor Trader Pivot + Session** (best on 1h)
- Asian session (00:00-08:00 UTC): accumulation. Range = high-low of session.
- London open (08:00 UTC): often breaks Asian range direction
- NY open (13:00 UTC): strongest move, often reversal or continuation
- Use previous day high/low/close for floor trader pivots
- Long: price holds above pivot + Asian range high breaks at London open

### TIER 8: ICHIMOKU CLOUD STRATEGIES (一目均衡表)

**19. Classic Ichimoku Trend** (best on 4h/1d — original params designed for daily)
- Tenkan-sen (Conversion): (9H + 9L) / 2  — fast signal
- Kijun-sen (Base): (26H + 26L) / 2  — slow signal / support
- Senkou Span A = (Tenkan + Kijun) / 2, plotted 26 bars ahead
- Senkou Span B = (52H + 52L) / 2, plotted 26 bars ahead
- Chikou Span = Close, plotted 26 bars BACK
- Cloud (Kumo) = area between Span A and Span B
- **TK Cross**: Tenkan crosses above Kijun → long signal (bearish if below cloud)
- **Cloud Breakout**: price breaks above cloud → strong long signal
- **Chikou confirmation**: Chikou above price of 26 bars ago → bullish
- Full bullish: price above cloud + TK cross up + Chikou clear + green cloud ahead
- Note: in backtest, use Span A/B from 26 bars AGO (already closed) to avoid look-ahead

**20. Ichimoku + RSI Combo** (best on 4h)
- Use Ichimoku cloud for trend direction only (price above/below cloud)
- RSI(14) for entry timing: oversold bounce (RSI < 40) in uptrend = long
- Exit: price enters cloud or RSI > 70
- This simplifies Ichimoku to a trend filter — more robust than full system

**21. Kijun Bounce Strategy** (best on 1d/4h)
- Kijun-sen acts as magnetic support/resistance in trending markets
- In uptrend (price above cloud): buy pullbacks to Kijun-sen
- Stop: below Kumo (cloud bottom)
- Target: previous swing high or Tenkan-sen
- Filter: only take bounces when Kijun is flat (acts as magnet, not dynamic)

**22. Kumo Breakout + ATR** (best on 4h/12h)
- Wait for price to be inside cloud (consolidation)
- On breakout above cloud top (Span A or B, whichever is higher): long
- Position size proportional to cloud thickness (wider = stronger breakout)
- Stop: below cloud bottom. Target: 2x cloud thickness above entry
- Filter: avoid if cloud ahead is thin (< 0.5% price) — weak breakout

**23. Ichimoku Flat Kijun Scalp** (best on 1h)
- Flat Kijun (no change for 10+ bars) = price magnet / equilibrium
- Price deviates > 1 ATR from flat Kijun → mean-revert back to Kijun
- Long: price > 1 ATR below flat Kijun. Exit when price touches Kijun.
- Short: price > 1 ATR above flat Kijun. Exit at Kijun.
- Avoid if Chikou shows trend (Chikou far from price 26 bars ago)

**24. Multi-TF Ichimoku** (best on 1d primary, 4h entry)
- 1d cloud for trend bias: above cloud = long-only mode
- 4h TK cross for entry timing within 1d trend direction
- 4h Kijun as stop level
- This is the most common professional Ichimoku setup

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
| CPR trend/range | ADX filter | Day type detection + trend confirmation |
| Weekly pivot S/R | Donchian breakout | Macro levels + breakout confirmation |
| Ichimoku cloud | RSI | Trend filter + oversold/overbought timing |
| Kijun bounce | ATR trailing stop | Mean-revert to equilibrium + risk control |
| Multi-TF Ichimoku | Kijun stop | Professional setup (1d trend, 4h entry) |
| CPR virgin levels | Volume breakout | Key untested levels + volume confirmation |

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

### Phase 5: Pivot Boss Pivots (priority: HIGH — unexplored)
Try each on 4h timeframe with 1d HTF pivots:
23. Daily CPR breakout (TC/BC levels as entries)
24. Weekly standard pivots (R1/S1 fade, R2 breakout)
25. CPR width filter (narrow=trend day, wide=range day)
26. Virgin CPR mean-reversion (untested prior CPR levels)
27. Camarilla R3/S3 mean-reversion

### Phase 6: Ichimoku Strategies (priority: HIGH — unexplored)
Try each on 4h/1d:
28. TK Cross with cloud filter (4h)
29. Cloud breakout with ATR stop (4h/12h)
30. Kijun bounce in uptrend (1d primary, 4h entry)
31. Ichimoku + RSI combo (simplified, 4h)
32. Multi-TF: 1d cloud direction + 4h TK cross entry
33. Flat Kijun mean-reversion (1h)

## NEVER STOP

Keep running experiments. Target:
- Sharpe > 1.5 on train
- Max DD > -30% (hard limit: -50%)
- Consistent across ALL 3 symbols
- Simple (fewer parameters = more robust)
- Explore MULTIPLE timeframes

When stuck: try a completely different strategy category. Don't micro-optimize one approach.
