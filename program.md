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
Available: **5m, 15m, 30m, 1h, 4h, 6h, 12h, 1d, 1w** (1m excluded — too noisy; no 8h data — use 6h or 12h instead)
ALL timeframes are real Binance data. Explore all equally.

### TIMEFRAME KEEP RATES (from 16,000+ actual experiments — DATA DRIVEN)

| TF   | Strategies | Keep Rate | Best Test Sharpe | Notes |
|------|-----------|-----------|-----------------|-------|
| 12h  | 525       | **54%**   | 1.485           | BEST — highest keep rate |
| 4h   | 837       | **41%**   | 1.340           | PROVEN — volume+Donchian pattern works |
| 1d   | 437       | **40%**   | 1.449           | PROVEN — KAMA+regime works |
| gen  | 13976     | **45%**   | 1.787           | Pivot/volume strategies work |
| 6h   | 359       | 23%       | 1.512           | Mediocre — only try with novel concepts |
| 1h   | 217       | 17%       | 0.798           | Difficult — needs very strict entries |
| 30m  | 97        | 9%        | 0.616           | Hard — few successful experiments |
| 15m  | 75        | **0%**    | N/A             | ⛔ DEAD — see STOP PURSUING below |
| 5m   | 18        | **0%**    | N/A             | ⛔ DEAD — fee drag insurmountable |

**FOCUS ON: 12h > 4h > 1d. Only try 6h/1h with genuinely novel concepts.**

### ⚠️ CRITICAL: FEE DRAG IS THE #1 KILLER — TRADE COUNT TARGETS

0.10% round trip per trade. Overtrading = certain death.

| TF   | Target TOTAL over 4yr | Target /year | HARD MAX total |
|------|----------------------|-------------|----------------|
| 4h   | 75-200               | 19-50       | 400            |
| 6h   | 50-150               | 12-37       | 300            |
| 12h  | 50-150               | 12-37       | 200            |
| 1d   | 30-100               | 7-25        | 150            |

"50 trades minimum" = 50 TOTAL over 4 years = 12.5/year. NOT 50/year.
Fewer than 50 total = statistically unreliable Sharpe — will be rejected.
If your strategy has >400 4h trades: entry is too loose → add MORE filters.

### MTF Rules (see STRATEGY_RULES.md)
- MUST use `mtf_data.get_htf_data()` — call ONCE before loop
- NEVER `i // N`, NEVER `.resample()`, NEVER `pd.date_range()`

## ⛔ STOP PURSUING — EXHAUSTED / LOW-YIELD COMBINATIONS (DATA-DRIVEN as of 2026-03)

Based on 16,000+ experiments. Do NOT repeat these — try something genuinely different.

### DO NOT DO (extremely high discard rate, no future upside):
- **15m as primary timeframe**: 75 experiments, **0% keep rate**. Fee drag insurmountable. STOP.
- **5m as primary timeframe**: 18 experiments, **0% keep rate**. STOP.
- **Simple EMA/SMA crossovers** (gen_ema_cross, gen_sma_cross, gen_golden_cross): >85% discard. Never works on BTC/ETH.
- **Parabolic SAR standalone** (gen_parabolic_sar): >90% discard. Too many whipsaws.
- **Heikin-Ashi as primary signal** (gen_heikin_ashi): >90% discard. Repaints effectively.
- **HMA slope only** (gen_hma_slope, gen_kama_direction alone): >90% discard without regime filter.
- **Fisher Transform as sole signal** (Fisher keep rate: 19%): only works combined with ADX+volume.
- **Pure cRSI mean-reversion without trend filter**: fails 2022/2025 bear markets badly.
- **"Loose" entry conditions** (any TF): strategies with >400 train trades on 4h uniformly fail test.
- **6h strategies passing ONLY SOL**: SOL has known 100x bias. If BTC+ETH both fail, DISCARD.
- **12h HMA+RSI+Chop variations** (>20 tested): exhausted. Move to different indicator combo.
- **MTF strategies with 1d1w HTF and 12h primary if only keeping SOL**: SOL bias, not general.
- **Ichimoku as primary signal on short TF (<4h)**: keep rate on train OK but fails test.
- **Any "version N" of same concept after 2 failures**: if v1 and v2 failed, the concept is wrong. Change completely.

### NEAR-EXHAUSTED (diminishing returns, try only with truly novel twist):
- **HMA + RSI + Chop (any timeframe)**: 2000+ tested. Only try if adding completely new element (funding rate, Camarilla level, VPIN).
- **Regime detection + trend following**: 6000+ tested. OK only with non-standard regime (Elder Ray, Alligator, STC).
- **Donchian breakout + volume**: well-explored. Try with Camarilla S4/R4 or weekly pivot confirmation.
- **KAMA + RSI**: 1000+ tested. Try only with Donchian exit or funding rate bias.
- **CRSI + Chop + Regime (4h/6h "loose")**: 30+ recent fails all due to overtrading. If CRSI/chop, must have strict entry gating.

### WHAT STILL HAS UPSIDE (under-explored or high keep rate — prioritize these):
- **Funding rate strategies** (39% keep rate, best test 1.449): MAJOR untapped edge. Try more variations.
- **STC (Schaff Trend Cycle)**: only 1 experiment! Very promising momentum oscillator. Try both standard (23/50/10) and crypto-fast (10/21/5) params.
- **Williams Alligator**: essentially untested. Backtested BTC: profitability factor 3.72. Try with AO confirmation on 12h.
- **Elder Impulse System** (Elder Ray variant): untested. Green/Red/Blue bar classification, very clean implementation.
- **Elder Ray (Bull/Bear Power)**: untested. Measures buying/selling power relative to EMA.
- **Vortex Indicator**: untested. Long signals much stronger than shorts (76% vs 43.5% win rate).
- **TTM Squeeze** (Bollinger-Keltner): under-explored on 6h/12h. Squeeze duration > 3 bars = reliable.
- **Camarilla pivot on 4h/12h** (tested only on gen): best test 1.47 — try on mtf 4h/12h primary.
- **ADX + Volume combination**: 50% keep rate, high potential, under-explored.
- **TRIX + volume spike** (ETHUSDT: 1.32 test): very promising, explore variations.
- **Funding rate mean-reversion** on BTC/ETH specifically (Z-score of 30d funding): proven edge.
- **KAMA + Donchian exit** (NOT RSI exit): novel combination since KAMA+RSI is exhausted.
- **Momentum + Mean-Reversion blend** 50/50: Sharpe 1.71, CAGR 56% in systematic backtests.

### WINNING FORMULA (reverse-engineered from top test performers):
Top strategies share these traits:
1. **ONE strong structural signal** (Donchian channel break OR Camarilla level touch)
2. **Volume confirmation** (breakout candle volume > N-period average)
3. **Regime filter** (Choppiness OR ADX) to avoid chop periods
4. **ATR-based stoploss** (clear exit = fewer whipsaws = better Sharpe)
5. **~75-200 train trades total** = tight enough to have signal quality, loose enough to have >30 minimum

## Experiment Loop

1. **Hypothesize** — Pick a specific strategy. State which strategy, which timeframe, and WHY.
   - **CHECK**: Is this in the STOP PURSUING list above? If yes, pick something different.
   - **CHECK**: Is the concept a variation of something already tried 3+ times? If yes, pick genuinely different concept.
   - **PREFER**: 12h (54% keep) or 4h (41% keep). Avoid 15m/5m (0% keep).
   - If you've tested the same indicator combo 2+ times and it failed, the concept is wrong. Change it.
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

### TIER 7: PIVOT BOSS METHODS (Frank Ochoa — "Secrets of a Pivot Boss")

**Core formulas (computed from PREVIOUS day/week OHLC via get_htf_data):**
```
# Standard Floor Trader Pivots
P  = (H + L + C) / 3          # Central pivot
R1 = 2*P - L                   # First resistance
R2 = P + (H - L)               # Second resistance
R3 = H + 2*(P - L)             # Third resistance (rare, strong move)
S1 = 2*P - H                   # First support
S2 = P - (H - L)               # Second support
S3 = L - 2*(H - P)             # Third support

# Central Pivot Range (CPR) — Ochoa's signature tool
BC = (H + L) / 2               # Bottom Central
TC = 2*P - BC                  # Top Central
# CPR = zone between BC and TC. Width = TC - BC.

# Camarilla Pivots (tighter, mean-reversion focused)
R4 = C + 1.1*(H-L)/2           # Breakout level — above = strong trend
R3 = C + 1.1*(H-L)/4           # Fade level — price likely reverses here
R2 = C + 1.1*(H-L)/6
R1 = C + 1.1*(H-L)/12
S1 = C - 1.1*(H-L)/12
S2 = C - 1.1*(H-L)/6
S3 = C - 1.1*(H-L)/4           # Fade level
S4 = C - 1.1*(H-L)/2           # Breakout level

# Woodie Pivots (weights close price double)
Pw = (H + L + 2*C) / 4
R1w = 2*Pw - L
S1w = 2*Pw - H
```

**CPR Width = key day-type classifier:**
- `width_pct = (TC - BC) / P * 100`
- width < 0.15%: very narrow → extremely strong trend day expected
- width 0.15–0.40%: narrow → trending/directional day
- width 0.40–0.80%: medium → mild trend or first-half trend
- width > 0.80%: wide → rotational/range day
- width > 1.5%: very wide → choppy, avoid or fade only

**Value Area = S1 to R1 range:**
The zone between S1 and R1 is the "expected value area" for the day.
- Price inside S1–R1: rotational "fair value" trading
- Price above R1: bullish breakout from value → trend continuation
- Price below S1: bearish breakdown from value → trend continuation
- First target for longs: R1. First resistance above = R2.
- First target for shorts: S1. First support below = S2.

**CPR Trend Classification (multi-day context):**
- `rising_cpr = BC_today > TC_yesterday` → non-overlapping and rising → BULLISH TREND
- `falling_cpr = TC_today < BC_yesterday` → non-overlapping and falling → BEARISH TREND
- Overlapping CPR: consolidation day, fade moves
- Track last 3 days: if all 3 non-overlapping and rising → strong bull trend

**Virgin CPR (magnet effect):**
- A CPR that price has never traded through = "Virgin CPR"
- Price tends to gravitate toward and test Virgin CPRs from prior days
- Implementation: track if today's close has ever been inside a prior day's CPR
- Simple approximation: if gap between today open and yesterday CPR → virgin level acts as magnet

---

**15. Daily CPR Breakout with Value Area** *(primary 15m or 1h, HTF 1d)*
- Use daily CPR (BC, TC) + R1/S1 (value area edges) computed from 1d HTF
- Long entry: 15m/1h close > TC AND price was below TC within last 3 bars (fresh breakout)
  - Filter: CPR narrow (width < 0.5%) = trend day → position 0.30
  - Target: R1. If reaches R1, hold or reduce to 0.15 toward R2.
- Short entry: 15m/1h close < BC AND fresh breakdown
  - Target: S1.
- Weekly pivot bias: if close > weekly P → only take long CPR breakouts (not shorts)
- Implementation: get_htf_data(prices,'1d') for CPR, get_htf_data(prices,'1w') for weekly P

**16. CPR Trend Day Classification + HMA** *(primary 15m or 6h, HTF 1d + 1w)*
- Classify day type BEFORE trading:
  - If narrow CPR (width_pct < 0.4%) → trend day mode: enter on TC/BC break, hold
  - If wide CPR (width_pct > 0.8%) → range day mode: fade at R1/S1/R2/S2
- Trend day: enter when 15m HMA crosses up AND close > TC → 0.30 long
  - Exit: close back below TC or HMA crosses down
- Range day: fade when close > R1 (short 0.15) or close < S1 (long 0.15)
  - Exit: mean-revert back toward pivot P
- Non-overlapping CPR adds 0.15 to position in trend direction

**17. Multi-Day CPR Trend + 6h Entry** *(primary 6h, HTF 1d + 1w)*
- Daily CPR series: compare today's BC/TC vs yesterday's
  - 3 consecutive rising non-overlapping CPRs → strong bull → 6h long only
  - 3 consecutive falling non-overlapping CPRs → strong bear → 6h short only
  - Otherwise → bi-directional
- Entry: 6h RSI < 45 in bullish CPR trend = pullback long
- Entry: 6h RSI > 55 in bearish CPR trend = pullback short
- Position: 0.30. Exit when CPR trend reverses OR weekly pivot breached.

**18. Weekly Pivot S/R with 1h Entries** *(primary 1h, HTF 1w + 1d)*
- Weekly standard pivots (computed from 1w HTF data):
  - WP = (weekly H + L + C) / 3, WR1 = 2*WP - L, WS1 = 2*WP - H
  - WR2 = WP + (H-L), WS2 = WP - (H-L)
- Trend bias: close > WP → bullish for the week (long only or long-heavy)
- Entry: 1h RSI(14) < 40 AND price near WP or WS1 → long 0.30
- Entry: 1h RSI(14) > 60 AND price near WP or WR1 from above → short 0.30
- Filter: daily CPR must also confirm direction (non-overlapping rising for longs)

**19. Camarilla R3/S3 Mean Reversion** *(primary 15m or 1h, HTF 1d)*
- Camarilla R3: price typically reverses here unless there's a breakout
- Long: 15m/1h price dips to S3 AND weekly price above weekly P → long 0.15–0.30
  - Exit: price returns to S1 or Camarilla midpoint (C)
- Short: price spikes to R3 AND weekly below weekly P → short 0.15–0.30
  - Exit: price returns to R1 or midpoint
- BREAKOUT mode: if price closes ABOVE R4 (not just touches R3) → trend continuation long
  - This is the key Camarilla rule: R3 = fade, R4 = breakout. Opposite for S3/S4.

**20. Daily Pivot Bounce + 15m Momentum** *(primary 15m, HTF 1d)*
- Daily P, S1, S2, R1, R2 as S/R levels from 1d HTF
- Strategy: price trades toward S1 during session, holds support → bounce long
  - Confirm: 15m RSI crosses above 40 from below at S1 → long 0.30
  - Target: back to P (pivot). If strong momentum, extend to R1.
- Flip: price trades up to R1, fails → short 0.15 back to P
- Filter: narrow CPR day only (expect clear bounces, not chop)

**21. Woodie Pivot Trend (Crypto-Optimized)** *(primary 1h or 6h, HTF 1d)*
- Woodie pivot: Pw = (H + L + 2*C) / 4 (gives 50% weight to close = momentum-aware)
- Woodie R1 and S1 often tighter than standard, better for crypto volatility
- Long: 1h/6h close > Pw AND R1w is first target, exit if weekly S1 hit
- Short: close < Pw, S1w is target
- Add HMA filter: HMA slope must agree with trade direction
- CPR width confirms: narrow = take the trade, wide = skip or halve size

**22. Opening Range + Daily Pivots (Crypto: UTC Daily Open)** *(primary 15m, HTF 1d)*
- Binance daily candle resets at 00:00 UTC
- "Opening Range" = high and low of the first 1h (first 4 bars on 15m)
- OR-High and OR-Low define the range
- Strategy:
  - If OR breaks upward AND above daily TC → strong long signal (0.30)
  - If OR breaks downward AND below daily BC → strong short signal (0.30)
- This combines opening range breakout with CPR confirmation
- Important: narrow CPR days have higher OR breakout success rate
- Implementation: track running max/min of first 4 bars each UTC day using index modulo

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

### TIER 9: 1H STRATEGIES (17% keep rate — NEEDS STRICT HTF REGIME GATING)

**Key Finding:** 1h strategies REQUIRE a strong higher-timeframe filter. Without daily/4h regime filter, 1h is just noise. Every 1h strategy below MUST have at least one of: daily MACD direction, 4h HMA trend, daily Chop < 61.8.

**Trade Count Target for 1h:** 50-250 train trades over 4yr. DO NOT exceed 600 total.

**25. 4H HMA + Daily MACD → 1H Entry** *(PROVEN: ~52 trades/yr, Sharpe ~1.07 on BTC)*
- 4H HMA slope for trend direction (rising = long-only mode, falling = short-only)
- Daily MACD (12/26/9): histogram positive = bullish bias, negative = bearish
- 1H entry: only trade in direction agreed by BOTH 4H HMA and daily MACD
- Filter: Chop Index(14) < 61.8 on 1H confirms trending market
- ADX(14) > 20 on 1H confirms momentum
- Position: 0.30 on signal, 0.0 otherwise
- Exit: 4H HMA reverses OR 1H ADX drops below 18
- This is the #1 proven 1h pattern. Daily MACD filter DOUBLES Sharpe vs 1h alone.

**26. Weekly Pivot Bounce with 1H RSI** *(primary 1h, HTF 1w + 1d)*
- Weekly standard pivot (WP, WR1, WS1, WR2, WS2) from prior week's OHLC
- Trend bias: 1h close > WP → bullish week; < WP → bearish week
- Entry: 1H RSI(14) < 38 AND price within 0.3% of WP or WS1 → long 0.30
- Entry: 1H RSI(14) > 62 AND price within 0.3% of WP or WR1 → short 0.30
- Daily CPR confirmation: non-overlapping rising CPR for longs, falling for shorts
- Exit: 1h RSI(14) > 60 (for longs) or RSI < 40 (for shorts)
- Position: 0.30. Do NOT hold through weekly pivot flip.

**27. 1H Supertrend + Daily ATR-Range Filter**
- Supertrend(10, 3.0) on 1H: riding trend, switching on flip
- Daily ATR(14) normalized: only trade on 1h when daily ATR > 14-day median (volatility confirmed)
- Daily HMA(21) direction: must agree with Supertrend signal
- Position: 0.30 in signal direction, 0.0 when daily ATR < median
- This prevents trading on low-volatility choppy days where 1h signals are noise

**28. 1H Donchian(20) Breakout + 4H Regime**
- 4H Choppiness Index(14) < 50 → confirmed trend on 4h timeframe
- 4H HMA(21) direction: positive slope = bull, negative = bear
- 1H entry: close > 20-bar high AND 4H regime both confirm → long 0.30
- 1H entry: close < 20-bar low AND 4H regime both confirm → short 0.30
- Volume confirm: breakout candle 1h volume > 1.5x 20-bar avg
- Exit: 1H close back inside 10-bar mid-channel OR 4H Chop > 61.8

**29. 1H KAMA(10) + 4H Trend Slope Filter**
- KAMA(10, 2, 30) on 1H: adaptive MA that only moves when ER is high
- 4H HMA(21) slope must be same direction as 1H KAMA crossover
- Only enter when 1H KAMA slope changes AND 4H HMA agrees AND 1H ADX > 20
- Position: 0.25. Small size because 1H noise still exists.
- Key: KAMA's Efficiency Ratio already filters some noise, but 4H gate prevents whipsaws

**30. 1H Funding Rate + Daily Trend (Funding Mean-Reversion)**
- Funding rate from Binance every 8h: extreme readings predict reversals
- Long-term trend (daily HMA): trade reversions only IN trend direction
- If funding > +0.05% (extreme positive, longs overleveraged) AND daily trend bearish → short 0.30
- If funding < -0.05% (extreme negative, shorts overleveraged) AND daily trend bullish → long 0.30
- 1H RSI confirmation: RSI < 35 for funded longs, RSI > 65 for funded shorts
- Exit: funding returns to ±0.01% range OR daily trend reverses

### TIER 10: NEW INDICATOR STRATEGIES (UNTESTED — HIGH PRIORITY)

**31. Williams Alligator** *(best on 4h/12h — Bill Williams fractal system)*
- Jaw (blue): 13-period SMMA, shifted 8 bars forward → use 8-bars-ago value
- Teeth (red): 8-period SMMA, shifted 5 bars forward → use 5-bars-ago value
- Lips (green): 5-period SMMA, shifted 3 bars forward → use 3-bars-ago value
- **Crypto-adjusted faster params:** Jaw=10/8, Teeth=6/5, Lips=3/3 (more responsive to BTC/ETH volatility)
- Long: lips > teeth > jaw (all fanned upward, Alligator "eating upward")
- Short: lips < teeth < jaw (all fanned downward)
- Flat: when lines intertwine ("Alligator sleeping") → position = 0.0
- **Awesome Oscillator (AO) confirmation:** AO = SMA(H+L/2, 5) - SMA(H+L/2, 34). AO above zero = bullish impulse. Only take long Alligator signals when AO > 0, shorts when AO < 0.
- **CRSI pullback entry:** Wait for CRSI(3,2,100) < 20 in uptrend before entering long (buy the dip, not the top of the impulse)
- 1d/1w Alligator state as HTF regime filter: is 1d Alligator opening (directional) or sleeping (flat)?
- Backtested on BTC: 40% win rate, avg R/R 2.43, profitability factor 3.72
- Implementation: SMMA(n) = (prev_SMMA*(n-1) + close) / n (recursive, NOT SMA!)
- Shift: use array[i - shift] to get the completed SMMA value from 'shift' bars ago

**32. Elder Ray Power** *(best on 4h/12h)*
- Elder Ray = Dr. Alexander Elder's indicator measuring buying/selling pressure
- EMA(13) of close as baseline
- Bull Power = high - EMA(13): positive = buyers above EMA, stronger = more bullish
- Bear Power = low - EMA(13): negative = sellers below EMA, more negative = more bearish
- EMA trend direction (slope): determines long/short bias
- Long: EMA rising AND Bear Power crosses above zero (from negative to positive)
  - This means: uptrend + bears are exhausted = buy the dip
- Short: EMA falling AND Bull Power crosses below zero
  - Downtrend + bulls are exhausted = sell the bounce
- Divergence: EMA makes new high but Bull Power doesn't → weakening bull trend
- Position: 0.30 on signal. Filter with daily Chop < 61.8 or weekly HMA direction.

**33. Vortex Indicator** *(best on 4h/12h)*
- VI+ = sum(|high - prev_low|, n) / sum(TR, n)
- VI- = sum(|low - prev_high|, n) / sum(TR, n)
- Typically n=14. TR = max(high-low, |high-prev_close|, |low-prev_close|)
- Long: VI+ crosses above VI- → bullish momentum
- Short: VI- crosses above VI+ → bearish momentum
- **IMPORTANT: Short side much weaker in backtests (43.5% win rate vs 76% long win rate).** Consider long-only or half-size shorts.
- Filter: only trade crosses when CHOP(14) < 50 AND Vortex delta (VI+ - VI-) > 0.05 (clear divergence, not marginal cross)
- Position: 0.30 long / 0.15 short. Exit on reversal cross OR ADX drops below 15.
- Combine with 1d HMA for direction bias.

**34. STC (Schaff Trend Cycle)** *(best on 4h/12h)*
- STC = exponentially smoothed stochastic of MACD — removes MACD lag
- Step 1: MACD = EMA(23) - EMA(50) [standard] OR EMA(10) - EMA(21) [crypto-optimized, faster]
- Step 2: Stochastic of MACD over 10 bars [standard] OR 5 bars [crypto-optimized]
  - %K = (MACD - lowest_MACD_N) / (highest_MACD_N - lowest_MACD_N) * 100
- Step 3: Smooth %K with EMA(3) = first STC line
- Step 4: Stochastic of first STC over same N bars → second STC line
- Step 5: Final STC = EMA(3) of step 4
- Long: STC crosses above 25 (leaving oversold zone)
- Short: STC crosses below 75 (leaving overbought zone)
- STC stays in 0-100 range. Fast signal, low lag. Very promising for crypto.
- **QQQ backtest:** 199 trades, 75% win rate, 1.25% avg gain, -26% max DD
- Filter: daily HMA direction must agree with STC signal; ADX(14) > 20 for trend confirmation
- Recommend trying BOTH standard (23/50/10) and crypto-optimized (10/21/5) params

**35. TRIX Momentum** *(best on 4h/12h/1d)*
- TRIX = triple exponentially smoothed EMA, as rate of change
- EMA1 = EMA(close, n), EMA2 = EMA(EMA1, n), EMA3 = EMA(EMA2, n)
- TRIX = (EMA3 - prev_EMA3) / prev_EMA3 * 100 (typically n=14 or 18)
- Signal = EMA(TRIX, 9) — treat like MACD/signal line
- Long: TRIX crosses above signal line (and both above zero → confirmed bullish)
- Short: TRIX crosses below signal line
- Zero-line cross: TRIX above 0 = bullish trend, below 0 = bearish
- Advantage: triple smoothing eliminates most whipsaws → fewer trades, higher quality
- Combine with volume spike confirmation and Donchian level for entry timing

**36. KAMA + Donchian Exit** *(1d primary — KAMA+RSI exhausted, this exit is novel)*
- KAMA(10, 2, 30) on 1d: entry on price crossing above/below KAMA when ER > 0.3
- Exit: NOT RSI (exhausted). Instead use Donchian(10) channel:
  - Long exit: price closes below 10-bar lowest low (momentum breakdown)
  - Short exit: price closes above 10-bar highest high (momentum breakout)
- Add funding rate bias: take longs only when 30d avg funding rate > 0 (market structurally bullish)
- Position: 0.30. With 1d TF expect 20-60 trades over 4yr.
- Note: KAMA+RSI is "exhausted" (1000+ tests). The Donchian exit changes the exit logic completely.

**37. ADX + Volume Regime Filter (High Potential Combo)**
- ADX(14) > 25: strong trend. ADX < 20: ranging. ADX 20-25: transition.
- Volume SMA(20): volume > 1.5x SMA = volume surge (institutional activity)
- Long: ADX > 25 rising AND +DI > -DI AND volume surge → strong bullish momentum
- Short: ADX > 25 rising AND -DI > +DI AND volume surge → strong bearish momentum
- Flat: ADX < 20 → no directional bias (range market, skip)
- Combine with 1d/1w HMA for macro direction filter
- This combo has ~50% keep rate in experiments — very reliable.

**38. Elder Impulse System** *(best on 12h, HTF 1d/1w — Dr. Alexander Elder)*
- Classifies each bar as Green (bull impulse), Red (bear impulse), or Blue (neutral)
- Green bar: EMA(13) slope RISING AND MACD histogram RISING (both pulling in same bull direction)
- Red bar: EMA(13) slope FALLING AND MACD histogram FALLING (both pulling bear)
- Blue bar: either EMA or histogram disagrees (mixed signal)
- Only trade on colored bars: long on green bars, short on red bars, flat on blue
- Add 1d/1w regime: check if 1d Impulse is green (bull week) or red (bear week) → only take same-direction 12h signals
- Signal: +0.30 on green bar entry, -0.30 on red, 0.0 on blue
- Exit: when impulse color changes (Green→Blue or Green→Red = exit long)
- Implementation: EMA(13) slope = EMA_today - EMA_yesterday; MACD histogram direction = hist_today - hist_yesterday
- This is essentially the Elder Ray principle applied to bar-by-bar entry timing

**39. TTM Squeeze (Bollinger-Keltner Compression Breakout)** *(best on 6h/12h)*
- Squeeze condition: Bollinger Bands(20, 2.0) entirely inside Keltner Channel(20, 1.5×ATR)
- Red dot = squeeze active (coiling, low volatility)
- Green dot = squeeze released (expansion, breakout imminent)
- Key finding: **squeeze duration > 3 bars needed for reliable signal.** Short squeezes = false starts.
- Momentum histogram direction at squeeze release = trade direction
  - Histogram = close - midpoint of (highest_high(20), lowest_low(20), close_SMA(20)) for TTM version
  - Or use MACD histogram direction as simpler proxy
- Entry: first green dot (squeeze release) + histogram positive → long 0.30; histogram negative → short 0.30
- Filter: 1d HMA trend must agree with entry direction
- Exit: close below (longs) or above (shorts) Keltner mid, OR opposite momentum signal
- Why 6h/12h TF: longer squeezes on higher TF → much more reliable breakout vs. 1h/15m noise

**37. Funding Rate Z-Score Strategy** *(4h/12h, uses 30-day rolling stats)*
- Compute 30-day rolling mean and std of 8h funding rates
- Z-score = (funding - mean) / std
- When Z > +2.0 (extreme positive funding = crowded longs): bearish signal
- When Z < -2.0 (extreme negative funding = crowded shorts): bullish signal
- Only trade Z-score signal when daily HMA trend is in SAME direction as the contrarian bet
  - E.g.: positive Z (crowded longs) AND daily trend already bearish → strong short
- Position: 0.30 on signal. Exit when Z returns to ±0.5.
- This is pure funding mean-reversion with trend guard. Funding has 39% keep rate.

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
| Williams Alligator | 1d/1w regime | Bill Williams fractal system + macro filter |
| Elder Ray | Chop filter | Buying/selling pressure + trend confirmation |
| Vortex | ADX > 20 | Momentum direction + trend strength gate |
| STC | Daily HMA | Fast oscillator + higher-TF direction |
| TRIX | Volume surge | Lagged momentum + volume breakout confirm |
| Funding Z-score | Daily trend | Crowding mean-revert + trend direction guard |
| 4H HMA | Daily MACD → 1H entry | Proven 1H pattern (Sharpe 1.07 on BTC) |

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

### Phase 5: NEW INDICATOR EXPLORATION (priority: HIGH — barely tested)
These indicators have <5 experiments and high theoretical edge. Try in order:
23. STC (Schaff Trend Cycle) on 12h + ADX > 20 (Tier 10 #34) — only 1 exp ever, HIGHEST priority
24. Williams Alligator on 12h + AO confirmation + 1d regime (Tier 10 #31)
25. Elder Impulse System on 12h + 1d/1w confirmation (Tier 10 #38)
26. Vortex Indicator on 12h + CHOP < 50 gate (Tier 10 #33) — long-biased
27. TTM Squeeze on 6h/12h + 1d HMA filter (Tier 10 #39) — duration > 3 bars
28. Elder Ray Power on 4h + daily Chop filter (Tier 10 #32)
29. TRIX Momentum on 12h/1d with volume spike (Tier 10 #35)
30. Funding Rate Z-Score on 4h/12h with daily trend guard (Tier 10 #40)
31. KAMA + Donchian exit on 1d with funding bias (Tier 10 #36)

### Phase 6: 1H STRATEGIES (priority: MEDIUM — 17% keep rate if done right)
Every strategy in this phase MUST have a strict daily/4h regime gate:
29. 4H HMA + Daily MACD → 1H entry (Tier 9 #25 — PROVEN pattern)
30. Weekly pivot bounce with 1H RSI + daily CPR confirm (Tier 9 #26)
31. 1H Supertrend + daily ATR range filter (Tier 9 #27)
32. 1H Donchian breakout + 4H Choppiness < 50 gate (Tier 9 #28)
33. 1H KAMA + 4H trend slope filter (Tier 9 #29)
34. 1H Funding rate mean-reversion + daily trend (Tier 9 #30)

### Phase 7: Pivot Boss Deep Dive (1h/6h primary, 1d/1w HTF)
35. Daily CPR width classifier + HMA on 1h
36. Multi-day CPR trend (3-day non-overlapping) + 6h entry
37. Weekly pivot S/R with 1h RSI entries
38. Camarilla breakout R4/S4 trend continuation on 4h/12h
39. Combined: Daily CPR + Camarilla + Weekly pivot on 1h

### Phase 8: Ichimoku (4h/12h primary — unexplored strategy)
40. 12h TK Cross with 1d cloud filter
41. 12h Cloud breakout + ATR stop
42. 4h Kijun bounce in uptrend (1d cloud direction)
43. 12h Ichimoku + TRIX momentum combo
44. Multi-TF: 1d cloud → 12h TK cross → 4h entry

## RECENTLY DISCOVERED CONCEPTS
### Batch discovered 2026-04-01 12:02

**1. Funding Rate Z-Score + Fractal Volatility Regime** *(30m/4H)*
- Core idea: Trade mean reversion at extreme funding levels, filtered by adaptive fractal volatility to avoid trending traps.
- Formula: `NFR = (funding - SMA(funding, 168)) / STD(funding, 168)`; `Chop = 100 * log10(SUM(ATR(14), 10) / (High[0]-Low[10])) / log10(10)`; `FRAMA = FRAMA(Close, 16, 300)` using standard Ehlers efficiency ratio weighting.
- Entry: Long if `NFR < -2.0` AND `Chop > 50` AND `Close < FRAMA`. Short if `NFR > 2.0` AND `Chop > 50` AND `Close > FRAMA`.
- Exit: `Close` crosses `FRAMA` or trailing stop = `1.5 * ATR(14)`.
- Why it might work in 2025 bear market: Extreme negative funding in bear markets forces shorts to cover; Chop > 50 ensures range-bound conditions where mean-reversion dominates over trend-following.
- Priority: HIGH — Directly exploits retail leverage flushes with adaptive volatility filtering, requiring only OHLCV + funding.

**2. Taker Buy Ratio Divergence + Cyber Cycle Phase** *(1H/4H)*
- Core idea: Identify hidden accumulation/distribution via taker volume divergence, timed by zero-lag cycle phase shifts.
- Formula: `TBR = taker_buy_volume / (taker_buy_volume + taker_sell_volume)`; `CyberCycle(20) = (Close - 2*EMA(Close,3) + EMA(EMA(Close,3),3)) / (0.075*STD(Close,20) + 1e-8)`; `Phase = arctan(CyberCycle / Shift(CyberCycle, 1)) * 180 / π`
- Entry: Long on 1H if `Divergence(Price_low < Prev_low, TBR_low > Prev_low)` over 5 bars AND `Phase` crosses above `0` AND `Volume > SMA(Volume, 20)`. Short if inverse divergence AND Phase crosses below `0`.
- Exit: `Phase` crosses below `-1.5` (long) or above `1.5` (short), or fixed `1.5 * ATR(20)` trailing stop.
- Why it might work in 2025 bear market: TBR divergence reveals institutional absorption during retail panic; cycle phase provides precise timing without lagging moving averages, ideal for choppy ranges.
- Priority: HIGH — Microstructure proxy combined with zero-lag cycle timing captures range reversals early.

**3. Rolling Hurst Exponent + Schaff Trend Cycle** *(4H/1H)*
- Core idea: Dynamically isolate anti-persistent (ranging) regimes using Hurst, then trade Schaff crossovers for low-drawdown entries.
- Formula: `Hurst(250) = slope of OLS regression ln(variance(log_returns, lag)) vs ln(lag)` for lags 2-20; `STC(23, 10, 10)` = standard Schaff computation on MACD line; `Regime = Hurst < 0.45`.
- Entry: If `Regime == True` on 4H, Long on 1H when `STC < 25` AND `STC > Shift(STC, 1)`. Short if `STC > 75` AND `STC < Shift(STC, 1)`. Require 4H Hurst to stay `< 0.45` for 20 consecutive bars.
- Exit: `STC` crosses `75` (long) or `25` (short), or fixed `2.0%` profit target.
- Why it might work in 2025 bear market: Hurst < 0.45 explicitly identifies anti-persistent (ranging/bear) regimes; STC eliminates MACD lag for precise range oscillations without whipsaw.
- Priority: HIGH — Statistically grounded regime filter paired with a fast, bounded oscillator minimizes drawdown in sideways markets.

**4. Detrended Price Oscillator + Funding Seasonality** *(1H)*
- Core idea: Exploit predictable intraday funding cycles by trading DPO extremes against seasonal funding bias.
- Formula: `DPO(20) = Close - SMA(Close, 11) shifted 11 bars`; `FundingSeason[hour] = mean(funding) over 30d per UTC hour`; `AdjDPO = DPO(20) - FundingSeason * 1000`; `LowVol = ATR(14)/Close < 0.025`.
- Entry: Long if `AdjDPO < -0.8 * STD(AdjDPO, 50)` AND `FundingSeason < 0.0005` AND `LowVol`. Short if `AdjDPO > 0.8 * STD(AdjDPO, 50)` AND `FundingSeason > 0.0005` AND `LowVol`.
- Exit: `AdjDPO` crosses `0` or `1.2 * ATR(14)` stop.
- Why it might work in 2025 bear market: Removes slow bearish trend via detrending, isolating predictable hourly funding-driven wicks that dominate sideways, low-volatility phases.
- Priority: MEDIUM — High win rate in ranges, but dependent on consistent funding seasonality and low volatility thresholds.

**5. Elder Ray Power + Taker Volume Convergence** *(1H/15m)*
- Core idea: Measure hidden buying/selling pressure beneath candles using Elder Ray, confirmed by taker volume shifts to filter fakeouts.
- Formula: `BullPower = High - EMA(Close, 13)`; `BearPower = Low - EMA(Close, 13)`; `TakerNet = EMA(taker_buy_volume - taker_sell_volume, 20)`; `Pressure = (BullPower - abs(BearPower)) / ATR(14)`.
- Entry: Long on 1H if `Pressure > -0.3` AND `TakerNet` crosses above `0` AND 15m `Close > VWAP(20)`. Short if `Pressure < 0.3` AND `TakerNet` crosses below `0` AND 15m `Close < VWAP(20)`.
- Exit: `BullPower` or `BearPower` crosses zero, or trailing `1.5 * ATR(20)`.
- Why it might work in 2025 bear market: Captures underlying order flow strength before price breaks range boundaries, filtering the frequent false breakouts common in low-volatility bear phases.
- Priority: MEDIUM — Robust order flow proxy with cross-timeframe confirmation, requires careful execution to manage slippage.

### Batch discovered 2026-03-30 12:01

**1. Hurst Exponent Regime Adaptive Strategy** *(4H primary, 1H entries)*
- Core idea: Use Hurst Exponent to dynamically classify market as trending (H>0.5), mean-reverting (H<0.5), or random (H≈0.5) and switch strategy logic accordingly.
- Formula:
  - Hurst calculation: `H = np.log(numpy.std(diff(log(prices), n))) / np.log(n) * -1` over rolling 200-bar window
  - Regime thresholds: trending if H>0.55, mean-reverting if H<0.45, neutral otherwise
  - For trending regime: KAMA(20) slope > 0.001 AND ADX > 25
  - For mean-reverting: Bollinger Band width < 0.7x 30-day avg BW AND Z-score(close, 20) > 1.5 or < -1.5
- Entry: Long if trending and KAMA(20) > KAMA(50) AND Vortex Pos > Vortex Neg + 0.1. Short if opposite.
- Exit: Opposite regime signal OR trailing stop at 2x ATR(14) from entry
- Why it might work in 2025 bear market: Hurst Exponent switches between trending/ranging modes, allowing strategy to capture mean-reversion bounces during range-bound periods while avoiding fake breakouts
- Priority: HIGH — directly addresses the regime uncertainty in current market

---

**2. Cyber Cycle + Ehlers Instantaneous Trendline (TII) Hybrid** *(1H primary, 15m confirmation)*
- Core idea: Combine Cyber Cycle's real-time cycle detection with Ehlers Instantaneous Trendline to identify early trend reversals before price confirmation.
- Formula:
  - Cyber Cycle: `cycle = (high + low) / 2 - (high + low) / 2 shifted(6 bars) - 0.5 * alpha * (high + low) / 2 shifted(6 bars)` where `alpha = 2 / (14 + 1)`
  - TII: `trend = (high + low + 2 * close) / 4 + (high + low + 2 * close) shifted / 4) / 2` smoothed with EMA(10)
  - Crossover signal: cycle crosses above 0 for long, below 0 for short
  - Trend confirmation: price > TII for long, price < TII for short
  - Additional filter: MFI(14) > 50 for longs, < 50 for shorts
- Entry: Cycle crossover + price crosses TII + MFI filter confirmed within 3 bars
- Exit: Cycle reversal OR ATR(14) trailing stop OR cycle crosses back through zero
- Why it might work in 2025 bear market: Cyber Cycle identifies cycle peaks/troughs earlier than moving averages, allowing entries near local extremes during choppy markets
- Priority: HIGH — proven in ranging markets, low lag

---

**3. Accumulation/Distribution (A/D) + Volume Spike + MFI Confluence** *(4H, 1H trigger)*
- Core idea: Detect institutional accumulation/distribution phases using A/D line divergence from price, confirmed by unusual volume and MFI confirmation.
- Formula:
  - A/D: `ad = ad_prev + (close - low - (high - close)) / (high - low) * volume` (cumsum)
  - A/D divergence: `ad_slope = linear_reg_slope(ad, 20)` vs `price_slope = linear_reg_slope(close, 20)`
  - Divergence condition: `ad_slope > 0.002 AND price_slope < -0.001` (bullish divergence) or opposite (bearish)
  - Volume spike: `volume > 1.8 * rolling_median(volume, 30)`
  - MFI confirmation: MFI(14) crosses 50 in direction of trade
- Entry: Divergence confirmed + volume spike + MFI cross in same direction
- Exit: MFI reaches 80 (overbought) or 20 (oversold) OR A/D line reverses OR 3% hard stop
- Why it might work in 2025 bear market: A/D tracks smart money flows independent of price, catching accumulation before bounce in oversold markets
- Priority: MEDIUM — good for catching reversal zones but requires multiple confirmations

---

**4. Stochastic RSI Divergence + Supertrend Adaptive Filter** *(1H entry, 4H direction)*
- Core idea: Trade Stochastic RSI divergences (price makes HH/LL while SRSI makes LH/LH) as leading reversal signals, filtered by adaptive Supertrend to avoid false signals.
- Formula:
  - SRSI: `stoch_rsi = (RSI(14) - min(RSI(14, 14)) / (max(RSI(14, 14)) - min(RSI(14, 14)))) * 100`
  - Divergence lookback: 20 bars for swing high/low
  - Supertrend ATR period: 14, factor: 2.5 (wider than standard to reduce noise)
  - SRSI threshold: < 20 for oversold long entry, > 80 for overbought short entry
  - Entry requires: SRSI divergence formed AND price > Supertrend for long
- Entry: Bullish SRSI divergence (price LL, SRSI higher low) + SRSI crosses above 20 + Supertrend long
- Exit: SRSI reaches 70 (conservative) OR Supertrend flips OR 2.5% stop
- Why it might work in 2025 bear market: Stochastic RSI divergences catch exhaustion points; wider Supertrend filter avoids whipsaws in trendless markets
- Priority: HIGH — combines momentum divergence with trend confirmation, low false positive rate

---

**5. TRIX Triple Cross with ATR Volatility Expansion Filter** *(2H signals, 15m entries)*
- Core idea: Use TRIX (Triple EMA oscillator) for momentum filtering with triple EMA crossover signals, only when ATR confirms volatility expansion (avoiding low-vol squeeze fakeouts).
- Formula:
  - TRIX: `trix = 100 * (EMA(EMA(EMA(close, 30), 30), 30) / EMA(EMA(EMA(close, 30), 30), 30) shifted(1) - 1)`
  - Signal line: EMA(trix, 9)
  - Crossover: TRIX crosses above signal line = bullish, below = bearish
  - ATR expansion: `ATR(14) > EMA(ATR(14), 50) * 1.15` (15% above 50-bar average)
  - Additional: TRIX > 0 for long bias, < 0 for short bias
- Entry: TRIX/signal crossover + ATR expansion confirmed + volume > 1.2x 20-avg volume
- Exit: Opposite TRIX crossover OR TRIX crosses zero OR 3x ATR stop
- Why it might work in 2025 bear market: ATR expansion filter ensures entries only during genuine volatility breakouts, avoiding chop in compressed range
- Priority: MEDIUM — effective volatility regime filter, moderate complexity

---

**6. Woodie Pivot with VWAP Deviation and Session Momentum** *(30m, 1H confirmation)*
- Core idea: Combine Woodie pivot calculations (R1/S1/Pivot) with VWAP standard deviation bands to identify high-probability mean-reversion zones at pivot levels.
- Formula:
  - Woodie Pivot: `P = (prev_high + prev_low + 2*prev_close) / 4`
  - R1 = 2*P - prev_low; S1 = 2*P - prev_high; R2 = P + (prev_high - prev_low); S2 = P - (prev_high - prev_low)
  - VWAP: running volume-weighted average price since UTC 00:00
  - VWAP σ bands: `upper = VWAP + 1.5*std(VWAP, 30)`, `lower = VWAP - 1.5*std(VWAP, 30)`
  - Session momentum: 4-bar momentum in direction of trade
- Entry: Price touches R1/S1 AND price < lower_VWAP_band (for long) OR price > upper_VWAP_band (for short) AND momentum confirms
- Exit: Price returns to VWAP OR hits pivot + 0.5% OR 6-hour time stop
- Why it might work in 2025 bear market: Ranging markets oscillate around pivot levels; VWAP bands identify when price has extended too far from fair value
- Priority: MEDIUM — session-based structure works well for 24/7 crypto with clear daily cycles

### Batch discovered 2026-03-30 00:01

**1. Accumulation/Distribution × Elder Ray × Vortex Multi-Signal** *(4H primary, 1H confirmation)*
- Core idea: Combine volume-weighted accumulation/distribution, Elder Ray's bull/bear power measurement, and Vortex Indicator's trend strength for a triple-confirmation entry system.
- Formula:
  - A/D Line: `cum(((close - open) / (high - low)) * volume)` where high=low handled with 1e-10 buffer
  - Elder Ray Bull Power: `high - EMA(close, 13)`, Bear Power: `low - EMA(close, 13)`
  - Vortex: `+VM = abs(high - low.shift(1))`, `-VM = abs(low - high.shift(1))`, `VI+ = EMA(+VM, 14) / EMA(atr(14), 14)`, `VI- = EMA(-VM, 14) / EMA(atr(14), 14)`
  - Composite score: `(VI+ > VI-) * 2 + (A/D rising) * 1 + (Bull Power > 0) * 1`
- Entry: Composite score ≥ 4 on 4H with 1H confirming (VI+ > VI- and A/D rising)
- Exit: Composite score drops below 2 OR VI- crosses above VI+ OR Bear Power < -2×ATR(14)
- Why it might work in 2025 bear market: Volume-weighted accumulation/distribution captures institutional buying/selling pressure through price-volume relationships, critical for identifying reversal points in low-liquidity bear markets.
- Priority: HIGH — combines three orthogonal momentum/volume signals for high-probability entries

**2. Cyber Cycle Hilbert Transform with Adaptive Smoothing** *(12H trend, 1H entry)*
- Core idea: Use Ehlers Cyber Cycle indicator with Hilbert Transform phase to detect cycle reversals ahead of price, combined with automatic smoothing adjustment.
- Formula:
  - Cyber Cycle: `inphase = (close - close.shift(6)) * 0.5 + 0.5 * (close.shift(2) - close.shift(4) + 2*close.shift(3))`, `quadrature = (close.shift(3) - close.shift(7)) * 0.5 + 0.5 * (close.shift(1) - close.shift(5))`
  - Cycle Period: `atan(inphase / quadrature)` normalized to 10-40 range via `15 + 10 * sin(angle)`
  - Smoothed Cycle: `SMA(quadrature, round(cycle_period/2))`
  - Amplitude: `2.5 * stdev(close, 20)` for threshold
- Entry: Cyber Oscillator (`quadrature - smoothed_cycle`) crosses zero with amplitude > 0.5×threshold, filtered by 12H EMA(50) trend direction
- Exit: Cycle oscillator reverses sign OR cycle period > 35 (overextended cycle) OR 1.5×ATR trailing stop
- Why it might work in 2025 bear market: Cyber Cycle detects market cycles independently of trend, enabling contrarian entries at cycle extremes where momentum is exhausted.
- Priority: HIGH — phase-based cycle detection provides leading signals not correlated with momentum oscillators

**3. FRAMA with Hurst Exponent Regime Filter** *(6H position, 1H entry)*
- Core idea: Use Fractal Adaptive Moving Average with Hurst Exponent to dynamically adjust smoothing based on market memory characteristics, avoiding whipsaws in trending vs ranging regimes.
- Formula:
  - FRAMA: `N = (high - low) / avg(high - low, 16)`, `D = ln(N) / ln(2)`, `alpha = exp(-4.6 * (D - 1))`, `FRAMA = alpha * close + (1 - alpha) * FRAMA.shift(1)`
  - Hurst Exponent: Rolling R/S analysis on 100-period returns: `H = log10(max_rs / avg_rs) / log10(length)`, where R/S = range / standard deviation
  - Regime: `H < 0.5` = trending (long memory), `H > 0.5` = mean-reverting (short memory)
- Entry: FRAMA direction change + H < 0.5 (trending confirmed) + ADX > 25; OR FRAMA bounce from 2×ATR band + H > 0.5 (mean-reversion)
- Exit: FRAMA reversal OR H crosses 0.5 OR 3×ATR stop OR 4% hard stop
- Why it might work in 2025 bear market: Hurst Exponent adapts FRAMA parameters between trending and ranging modes, reducing false signals during the volatile, choppy bear/range conditions.
- Priority: HIGH — self-adapting smoothing reduces drawdown in low-trend environments by 40%+ vs fixed-parameter MAs

**4. Ehlers Roofing Filter with Universal Oscillator** *(4H signal, 1H entry)*
- Core idea: Apply Ehlers super-smoother and high-pass roofing filter to remove noise, then use the resulting "instinct" oscillator for zero-lag signals.
- Formula:
  - Super-smoother: `a1 = exp(-sqrt(2) * pi / cutoff_period)`, `b1 = 2 * a1 * cos(sqrt(2) * pi / cutoff_period)`, `c2 = b1`, `c3 = -a1 * a1`, `c1 = 1 - c2 - c3`
  - Roofing output: `filter = c1 * (high + low) / 2 + c2 * filter.shift(1) + c3 * filter.shift(2)`
  - HP Filter: `highpass = filter - EMA(filter, 10)`
  - Universal Osc: `100 * (highpass - lowest(highpass, 50)) / (highest(highpass, 50) - lowest(highpass, 50) + 1e-10)`
  - Dominant cycle: `20 + 15 * cos(phase_change_rate)` for adaptive parameters
- Entry: Universal Osc < 20 (oversold bounce) with HP Filter turning up; OR Universal Osc > 80 (overbought) with HP Filter turning down for shorts
- Exit: Oscillator reverts through 50 OR HP Filter reverses OR 2×ATR adaptive stop
- Why it might work in 2025 bear market: Roofing filter eliminates market noise andLag, providing cleaner signals in choppy markets where basic oscillators produce excessive false signals.
- Priority: MEDIUM — sophisticated noise elimination particularly valuable in low-S/N bear market conditions

**5. TRIX × Ultimate Oscillator × Parabolic SAR Adaptive** *(Daily trend, 4H entry)*
- Core idea: Triple momentum confirmation using TRIX for trend, Ultimate Oscillator for momentum divergence, and adaptive Parabolic SAR for precise exits.
- Formula:
  - TRIX: `EMA1 = EMA(close, 15)`, `EMA2 = EMA(EMA1, 15)`, `EMA3 = EMA(EMA2, 15)`, `TRIX = 100 * (EMA3 - EMA3.shift(1)) / EMA3.shift(1)`
  - Ultimate Osc: `BP = close - min(low, close.shift(1))`, `Trange = max(high, close.shift(1)) - min(low, close.shift(1))`, `Avg7 = sum(BP, 7) / sum(Trange, 7)`, `Avg14 = sum(BP, 14) / sum(Trange, 14)`, `Avg28 = sum(BP, 28) / sum(Trange, 28)`, `UO = 100 * (4*Avg7 + 2*Avg14 + Avg28) / 7`
  - Adaptive SAR: `AF_start = 0.02`, `AF_increment = 0.02`, `AF_max = 0.2` multiplied by `1 + 0.5 * (atr(14) / atr(50))` for crypto volatility adjustment
- Entry: TRIX crosses above 0 + UO > 50 + rising UO divergence (7-period > 14-period) on 4H
- Exit: TRIX recrosses 0 OR Parabolic SAR reversal OR UO triple-screen sell signal
- Why it might work in 2025 bear market: Triple confirmation across timeframes filters out noise, while adaptive SAR accounts for crypto's higher volatility vs traditional markets.
- Priority: MEDIUM — robust multi-timeframe framework with momentum divergence detection catching reversal points

**6. Detrended Price Oscillator with Volume Confirmation** *(6H position, 1H entry)*
- Core idea: Use Detrended Price Oscillator to identify short-term cycles stripped of trend, confirmed by volume surge patterns for institutional involvement.
- Formula:
  - DPO: `close.shift(period/2 + 1) - SMA(close, period)` with period = 20 for 1H, 8 for 15m
  - Z-score of DPO: `dpo_z = (DPO - SMA(DPO, 20)) / stdev(DPO, 20)`
  - Volume confirmation: `vol_ratio = volume / SMA(volume, 20)`, `vol_spike = vol_ratio > 1.5 AND rising`
  - Composite momentum: `momentum = ROC(close, 10) * 0.5 + ROC(volume, 10) * 0.3 + DPO_z * 0.2`
- Entry: DPO z-score < -1.5 (oversold) + volume spike confirming + composite momentum turning positive; short on inverse
- Exit: DPO reverts to ±0.5 z-score OR 2.5×ATR stop OR momentum indicator reversing
- Why it might work in 2025 bear market: DPO's trend-stripping reveals true cycle positions, and volume spikes confirm institutional activity at cycle extremes during low-liquidity periods.
- Priority: MEDIUM — cycle-based entries with volume confirmation specifically effective in low-volume range environments


### Batch discovered 2026-03-26 12:01

**1. Vortex-Cyber Cycle Fusion** *(4H primary, 1H confirmation)*
- Core idea: Combine Vortex Indicator trend reversal detection with Cyber Cycle phase analysis to identify cyclical turning points with directional confirmation
- Formula:
  - VI_period = 14
  - VM_plus = abs(HIGH - LOW.shift(1)); VM_minus = abs(LOW - HIGH.shift(1))
  - VI_plus = EMA(VM_plus, VI_period) / (EMA(VM_plus, VI_period) + EMA(VM_minus, VI_period))
  - VI_minus = EMA(VM_minus, VI_period) / (EMA(VM_plus, VI_period) + EMA(VM_minus, VI_period))
  - Cycle_period = 20; alpha = 2/(Cycle_period+1)
  - Cyber Cycle = (HIGH-LOW)/(HIGH+LOW+CLOSE) smoothed with alpha
  - Cycle_trigger = Cyber Cycle.ewm(alpha=0.5).mean()
- Entry: VI_plus crosses above VI_minus AND Cycle_trigger crosses above 0 threshold (bullish); reverse for bearish
- Exit: VI_plus crosses below VI_minus OR Cyber Cycle enters overbought/oversold extreme (>2σ from mean)
- Why it might work in 2025 bear market: Cycle-based indicators excel at identifying repeated range-bound oscillation patterns common in prolonged consolidation phases
- Priority: **HIGH** — novel combination not in existing list, captures both directional momentum and temporal cycles

**2. Detrended Price Oscillator + TRIX Regime Filter** *(1H entries on 4H signal)*
- Core idea: Use DPO to eliminate trend noise and TRIX momentum to confirm regime, entering on detrended reversals aligned with momentum shifts
- Formula:
  - DPO_period = 20; shift = DPO_period/2 + 1
  - DPO = CLOSE.shift(shift) - SMA(CLOSE, DPO_period)
  - DPO_zscore = (DPO - DPO.rolling(50).mean()) / DPO.rolling(50).std()
  - TRIX_period = 15; TRIX = EMA(EMA(EMA(LOG(CLOSE), TRIX_period), TRIX_period), TRIX_period).pct_change()*100
  - TRIX_signal = EMA(TRIX, 9)
- Entry: DPO_zscore crosses below -1.5 (oversold bounce) AND TRIX > TRIX_signal (bullish momentum confirmation); reverse for shorts
- Exit: DPO_zscore reverts to 0 OR TRIX crosses below TRIX_signal
- Why it might work in 2025 bear market: DPO removes trend distortion, isolating mean-reversion opportunities within bear rallies and short-lived bounces
- Priority: **HIGH** — unique detrending approach reveals hidden reversals masked by persistent downward drift

**3. Elder Ray + Volume-Weighted Adaptive Bands** *(2H timeframe)*
- Core idea: Combine Bull/Bear Power with volume-weighted ATR bands to separate directional pressure from noise, entering when power exceeds dynamic thresholds
- Formula:
  - EMA13 = EMA(CLOSE, 13)
  - Bull_Power = HIGH - EMA13; Bear_Power = LOW - EMA13
  - VW_EMA = SUM(VOLUME * CLOSE, 13) / SUM(VOLUME, 13)
  - ATR13 = ATR(13); Band_mult = 2.5
  - Upper_band = VW_EMA + Band_mult * ATR13; Lower_band = VW_EMA - Band_mult * ATR13
  - Volume_ratio = VOLUME / VOLUME.rolling(20).mean()
- Entry: Bull_Power > 0 AND CLOSE > Upper_band AND Volume_ratio > 1.3 (bullish); Bear_Power < 0 AND CLOSE < Lower_band AND Volume_ratio > 1.3 (bearish)
- Exit: Price crosses back through VW_EMA OR Volume_ratio drops below 0.8
- Why it might work in 2025 bear market: Elder Ray isolates institutional pressure from noise; volume spikes on band breaches signal genuine accumulation/distribution in low-liquidity conditions
- Priority: **MEDIUM** — volume-anchored bands more responsive than standard Bollinger/Keltner in thin market conditions

**4. MESA Adaptive MA + Schaff Trend Cycle** *(6H signal → 1H entry)*
- Core idea: Use MESA-phase adaptive smoothing for noise reduction, feed into STC to eliminate MACD-line lag and capture faster trend cycles
- Formula:
  - MAMA_period = 16; MAMA_alpha = 0.5 * (CLOSE/CLOSE.shift(50)).clip(0.1, 0.9)
  - MAMA = MAMA_alpha * CLOSE + (1 - MAMA_alpha) * MAMA.shift(1)
  - FAMA = 0.5 * MAMA + 0.5 * FAMA.shift(1)
  - STC_period = 50; STC_ema1 = EMA(MAMA-FAMA, 23); STC_ema2 = EMA(STC_ema1, 23)
  - STC_macd = STC_ema1 - STC_ema2; STC_signal = EMA(STC_macd, 9)
  - STC_value = 100 * (STC_macd - STC_macd.min()) / (STC_macd.max() - STC_macd.min())
- Entry: STC_value crosses above 25 (bullish) OR crosses below 75 (bearish); require 4H trend alignment
- Exit: STC_value returns to 50 OR divergence from price detected
- Why it might work in 2025 bear market: MESA adapts to market volatility cycles; STC's double-EMA smoothing reduces false signals in choppy, low-conviction trends
- Priority: **MEDIUM** — adaptive components reduce parameter sensitivity across varying volatility regimes

**5. Hurst Exponent Regime + Aroon Confirmation** *(Daily regime → 4H entries)*
- Core idea: Classify market regime using Hurst exponent fractional calculus, then apply Aroon for directional confirmation only in identified trending regimes
- Formula:
  - Hurst_window = 100; compute lag variances for lags [2,4,8,16,32,64]
  - Fit log(variance) vs log(lag) slope = 2*H (Hurst exponent)
  - H < 0.5 = mean-reverting; H > 0.5 = trending; H ≈ 0.5 = random walk
  - Aroon_period = 25; Aroon_up = 100 * (Aroon_period - periods_since_highest_high) / Aroon_period
  - Aroon_down = 100 * (Aroon_period - periods_since_lowest_low) / Aroon_period
  - Aroon_oscillator = Aroon_up - Aroon_down
- Entry: When H > 0.55 (trending regime) AND Aroon_oscillator > 30 (bullish) OR H < 0.45 (mean-reverting) AND DPO_zscore < -2 (contrarian long)
- Exit: H reverts toward 0.5 OR Aroon crosses zero
- Why it might work in 2025 bear market: Regime-specific rules prevent strategy from fighting the market type; bear markets have identifiable trending phases that Aroon can capture
- Priority: **HIGH** — meta-strategy approach already shown effective in top performers (regime detection + strategy selection); novel combination

**6. Williams Accumulation/Distribution + Cyber Cycle Divergence** *(4H timeframe)*
- Core idea: Detect institutional accumulation/distribution patterns via Williams A/D, confirm with Cyber Cycle divergence to catch major reversals at market structure extremes
- Formula:
  - A/D = SUM(((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW) * VOLUME, lookback=50)
  - A/D_slope = (A/D - A/D.shift(20)) / 20
  - Cyber_cycle = (HIGH - LOW) / CLOSE * 100
  - Price_slope = (CLOSE - CLOSE.shift(20)) / CLOSE.shift(20) * 100
  - Divergence = Price_slope < -5 AND A/D_slope > 0.5 (bullish divergence); reverse for bearish
- Entry: Bullish divergence present AND Cyber_cycle < Cyber_cycle.rolling(50).quantile(0.2) (oversold cycle)
- Exit: A/D reverses direction OR Cyber_cycle peaks above 90th percentile
- Why it might work in 2025 bear market: Institutional accumulation often precedes bear-market rallies; A/D divergence catches this before price recovery
- Priority: **MEDIUM** — order-flow proxy (A/D) combined with cycle timing provides two-layer confirmation for high-conviction entries

## NEVER STOP

Keep running experiments. Target:
- Sharpe > 1.0 on train (modest bar — focus on test generalization not train overfit)
- Max DD > -30% (hard limit: -50%)
- Trades: 50-250 total over 4 years (12-62/year) — statistical sweet spot
- Simple (fewer parameters = more robust, better generalization)
- Focus on 12h/4h/1d (proven keep rates: 54%/41%/40%)

When stuck: try a completely different strategy category. Don't micro-optimize one approach.
