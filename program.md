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

## NEVER STOP

Keep running experiments. Target:
- Sharpe > 1.0 on train (modest bar — focus on test generalization not train overfit)
- Max DD > -30% (hard limit: -50%)
- Trades: 50-250 total over 4 years (12-62/year) — statistical sweet spot
- Simple (fewer parameters = more robust, better generalization)
- Focus on 12h/4h/1d (proven keep rates: 54%/41%/40%)

When stuck: try a completely different strategy category. Don't micro-optimize one approach.
