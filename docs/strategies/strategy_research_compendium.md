# Comprehensive Trading Strategy Research Compendium

Research compiled from quantitative finance literature, backtested strategy databases, and crypto-specific sources.

---

## 1. TREND FOLLOWING STRATEGIES

### 1.1 Hull Moving Average (HMA) Crossover

**Concept**: HMA reduces lag vs SMA/EMA by using weighted moving averages with a square root period. Developed by Alan Hull (2005).

**Formula**: HMA(n) = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

**Entry/Exit Rules**:
- Long: Fast HMA (e.g., HMA-16) crosses above Slow HMA (e.g., HMA-48)
- Short: Fast HMA crosses below Slow HMA
- Alternative: Use HMA slope change (HMA rising = long, falling = short)

**Typical Parameters**:
- Fast HMA: 9-16 periods
- Slow HMA: 36-64 periods
- Best timeframes: 1h, 4h for crypto futures

**Weaknesses**: Whipsaws in ranging markets; still has some lag despite improvements. Needs trend filter (e.g., ADX > 20) to avoid chop.

**Combination**: Use HMA direction as trend filter for mean-reversion entries. Combine with volume confirmation.

---

### 1.2 Kaufman's Adaptive Moving Average (KAMA)

**Concept**: Adapts smoothing speed based on market noise via an Efficiency Ratio (ER). Fast in trends, slow in chop.

**Parameters** (Kaufman's recommended):
- ER period: 10
- Fast EMA constant: 2 periods (SC = 2/(2+1) = 0.6667)
- Slow EMA constant: 30 periods (SC = 2/(30+1) = 0.0645)

**Entry/Exit Rules**:
- Long: Price crosses above KAMA and KAMA slope is positive
- Short: Price crosses below KAMA and KAMA slope is negative
- Filter: Only trade when ER > 0.3 (meaningful trend exists)

**Weaknesses**: Slow to react to sudden reversals. ER can be noisy on very short timeframes.

**Combination**: Use KAMA as the trend backbone with RSI for entry timing.

---

### 1.3 DEMA Crossover

**Concept**: Double Exponential Moving Average reduces lag further than EMA. DEMA = 2*EMA(n) - EMA(EMA(n)).

**Entry/Exit Rules**:
- Long: Fast DEMA(n) crosses above Slow DEMA(2n)
- Short: Fast DEMA(n) crosses below Slow DEMA(2n)

**Typical Parameters**:
- Fast: DEMA-12, Slow: DEMA-26 (classic)
- Fast: DEMA-8, Slow: DEMA-21 (crypto-adjusted)

**Weaknesses**: More false signals than HMA in choppy markets due to higher sensitivity. Requires additional filtering.

---

### 1.4 Donchian Channel Breakout

**Concept**: Enter on new N-period high/low breakouts. The original Turtle Trading system.

**Entry/Exit Rules**:
- Long: Close > highest high of last N bars
- Short: Close < lowest low of last N bars
- Exit long: Close < lowest low of last M bars (M < N)
- Exit short: Close > highest high of last M bars

**Typical Parameters**:
- Entry channel: 20 bars (intraday), 55 bars (swing), 15 bars (crypto-optimized)
- Exit channel: 10 bars (half the entry period)
- Volume filter: Breakout candle volume > 150% of 20-period average
- Buffer: Add 0.5-1 ATR buffer above/below channel

**ADX Filter (two approaches)**:
1. Traditional: Only trade when ADX > 25 (strong trend)
2. Crypto-specific: Only trade when ADX < 25 (low-volatility breakout, as BTC often explodes from consolidation)

**Best Markets/Timeframes**: 4h and daily for crypto. 15-day lookback backtested best for BTC.

**Weaknesses**: Many false breakouts in ranging markets. Late entries in fast trends. High drawdowns during mean-reverting regimes.

**Combination**: Add ATR-based position sizing (risk 1-2% per ATR move). Combine with volume confirmation and regime detection.

---

### 1.5 Supertrend Indicator

**Concept**: Trend-following overlay using ATR to set dynamic support/resistance.

**Formula**:
- Upper band = (High + Low)/2 + (Multiplier * ATR)
- Lower band = (High + Low)/2 - (Multiplier * ATR)

**Entry/Exit Rules**:
- Long: Price closes above Supertrend line (trend flips bullish)
- Short: Price closes below Supertrend line (trend flips bearish)
- Supertrend line acts as trailing stop

**Parameter Settings by Use Case**:
| Use Case | ATR Period | Multiplier | Notes |
|----------|-----------|------------|-------|
| Scalping (1m-5m) | 5 | 1.2 | Very sensitive, many signals |
| Intraday (15m-1h) | 10 | 2.0 | Good balance |
| Default | 10 | 3.0 | Standard setting |
| Swing (4h-1d) | 14 | 3.0-4.0 | Fewer false signals |
| High volatility crypto | 7 | 3.0 | Short ATR, medium multiplier |
| Noise reduction | 20 | 5.0-7.0 | Very few signals |

**Double Supertrend Strategy**:
- Fast: ATR 10, Multiplier 3
- Slow: ATR 20, Multiplier 7
- Enter only when both agree on direction

**Backtest (BTC Quantitative)**: ATR 9, Multiplier 2.5 with ATR stop-loss showed strong results.

**Weaknesses**: Whipsaws in sideways markets. Large stop distances with high multipliers reduce R:R.

**Combination**: Add MACD or RSI confirmation. Use ADX > 20 filter. Layer with volume.

---

### 1.6 ADX-Filtered Momentum

**Concept**: Only trade momentum signals when ADX confirms trend strength.

**Entry/Exit Rules**:
- Long: +DI crosses above -DI AND ADX > 25
- Short: -DI crosses above +DI AND ADX > 25
- Exit: ADX drops below 20 OR opposite DI crossover

**Parameters**:
- ADX period: 14 (standard)
- Trend threshold: 25 (strong), 20 (moderate)
- DI period: 14

**Weaknesses**: ADX is lagging -- by the time it confirms a trend, significant move may have occurred. DI crossovers are noisy.

---

## 2. MEAN REVERSION STRATEGIES

### 2.1 Bollinger Band Mean Reversion

**Concept**: Price tends to revert to the mean (middle band) after touching extremes.

**Entry/Exit Rules**:
- Long: Close < Lower BB AND RSI < 30
- Short: Close > Upper BB AND RSI > 70
- Exit long: Close > Middle BB (SMA-20)
- Exit short: Close < Middle BB
- Stop: 2x ATR(14) beyond entry

**Parameters**:
- BB period: 20 (standard), can test 14-30
- BB std dev: 2.0 (standard), use 2.5-3.0 for crypto
- RSI period: 14
- RSI thresholds: 30/70 (standard), 20/80 (stricter)

**Best Markets/Timeframes**: Works best in ranging/mean-reverting markets. 1h-4h for crypto. Avoid during strong trends.

**Weaknesses**: Devastating losses during trend breakouts (price can stay outside bands). Requires regime filter.

**Combination**: Add ADX < 25 filter (only trade in non-trending markets). Confirm with volume decreasing (exhaustion). Add Keltner squeeze detection.

---

### 2.2 Z-Score Mean Reversion

**Concept**: Standardize price deviation from moving average; trade when z-score reaches extremes.

**Formula**: Z = (Price - SMA(n)) / StdDev(n)

**Entry/Exit Rules**:
- Long: Z-score < -2.0
- Short: Z-score > +2.0
- Exit: Z-score returns to 0 (mean)
- Stop: Z-score reaches -3.0 / +3.0

**Parameters**:
- Lookback period: 20-50 bars
- Entry threshold: |2.0| (standard), |2.5| for crypto
- Exit threshold: 0.0 (mean) or 0.5 for partial

**Best Timeframes**: 1h and 4h for crypto. Daily for traditional markets.

**Weaknesses**: Assumes normal distribution, which crypto violates frequently. Tail risk is significant.

**Combination**: Add volume confirmation (declining volume at extremes). Filter with Bollinger bandwidth for regime.

---

### 2.3 Keltner Channel Mean Reversion

**Concept**: Similar to Bollinger Bands but uses ATR instead of standard deviation, making it less sensitive to extreme outliers.

**Parameters**:
- Middle line: 20-period EMA of typical price
- Band width: 2x ATR(14)
- Optimized for mean reversion: 6-day period, 1.3 ATR multiplier
- Optimized for momentum: 30-day period, 1.3 ATR multiplier

**Entry/Exit Rules (Mean Reversion)**:
- Long: Close below lower Keltner band
- Exit long: Close above typical price (middle line)
- Backtested win rate on S&P 500: 77%, 288 trades, profit factor 2.0

**Weaknesses**: Parameters need adjustment per market. Less responsive than BB in volatile crypto.

---

### 2.4 Bollinger-Keltner Squeeze (TTM Squeeze)

**Concept**: Identify low-volatility compression (BB inside KC), then trade the breakout direction.

**Squeeze Detection**: Bollinger Bands are INSIDE Keltner Channels

**Parameters** (backtested):
- BB period: 7, deviation: 1.0
- KC period: 30, ATR multiplier: 1.0
- MACD: 7/30/14 (for momentum confirmation)
- ATR trailing stop: 7-period ATR * 3.0

**Entry/Exit Rules**:
- Squeeze detected: BB upper < KC upper AND BB lower > KC lower
- Long entry: During squeeze, close > upper BB AND MACD crosses above signal
- Short entry: During squeeze, close < lower BB AND MACD crosses below signal
- Exit: ATR trailing stop (3.0 * ATR from highest/lowest since entry)

**For Crypto**: Use wider BB (2.5-3.0 std dev). BandWidth < 5% signals squeeze on BTC. BandWidth > 15% signals extreme volatility.

**Weaknesses**: Squeeze can last a long time before breaking. Direction of breakout is uncertain until confirmed.

**Combination**: This IS already a combination strategy. Add volume surge confirmation on breakout candle.

---

## 3. MOMENTUM STRATEGIES

### 3.1 Rate of Change (ROC)

**Concept**: Pure momentum oscillator measuring percentage price change over N periods.

**Formula**: ROC = ((Close - Close[n]) / Close[n]) * 100

**Entry/Exit Rules**:
- Long: ROC crosses above zero
- Short: ROC crosses below zero
- Alternative: Long when ROC > threshold (e.g., +2%), short when ROC < -2%
- Divergence: Bullish when price makes lower lows but ROC makes higher lows

**Parameters**:
- Standard period: 12
- Short-term: 9
- Long-term: 25
- Trend filter: 50-period MA (only long above, short below)
- Volume filter: Above-average volume on signal candle

**Weaknesses**: Noisy on short timeframes. Zero-line crossovers produce many false signals in ranging markets.

**Combination**: Combine ROC + OBV + ADX for triple confirmation. ROC for momentum direction, OBV for volume pressure, ADX for trend strength.

---

### 3.2 MACD Histogram Divergence

**Concept**: Detect momentum exhaustion through divergence between price and MACD histogram.

**Parameters by Trading Style**:
| Style | Fast | Slow | Signal | Notes |
|-------|------|------|--------|-------|
| Standard | 12 | 26 | 9 | Default |
| Fast/scalping | 6 | 13 | 5 | More signals, more noise |
| Long-term | 24 | 52 | 18 | Fewer, higher quality |
| Squeeze breakout | 7 | 30 | 14 | Optimized for squeeze |

**Entry/Exit Rules (Divergence)**:
- Bullish divergence: Price makes lower low, MACD histogram makes higher low -> Long
- Bearish divergence: Price makes higher high, MACD histogram makes lower high -> Short
- Confirmation: Wait for histogram to cross zero after divergence
- Exit: Opposite divergence or histogram reversal

**Histogram Peak Strategy**:
- Sell signal: Histogram peaks and starts to decline
- Buy signal: Histogram bottoms and begins to rise

**Weaknesses**: Divergence can persist for extended periods before price reverses. False divergences common in strong trends.

**Combination**: Confirm with RSI, volume, or support/resistance levels. Use higher timeframe trend as filter.

---

### 3.3 Stochastic Momentum Index (SMI)

**Concept**: Refined stochastic using distance from midpoint of high/low range (not just close vs low). Created by William Blau (1993).

**Parameters**:
- K period: 10 (or 5 for faster)
- D smoothing: 3
- Signal line: 3-period EMA
- Range: -100 to +100

**Entry/Exit Rules**:
- Long: SMI crosses above -40 (oversold zone ends) with %K crossing above %D
- Short: SMI crosses below +40 (overbought zone ends) with %K crossing below %D
- Alternative thresholds: -35/+35

**Backtest Results (BTC, 2014-present)**:
- Win rate: 43%
- Average gain per trade: 1.7%
- CAGR: 67%
- Max drawdown: 48%
- Time in market: 50%

**Best Markets**: Gold and Bitcoin (trend-following assets). Performs POORLY on stocks.

**Weaknesses**: Low win rate (relies on big winners). Poor in consolidation. Needs trend confirmation.

---

### 3.4 Williams %R

**Concept**: Momentum oscillator showing where close is relative to high-low range. Inverse of fast stochastic.

**Formula**: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

**Parameters**:
- Standard period: 14
- Range: 0 to -100
- Overbought: -20 to 0
- Oversold: -80 to -100

**Entry/Exit Rules**:
- Long: %R crosses above -80 (exits oversold)
- Short: %R crosses below -20 (exits overbought)
- Trend filter: %R above -50 = upper half (bullish), below -50 = lower half (bearish)

**Weaknesses**: Very noisy as a standalone indicator. Overbought/oversold can persist in strong trends.

**Combination**: Never trade Williams %R alone. Combine with trend-following MA for direction and volume for confirmation.

---

## 4. VOLUME-BASED STRATEGIES

### 4.1 VWAP Strategies

**Concept**: Volume Weighted Average Price acts as institutional fair value reference.

**Crypto Challenge**: No market open/close for 24/7 markets. Solutions:
- Use rolling 24h VWAP
- Use daily reset VWAP (midnight UTC)
- Use session-based VWAP (Asian/European/US sessions)

**Entry/Exit Rules (Mean Reversion)**:
- Long: Price drops >1% below VWAP with declining volume
- Short: Price rises >1% above VWAP with declining volume
- Target: Return to VWAP

**Entry/Exit Rules (Trend)**:
- Long: Price crosses above VWAP with increasing volume
- Short: Price crosses below VWAP with increasing volume
- Stay in direction as long as price remains on same side of VWAP

**Parameters**:
- Standard VWAP (no period -- cumulative from session start)
- VWAP bands: +/- 1, 2, 3 standard deviations
- Volume confirmation: Current volume > 1.5x average

**Best Timeframes**: Intraday (1m-15m) for mean reversion. 1h for trend following.

**Weaknesses**: Resets daily -- no multi-day context. Requires real volume data (exchange-specific). Fragmented crypto exchanges make aggregate VWAP hard.

**Combination**: Use VWAP for entry timing within a higher-timeframe trend. Combine with OBV for volume trend.

---

### 4.2 OBV Divergence

**Concept**: On-Balance Volume accumulates volume based on close direction. Divergence from price signals hidden accumulation/distribution.

**Formula**: If close > prev_close: OBV += volume. If close < prev_close: OBV -= volume.

**Entry/Exit Rules**:
- Bullish divergence: Price makes lower lows, OBV makes higher lows -> Accumulation, go long
- Bearish divergence: Price makes higher highs, OBV makes lower highs -> Distribution, go short
- Confirmation: Wait for price structure break (e.g., trendline break) after divergence
- Exit: OBV divergence in opposite direction or OBV trend reversal

**Important Rule**: Use divergence as SETUP, not TRIGGER. Wait for price confirmation.

**Best Markets/Timeframes**: BTC and ETH on daily and 4h charts. Divergence at long-term support/resistance is most reliable.

**Weaknesses**: OBV is cumulative -- absolute values are meaningless (only trend matters). Gap events can distort OBV. Crypto volume data may include wash trading.

**Combination**: Combine OBV divergence with RSI divergence for double confirmation. Add price structure break as entry trigger.

---

### 4.3 Volume Profile

**Concept**: Horizontal histogram of volume at each price level. Identifies value areas and points of control (POC).

**Key Levels**:
- POC (Point of Control): Price with highest traded volume -- acts as magnet
- Value Area High (VAH): Upper boundary of 70% volume area -- resistance
- Value Area Low (VAL): Lower boundary of 70% volume area -- support
- Low Volume Nodes (LVN): Price levels with minimal volume -- price moves quickly through these

**Entry/Exit Rules**:
- Long: Price pulls back to VAL or POC in uptrend
- Short: Price rallies to VAH or POC in downtrend
- Breakout: Price breaks through LVN with momentum -> follow direction
- Target: Next POC or value area boundary

**Parameters**: Typically use fixed-range volume profile over last 20-50 sessions.

**Weaknesses**: Requires significant historical data. Subjective interpretation. Not easily automated without clear rules.

---

### 4.4 Accumulation/Distribution (A/D) Line

**Concept**: Similar to OBV but weights volume by where close falls within the high-low range.

**Formula**:
- Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
- Money Flow Volume = MFM * Volume
- A/D Line = cumulative sum of Money Flow Volume

**Entry/Exit Rules**:
- Bullish: A/D line rising while price flat/falling (accumulation)
- Bearish: A/D line falling while price flat/rising (distribution)
- Confirmation: A/D line breakout from its own trendline

**Weaknesses**: Ignores gaps. Less intuitive than OBV for programmatic use.

---

## 5. VOLATILITY STRATEGIES

### 5.1 ATR-Based Position Sizing

**Concept**: Normalize risk across trades by sizing positions inversely to volatility.

**Formula**:
- Position Size = (Account * Risk%) / (ATR(14) * Multiplier)
- Example: $100,000 account, 1% risk, ATR = $500, Multiplier = 2
- Size = ($100,000 * 0.01) / ($500 * 2) = 1 contract

**Stop Loss Placement**:
- Tight: 1.5 * ATR from entry
- Standard: 2.0 * ATR from entry
- Wide: 3.0 * ATR from entry

**Trailing Stop**:
- Update stop to: Entry price + (N * ATR) after price moves favorably
- Chandelier exit: Highest high - 3 * ATR(22)

**Parameters**:
- ATR period: 14 (standard), 7 (responsive), 21 (smooth)
- Risk per trade: 0.5%-2% of account
- ATR multiplier for stops: 1.5-3.0

**Best Practice**: In high-volatility crypto, use smaller risk% (0.5-1%) and wider ATR multiplier (2.5-3.0).

---

### 5.2 Squeeze Momentum (Volatility Breakout)

**Concept**: Detect volatility compression (squeeze) then trade the expansion.

**Squeeze Detection Methods**:
1. Bollinger BandWidth < threshold (e.g., < 5% for BTC)
2. Bollinger Bands inside Keltner Channels (TTM Squeeze)
3. ATR at N-period low
4. Historical volatility percentile < 20th

**Entry Rules**:
- Squeeze ends (bands expand beyond Keltner)
- Momentum direction determines trade side (MACD or linear regression slope)
- Volume surge confirms (>150% of 20-period average)

**Exit Rules**:
- ATR trailing stop: 3.0 * ATR(7)
- Momentum reversal (MACD histogram changes direction)
- Opposite squeeze signal

**Crypto-Specific**: Use wider BB (2.5 std dev) due to higher baseline volatility. BandWidth < 5% is squeeze territory for BTC.

---

### 5.3 Regime Detection with Bollinger Bandwidth

**Concept**: Classify market into volatility regimes to select appropriate strategy.

**Regime Classification**:
- Ultra-low vol: BandWidth < 3% -> Expect breakout, prepare breakout strategy
- Low vol: BandWidth 3-8% -> Mean reversion strategies work best
- Normal vol: BandWidth 8-15% -> Trend following works
- High vol: BandWidth > 15% -> Reduce position sizes, use wider stops
- Extreme vol: BandWidth > 25% -> Stay flat or use very short-term only

**Implementation**:
```
bw = (upper_bb - lower_bb) / middle_bb * 100
bw_percentile = rolling_percentile(bw, 100)

if bw_percentile < 20: regime = "squeeze"
elif bw_percentile < 50: regime = "low_vol"
elif bw_percentile < 80: regime = "normal"
else: regime = "high_vol"
```

**Strategy Selection by Regime**:
- Squeeze: Wait for breakout, use momentum strategies
- Low vol: Mean reversion (BB, z-score)
- Normal: Trend following (MA crossover, Supertrend)
- High vol: Reduce exposure, tighten stops, or stay flat

---

## 6. MULTI-TIMEFRAME STRATEGIES

### 6.1 The 3-Timeframe Framework

**Concept**: Higher timeframe for trend, middle for setup, lower for entry.

**Recommended Combinations**:
| Trading Style | Trend TF | Setup TF | Entry TF |
|--------------|----------|----------|----------|
| Scalping | 1h | 15m | 1m-5m |
| Day trading | 4h | 1h | 15m |
| Swing trading | 1d | 4h | 1h |
| Position | 1w | 1d | 4h |

### 6.2 Implementation Rules

**Step 1 - Trend (4h)**:
- Determine direction using MA slope, MACD position, or price structure
- If 4h MACD line > signal line -> uptrend -> only look for longs
- If 4h MACD line < signal line -> downtrend -> only look for shorts

**Step 2 - Setup (1h)**:
- Wait for pullback to support (uptrend) or resistance (downtrend)
- Identify setup zone using VWAP, moving averages, or Fibonacci levels
- Confirm with volume decline (pullback on low volume)

**Step 3 - Entry (15m)**:
- Enter when structure shifts in trend direction
- Use Supertrend flip, MA crossover, or candlestick pattern
- Place stop below 15m structure (tight stop)

### 6.3 Backtested Multi-TF MACD Strategy (QuantPedia)

**Data**: BTC/USD, Dec 2018 - Nov 2025

**Version 1 - 1H MACD Only**:
- Annual return: 4.6%, Sharpe: 0.33, Max DD: -23.9%, 2,262 trades

**Version 2 - Daily + 1H MACD Filter**:
- Only trade 1H MACD signals in direction of Daily MACD
- Annual return: 6.6%, Sharpe: 0.80, Max DD: -12.4%, ~1,000 trades

**Version 3 - D1H1 + Trailing Stop**:
- Exit on first negative bar (close < open) on 1H
- Sharpe: 1.07, Calmar: 0.87

**Key Insight**: Adding the daily timeframe filter nearly halved drawdown and doubled the Sharpe ratio with half the trades.

---

## 7. MARKET MICROSTRUCTURE FOR CRYPTO

### 7.1 Funding Rate Strategy

**Concept**: Perpetual futures use funding rates (every 8h) to anchor price to spot. Positive rate = longs pay shorts. Negative rate = shorts pay longs.

**Funding Rate Arbitrage (Delta-Neutral)**:
- Entry: When funding rate > 0.05% (annualized ~55%)
  - Buy spot (or low-funding exchange)
  - Short perpetual (high-funding exchange)
- Hold: Collect funding payments while delta-neutral
- Exit: When funding normalizes or becomes negative

**Funding Rate as Signal (Directional)**:
- Extreme positive funding (>0.1%): Market overleveraged long -> Contrarian short signal
- Extreme negative funding (<-0.05%): Market overleveraged short -> Contrarian long signal
- Rising funding + rising OI: Crowded trade, expect liquidation cascade

**Risk Management**:
- Monitor liquidity conditions
- Size positions to survive basis risk (perp-spot spread can widen)
- Account for funding carry cost/gain in Kelly calculations

**Reported Returns**: Up to 115.9% over 6 months in research, with max loss of 1.92% (delta-neutral).

---

### 7.2 Open Interest Signals

**Concept**: Rising OI = new money entering. Falling OI = positions closing.

**Signal Matrix**:
| Price | OI | Interpretation | Signal |
|-------|----|----------------|--------|
| Rising | Rising | New longs entering | Bullish continuation |
| Rising | Falling | Short covering | Weak rally, caution |
| Falling | Rising | New shorts entering | Bearish continuation |
| Falling | Falling | Long liquidation | Selling exhaustion |

**Implementation**:
- Track OI change rate (OI ROC) over 4h-24h windows
- Extreme OI buildup (>2 std dev above mean) -> Expect volatility event
- Use OI divergence similar to volume divergence

---

### 7.3 Liquidation Cascade Detection

**Concept**: Leveraged positions have liquidation prices. Price approaching liquidation clusters triggers cascading forced sells/buys.

**Signals**:
- High funding rate + high OI + price approaching support/resistance -> Liquidation risk
- Large OI concentrated at specific price levels -> Liquidation magnet
- Post-liquidation (large OI drop + large volume) -> Potential reversal

**Implementation**: Monitor aggregate liquidation data from exchanges. Use heatmaps to identify concentration zones.

---

### 7.4 Basis Trading (Cash & Carry)

**Concept**: Exploit the spread between futures and spot price.

**Entry**:
- When futures premium (basis) > threshold (e.g., annualized >15%)
- Buy spot, sell futures
- Collect the basis convergence + funding

**Exit**: At futures expiry (quarterly contracts) or when basis narrows.

**Crypto-Specific**: Perpetual futures don't expire, so basis trading relies on funding rate convergence rather than expiry.

---

## 8. ENSEMBLE / COMBINATION STRATEGIES

### 8.1 Signal Voting System

**Concept**: Combine multiple uncorrelated signals; trade only when majority agree.

**Implementation**:
```
signals = {
    'trend_ma': get_ma_signal(),        # -1, 0, +1
    'momentum_rsi': get_rsi_signal(),    # -1, 0, +1
    'volume_obv': get_obv_signal(),      # -1, 0, +1
    'volatility_bb': get_bb_signal(),    # -1, 0, +1
}

total = sum(signals.values())
if total >= 2: position = LONG
elif total <= -2: position = SHORT
else: position = FLAT
```

**Key Principle**: Strategies must be DIVERSE:
- One price-based (MA, Supertrend)
- One momentum-based (RSI, MACD)
- One volume-based (OBV, VWAP)
- One volatility-based (BB, ATR)

### 8.2 Sharpe-Weighted Ensemble

**Concept**: Weight each signal by its recent risk-adjusted performance.

**Method**:
1. Track rolling Sharpe ratio of each sub-strategy (e.g., 60-day window)
2. Discard strategies with Sharpe < 0
3. Apply softmax to remaining Sharpe ratios to get weights
4. Combined signal = sum(weight_i * signal_i)

**Formula**:
```
sharpe_i = rolling_sharpe(strategy_i, window=60)
weights = softmax([max(0, s) for s in sharpes])
combined = sum(w * signal for w, signal in zip(weights, signals))
```

### 8.3 Adaptive Weighted Majority

**Concept**: Exponential moving average of performance weights.

**Update Rule**: w_j(t) = alpha * score_j(t) + (1 - alpha) * w_j(t-1)
- alpha = 2 / (window + 1)
- Higher alpha favors recent performance
- Window sizes: 5-20 periods

**Key Results (from Numin research)**:
- Ensemble consistently outperforms individual models in utility
- The system "judiciously favors the right model at the right time"
- Achieves positive profitability when individual models show negative returns

### 8.4 Regime-Based Strategy Selection

**Concept**: Instead of combining signals, select the BEST strategy for current regime.

**Implementation**:
1. Detect regime (trending, ranging, volatile, squeeze)
2. Backtest each strategy's performance per regime
3. Deploy the best strategy for detected regime

**Regime Indicators**:
- ADX for trend strength
- Bollinger BandWidth for volatility regime
- Hurst exponent for mean reversion vs trending tendency

---

## 9. RISK MANAGEMENT

### 9.1 Kelly Criterion

**Formula**: f* = (b*p - q) / b
- f* = optimal fraction of capital to risk
- b = odds (avg win / avg loss)
- p = probability of winning
- q = probability of losing (1 - p)

**Alternative (from historical data)**: Kelly% = W - [(1-W) / R]
- W = win rate
- R = win/loss ratio (avg win / avg loss)

**Fractional Kelly**:
| Fraction | Growth Rate | Drawdown | Recommendation |
|----------|-------------|----------|----------------|
| Full Kelly | 100% optimal | Maximum | Never for crypto |
| Half Kelly | ~75% optimal | ~50% of full | Professional standard |
| Quarter Kelly | ~50% optimal | ~25% of full | Conservative |
| Tenth Kelly | ~20% optimal | Minimal | Starting point for crypto |

**Crypto Adjustments**:
- Start with 1/10 Kelly, scale up as confidence grows
- Factor in funding rate carry (improves edge if receiving, reduces if paying)
- Does NOT account for black swan events (common in crypto)

### 9.2 Fixed Fractional Position Sizing

**Concept**: Risk a fixed percentage of account on each trade.

**Formula**: Position Size = (Account * Risk%) / (Entry - Stop)

**Parameters**:
- Conservative: 0.5% risk per trade
- Moderate: 1.0% risk per trade
- Aggressive: 2.0% risk per trade
- Maximum recommended: 2% (even for high-confidence setups)

**Anti-Martingale Variant**: Increase size after wins, decrease after losses (naturally achieved by fixed fractional since account grows/shrinks).

### 9.3 Volatility-Adjusted Position Sizing

**Formula**: Position Size = (Account * Risk%) / (N * ATR(14))
- N = ATR multiplier (typically 2-3)

**Adaptive Version**:
```
vol_regime = current_atr / rolling_mean_atr(100)
if vol_regime > 1.5: risk_pct = base_risk * 0.5    # High vol: halve risk
elif vol_regime > 1.2: risk_pct = base_risk * 0.75  # Elevated vol
elif vol_regime < 0.7: risk_pct = base_risk * 1.25  # Low vol: increase
else: risk_pct = base_risk                           # Normal
```

### 9.4 Trailing Stop Methods

**1. ATR Trailing Stop (Chandelier Exit)**:
- Long stop: Highest high since entry - N * ATR(22)
- Short stop: Lowest low since entry + N * ATR(22)
- N typically 3.0

**2. Percentage Trailing Stop**:
- Stop trails at X% below highest price since entry
- Crypto: 3-5% for scalping, 8-15% for swing

**3. Parabolic SAR Trailing Stop**:
- Acceleration factor starts at 0.02, max 0.20
- Step: 0.02
- Natural trailing mechanism that tightens over time

**4. Moving Average Trailing Stop**:
- Exit when price closes below EMA(21) for longs
- More forgiving than ATR-based stops in trending markets

### 9.5 Dynamic Stop-Loss

**Concept**: Adjust stop-loss based on market conditions rather than fixed parameters.

**Implementation**:
- Initial stop: 2 * ATR(14) from entry
- After 1R profit: Move stop to breakeven
- After 2R profit: Trail at 1.5 * ATR
- After 3R profit: Trail at 1.0 * ATR (tighten)

**Regime-Adjusted Stops**:
- Low volatility regime: Tighter stops (1.5 * ATR)
- High volatility regime: Wider stops (3.0 * ATR)
- Use Bollinger BandWidth percentile to determine regime

---

## CROSS-STRATEGY COMBINATION MATRIX

Which strategies combine well together:

| Strategy A | Strategy B | Rationale |
|-----------|-----------|-----------|
| Supertrend (trend) | RSI (momentum) | Trend direction + overbought/oversold timing |
| Donchian breakout | Volume surge | Confirms genuine breakout vs false |
| Bollinger squeeze | MACD histogram | Squeeze detection + momentum direction |
| KAMA (trend) | Z-score (reversion) | Adaptive trend + statistical extreme entries |
| Multi-TF MACD | ATR position sizing | Direction from multiple TFs + volatility-adjusted risk |
| OBV divergence | Supertrend | Hidden accumulation + trend confirmation |
| Funding rate | OI analysis | Crowding detection + directional bias |
| Regime detection | Strategy ensemble | Select best strategy per regime |
| ROC + ADX | ATR trailing stop | Momentum strength + dynamic exit |

---

## IMPLEMENTATION PRIORITY FOR THIS PROJECT

Based on available data (OHLCV from Binance), here are strategies ranked by implementation feasibility:

**Tier 1 - Directly Implementable** (OHLCV data sufficient):
1. Supertrend with ADX filter
2. Bollinger-Keltner Squeeze breakout
3. Multi-timeframe MACD (already have 1m to 1d data)
4. DEMA/HMA crossover with volume
5. Donchian channel breakout
6. ATR volatility regime detection + strategy selection
7. ROC + RSI momentum combo

**Tier 2 - Implementable with Derived Data**:
1. VWAP (can compute from OHLCV)
2. Volume profile (from volume + price bins)
3. OBV divergence
4. Z-score mean reversion

**Tier 3 - Requires Additional Data**:
1. Funding rate strategies (need funding rate data from Binance API)
2. Open interest signals (need OI data)
3. Liquidation cascade detection (need liquidation data)
4. Basis trading (need spot + futures price comparison)

---

---

## 10. BEAR/RANGE MARKET STRATEGIES (Added 2026-03-22)

These strategies specifically address the challenge of profitability in bear/range-bound crypto markets (like 2025).

### 10.1 Connors RSI (CRSI) Short-Term Mean Reversion

**Concept**: Composite of 3 RSI components — catches short-term reversals with 75% win rate.

**Formula**: CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- RSI(3): ultra-short-term RSI
- RSI_Streak: RSI of consecutive up/down bar streak count
- PercentRank(100): where current return ranks in last 100 bars

**Entry/Exit**:
- Long: CRSI < 10 AND price > SMA(200) (oversold dip in uptrend)
- Short: CRSI > 90 AND price < SMA(200) (overbought rally in downtrend)
- Exit: price crosses 5-period SMA

**Bear market**: EXCELLENT — designed for short-term mean reversion in choppy markets.

### 10.2 Ehlers Fisher Transform

**Concept**: Transforms price into Gaussian distribution, making turning points identifiable.

**Parameters**: Period=9, Overbought > +1.5, Oversold < -1.5
- Short: Fisher crosses below +1.5 (overbought reversal)
- Long: Fisher crosses above -1.5 (oversold reversal)
- Exit: Fisher crosses zero
- Reported: 233% gross return on 8h BTC, PF 1.627

**Bear market**: GOOD — catches tops of bear rallies and bottoms of oversold bounces.

### 10.3 Choppiness Index + Aroon Regime Filter

**Concept**: Detects chop vs trend, different from BBW. Use as meta-filter.

**Parameters**: CHOP period=14, Aroon period=14
- CHOP > 61.8: ranging → use mean reversion
- CHOP < 38.2 + Aroon confirms: trending → use trend following
- CHOP 38.2-61.8: reduce size or flat (transition)

**Bear market**: EXCELLENT — primary purpose is detecting when NOT to trade.

### 10.4 Larry Williams Volatility Breakout

**Concept**: Different from Donchian. Uses previous day's range for breakout levels.

- Range = Previous day High - Low
- Long: Today open + K × Range (K=0.5-0.6)
- Short: Today open - K × Range
- Stop: 2% from entry
- Size: inverse ATR

**Bear market**: MODERATE — captures intraday momentum in both directions.

### 10.5 Pairs Trading (BTC-ETH Spread)

**Concept**: Market-neutral. Profit from relative mispricing regardless of direction.

- Test cointegration with ADF test (p < 0.05)
- Spread = BTC - beta × ETH (beta from rolling OLS, 120 bars)
- Entry: Z-score of spread > +2.0 (short spread) or < -2.0 (long spread)
- Exit: Z-score reverts to ±0.5
- Stop: Z-score hits ±3.5

**Bear market**: EXCELLENT — market-neutral by design. Reported 43.4% in 6 months.

### 10.6 Funding Rate Contrarian

**Concept**: Trade against extreme funding rate (crowded trade reversal).

- Funding > +0.03% (3x normal): overcrowded longs → go short
- Funding < -0.03%: overcrowded shorts → go long
- Confirm with price vs 20-period VWAP
- Exit: funding normalizes to ±0.01%

**Bear market**: GOOD — deep negative funding in bear = squeeze opportunities.
**Note**: Requires funding rate data (already available in data/processed/funding/).

### 10.7 Adaptive Kelly Position Sizing

**Layer on ANY existing strategy**:
- Kelly: f* = (p × b - q) / b, use Quarter-Kelly (f*/4) for crypto
- Vol scaling: size = target_vol(15%) / realized_vol(20)
- High vol (ATR >75th pctile): 1/8 Kelly
- Low vol (ATR <25th pctile): 1/2 Kelly
- Recalculate every 50 trades

**Bear market**: CRITICAL — cuts position size during drawdowns automatically.

---

### 10.8 BTC/ETH-Specific: Why Trend Following Fails + What Works

**Why BTC/ETH always negative Sharpe with trend following:**
- 2022 crash had 20-40% counter-trend rallies WITHIN the downtrend → whipsaw
- EMA/Supertrend lags the initial dump, enters short before relief rally
- Negative funding during bear = shorts PAY longs (cost of being short)
- BTC/ETH are most efficiently priced crypto → less edge than SOL

**Strategies with documented positive Sharpe on BTC during 2022:**

**A. Funding Rate Mean-Reversion** (Sharpe 0.8-1.5 reported)
- Funding Z-score (vs 30-day rolling) < -2.0 → Long (shorts overcrowded)
- Funding Z-score > +2.0 → Short (longs overcrowded)
- Exit: Z-score returns to [-0.5, +0.5] or 24-72h max hold
- Size: scale with |Z-score|
- KEY: uses funding_rate data from data/processed/funding/

**B. Volatility Spike Mean-Reversion** (Sharpe 0.5-1.0)
- ATR(7) / ATR(30) > 2.0 AND price below BB(20, 2.5) → Long
- Exit: ATR ratio < 1.2 OR price > BB middle
- Logic: vol spikes on dumps then mean-reverts, capturing the "vol crush"

**C. Asymmetric Regime with ADX Hysteresis**
- ADX > 25 + price < SMA(50) = bear → ONLY short on EMA(21) retrace rejection
- ADX < 20 = range → mean revert (BB bounds)
- ADX > 25 + price > SMA(50) = bull → ONLY long
- Hysteresis: enter regime at ADX 25, exit at ADX 18

**D. Cross-Asset Lead-Lag** (BTC leads, trade ETH)
- BTC breaks below Donchian(20) low on 4H → Short ETH
- BTC breaks above Donchian(10) high → Close ETH short
- Edge: ETH lags BTC by 1-4 hours on larger timeframes

**E. Volatility Breakout + Bear Regime Filter**
- Setup: BB Width reaches 30-day low (squeeze)
- Entry: price breaks below Donchian(20) AFTER squeeze AND price < SMA(200)
- Exit: price > Donchian(10) high
- Only takes shorts in bear regime = eliminates false upside breakouts

## COMBINATION PRIORITY FOR BEAR MARKETS

| Priority | Strategy | Rating |
|----------|----------|--------|
| 1 | CRSI mean reversion + CHOP regime filter | EXCELLENT |
| 2 | BTC-ETH pairs trading (market neutral) | EXCELLENT |
| 3 | Adaptive Kelly sizing on existing strategies | CRITICAL add-on |
| 4 | Fisher Transform + trend filter | GOOD |
| 5 | Larry Williams volatility breakout | MODERATE |
| 6 | Funding rate contrarian | GOOD |

---

## Sources

- [Hull Kaufman SuperTrend Cloud](https://www.tradingview.com/script/XHnsbKXg-Hull-Kaufman-SuperTrend-Cloud-HKST-Cloud/)
- [Supertrend Indicator Best Settings - Mudrex](https://mudrex.com/learn/supertrend-indicator/)
- [Supertrend Quantitative Strategy for Bitcoin](https://medium.com/@redsword_23261/supertrend-quantitative-trading-strategy-for-bitcoin-11a15cfeb138)
- [Supertrend Indicator Strategy Backtested](https://www.quantifiedstrategies.com/supertrend-indicator-trading-strategy/)
- [Supertrend Best Settings Guide](https://goodcrypto.app/supertrend-indicator-how-to-set-up-use-and-create-profitable-crypto-trading-strategy/)
- [Supertrend Day Trading Crypto](https://beincrypto.com/learn/supertrend-indicator-crypto-explained/)
- [Bollinger-Keltner Squeeze Breakout with ATR Trailing Stops](https://www.pyquantlab.com/article.php?file=Bollinger-Keltner+Squeeze+Breakout+Trading+Strategy+with+ATR+Trailing+Stops.html)
- [Bollinger Bands Strategy: Squeeze then Surge - LuxAlgo](https://www.luxalgo.com/blog/bollinger-bands-strategy-squeeze-then-surge/)
- [Bollinger Bands Trading Strategy Backtested](https://www.quantifiedstrategies.com/bollinger-bands-trading-strategy/)
- [Keltner Channel Strategy 77% WinRate Backtested](https://www.quantifiedstrategies.com/keltner-bands-trading-strategies/)
- [Multi-Timeframe Trend Strategy on Bitcoin - QuantPedia](https://quantpedia.com/how-to-design-a-simple-multi-timeframe-trend-strategy-on-bitcoin/)
- [Multi-Timeframe Trading Strategy 2026 Guide](https://www.mindmathmoney.com/articles/multi-timeframe-analysis-trading-strategy-the-complete-guide-to-trading-multiple-timeframes)
- [Refining Entry Using Lower Timeframes](https://tradingstrategyguides.com/lecture-13-refining-entry-using-lower-timeframes-15m-to-1m-entry-techniques/)
- [Multi-Timeframe MA and RSI Strategy](https://medium.com/@redsword_23261/multi-timeframe-moving-average-and-rsi-trend-trading-strategy-0ea227651279)
- [MACD Indicator Complete Guide - altFINS](https://altfins.com/knowledge-base/macd-line-and-macd-signal-line/)
- [MACD Trading Strategy for Crypto - Bitunix](https://blog.bitunix.com/en/macd-indicator-crypto-trading-strategies/)
- [Stochastic Momentum Index - QuantifiedStrategies](https://www.quantifiedstrategies.com/stochastic-momentum-index/)
- [SMI Comprehensive Guide - TrendSpider](https://trendspider.com/learning-center/the-stochastic-momentum-index-smi-a-refined-indicator-for-traders/)
- [Williams %R - Fidelity](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/williams-r)
- [Williams %R Crypto - UEEx](https://blog.ueex.com/williams-r-indicator/)
- [ROC Indicator Strategy Backtested](https://www.quantifiedstrategies.com/rate-of-change-trading-strategy/)
- [Top Crypto Momentum Indicators ROC OBV ADX](https://bitsgap.com/blog/top-momentum-indicators-for-cryptocurrency-trading-roc-obv-adx)
- [VWAP Indicator Guide](https://www.tradervue.com/blog/vwap-indicator)
- [VWAP in Crypto Trading - Hyrotrader](https://www.hyrotrader.com/blog/vwap-trading-strategy/)
- [Volume Analysis in Bitcoin - Technollogy](https://www.technollogy.com/2026/02/understanding-volume-analysis-in_01662373514.html)
- [OBV Indicator Complete Guide - Mind Math Money](https://www.mindmathmoney.com/articles/on-balance-volume-trading-strategy-amp-settings-obv-indicator-in-tradingview)
- [OBV Divergence in Crypto - altFINS](https://altfins.com/knowledge-base/obv/)
- [OBV 5 Powerful Ways Crypto - Pi42](https://pi42.com/blog/on-balance-volume-obv-crypto-trading/)
- [ATR-Based Stop-Loss for Breakouts - LuxAlgo](https://www.luxalgo.com/blog/atr-based-stop-loss-for-high-volatility-breakouts/)
- [Volatility Indicators Crypto 2025 - Zignaly](https://zignaly.com/crypto-trading/indicators/volatility-indicators)
- [Funding Rate Arbitrage Guide - Amberdata](https://blog.amberdata.io/the-ultimate-guide-to-funding-rate-arbitrage-amberdata)
- [Funding Rate Explained - BingX](https://bingx.com/en/learn/article/what-is-funding-rate-and-how-use-it-in-crypto-trading)
- [Funding Rate Arbitrage Returns - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
- [Derivatives Market Signals - Gate](https://web3.gate.com/en/crypto-wiki/article/what-are-crypto-derivatives-market-signals-and-how-do-futures-open-interest-funding-rates-and-liquidation-data-impact-trading-20251221)
- [Numin: Weighted-Majority Ensembles for Trading](https://arxiv.org/html/2412.03167v1)
- [Ensemble Methods for Crypto Trading - FinRL](https://arxiv.org/html/2501.10709v1)
- [Ensemble Strategies - Build Alpha](https://www.buildalpha.com/trading-ensemble-strategies/)
- [Combining Investment Signals - GSAM](https://www.gsam.com/content/dam/gsam/pdfs/institutions/en/articles/2018/Combining_Investment_Signals_in_LongShort_Strategies.pdf)
- [Kelly Criterion for Crypto Traders](https://medium.com/@tmapendembe_28659/kelly-criterion-for-crypto-traders-a-modern-approach-to-volatile-markets-a0cda654caa9)
- [Kelly Criterion in Crypto - OSL](https://www.osl.com/hk-en/academy/article/what-is-the-kelly-bet-size-criterion-and-how-to-use-it-in-crypto-trading)
- [Kelly Criterion Trading Guide - LiteFinance](https://www.litefinance.org/blog/for-beginners/best-technical-indicators/kelly-criterion-trading/)
- [Kelly Criterion - Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Donchian Channel Breakout - Algomatic Trading](https://algomatictrading.substack.com/p/strategy-8-the-easiest-trend-system)
- [Donchian Breakout Strategy - PyQuantLab](https://pyquantlab.medium.com/a-donchian-channel-breakout-strategy-a-simple-trend-following-approach-18b7b74c4358)
- [KAMA - StockCharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama)
- [Adaptive Moving Average Strategy - FMZ](https://medium.com/@FMZQuant/adaptive-moving-average-trading-strategy-422798f0d419)
- [20 Moving Average Strategies Backtested](https://www.quantifiedstrategies.com/moving-average-trading-strategy/)
- [Advanced Moving Averages Guide - GoodCrypto](https://goodcrypto.app/advanced-moving-averages-smma-ama-lwma-dema-tema-a-complete-guide-for-crypto-traders/)
- [Lessons from 7 Years of Algo Trading](https://medium.com/@josh.malizzi/lessons-from-7-years-of-algorithmic-trading-research-and-development-c63f1d319831)
