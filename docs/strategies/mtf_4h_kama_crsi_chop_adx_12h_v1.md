# Strategy: mtf_4h_kama_crsi_chop_adx_12h_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.597 | -8.0% | -23.6% | 979 | FAIL |
| ETHUSDT | -0.708 | -22.0% | -29.6% | 1025 | FAIL |
| SOLUSDT | 0.057 | +15.1% | -41.3% | 1025 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.074 | +6.2% | -14.0% | 333 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #254: 4h Primary + 12h/1d HTF — KAMA Trend + Connors RSI + Choppiness Regime

Hypothesis: After 250+ experiments, the winning pattern is KAMA trend (12h) + regime filter.
#246 proved KAMA+Choppiness on 12h achieves Sharpe=0.350. For 4h, we adapt:

1. KAMA(10,2,30) on 12h for PRIMARY trend (proven adaptive trend follower)
2. Connors RSI on 4h for entries (75% win rate, better than regular RSI)
3. Choppiness Index(14) on 4h for regime (trend vs mean-revert mode)
4. ADX(14) on 4h for trend strength confirmation (>25 = trending)
5. Asymmetric sizing: 0.25 base, 0.35 strong conviction
6. Force entry every 15 bars to guarantee 10+ trades/year

Key improvements over #251 (Sharpe=0.155):
- KAMA instead of HMA (adaptive to volatility, proven on #246)
- Connors RSI instead of RSI(14) (faster mean reversion signal)
- ADX filter to avoid whipsaw in weak trends
- 12h KAMA slope for regime (not 1d, more responsive on 4h)

Position sizing: 0.25 base, 0.35 strong (discrete levels)
Target: 25-50 trades/year per symbol
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_crsi_chop_adx_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - fast in trends, slow in chop.
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        # Efficiency Ratio
        if i >= er_period:
            signal = abs(close[i] - close[i - er_period])
            noise = 0.0
            for j in range(i - er_period + 1, i + 1):
                noise += abs(close[j] - close[j - 1])
            er = signal / noise if noise > 0 else 0.0
        else:
            er = 0.0
        
        # Smoothing Constant
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_kama_slope(kama_values, lookback=3):
    """Calculate KAMA slope as percentage change over lookback."""
    slope = np.zeros(len(kama_values))
    for i in range(lookback, len(kama_values)):
        prev = kama_values[i - lookback]
        curr = kama_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): 3-period RSI on close
    RSI_Streak(2): 2-period RSI on streak length (consecutive up/down)
    PercentRank(100): Percentile rank of 1-day price change over 100 periods
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on absolute streak values
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, streak_period)
    
    # Percent Rank
    pct_rank = np.zeros(n)
    returns = close_s.pct_change().values
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current_ret = returns[i]
            if not np.isnan(current_ret):
                pct_rank[i] = np.sum(valid < current_ret) / len(valid) * 100
    
    # Combine
    crsi = (rsi_3 + streak_rsi + pct_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (primary trend regime)
    kama_12h_21 = calculate_kama(df_12h['close'].values, 10, 2, 30)
    kama_12h_slope = calculate_kama_slope(kama_12h_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_12h_21_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_21)
    kama_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # 4h KAMA for local trend
    kama_4h_21 = calculate_kama(close, 10, 2, 30)
    kama_4h_50 = calculate_kama(close, 10, 2, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_12h_21_aligned[i]) or np.isnan(kama_12h_slope_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(kama_4h_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 12H TREND REGIME (primary direction filter) ===
        # Bull: 12h KAMA slope > 0.20%
        # Bear: 12h KAMA slope < -0.20%
        regime_bull = kama_12h_slope_aligned[i] > 0.20
        regime_bear = kama_12h_slope_aligned[i] < -0.20
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_12h_kama = close[i] > kama_12h_21_aligned[i]
        price_below_12h_kama = close[i] < kama_12h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === TREND STRENGTH ===
        is_strong_trend = adx_14[i] > 25.0
        is_weak_trend = adx_14[i] < 20.0
        
        # === 4H LOCAL SIGNALS ===
        price_above_4h_kama = close[i] > kama_4h_21[i]
        price_below_4h_kama = close[i] < kama_4h_21[i]
        kama_4h_bullish = kama_4h_21[i] > kama_4h_50[i]
        kama_4h_bearish = kama_4h_21[i] < kama_4h_50[i]
        
        # === CONNORS RSI THRESHOLDS (wider for more trades) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        crsi_mid_bull = crsi[i] > 40.0
        crsi_mid_bear = crsi[i] < 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + strong ADX + regime aligned)
        if is_trending and is_strong_trend:
            # LONG: Trending + bull regime + price above 4h KAMA + CRSI confirming
            if regime_bull and price_above_4h_kama and crsi_mid_bull:
                new_signal = STRONG_SIZE
            # LONG: Trending + price above 12h KAMA + 4h KAMA bullish
            elif price_above_12h_kama and kama_4h_bullish and crsi[i] > 35:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + bear regime + price below 4h KAMA + CRSI confirming
            if regime_bear and price_below_4h_kama and crsi_mid_bear:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Trending + price below 12h KAMA + 4h KAMA bearish
            elif price_below_12h_kama and kama_4h_bearish and crsi[i] < 65:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy + weak ADX)
        if is_choppy or is_weak_trend:
            # LONG: Choppy + CRSI oversold (<15) + not in strong bear
            if crsi_oversold and not regime_bear:
                new_signal = BASE_SIZE
            # LONG: Choppy + CRSI extreme oversold (<10) in any regime
            if crsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.9
            
            # SHORT: Choppy + CRSI overbought (>85) + not in strong bull
            if crsi_overbought and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + CRSI extreme overbought (>90) in any regime
            if crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.9
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 15 bars (~2.5 days on 4h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] > 35 and price_above_4h_kama:
                new_signal = BASE_SIZE * 0.7
            elif regime_bear and crsi[i] < 65 and price_below_4h_kama:
                new_signal = -BASE_SIZE * 0.7
            elif is_choppy and crsi[i] < 25:
                new_signal = BASE_SIZE * 0.6
            elif is_choppy and crsi[i] > 75:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_12h_kama:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_12h_kama:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-23 01:31
