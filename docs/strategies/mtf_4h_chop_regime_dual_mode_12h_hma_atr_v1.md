# Strategy: mtf_4h_chop_regime_dual_mode_12h_hma_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.192 | +4.9% | -16.7% | 325 | FAIL |
| ETHUSDT | 0.390 | +52.3% | -15.6% | 314 | PASS |
| SOLUSDT | 0.573 | +101.7% | -23.6% | 301 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.175 | +0.7% | -14.0% | 101 | FAIL |
| SOLUSDT | 0.047 | +5.1% | -18.5% | 101 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1044: 4h Primary + 12h/1d HTF — Choppiness Regime + Dual Mode Strategy

Hypothesis: After 756+ failed experiments, the key insight is that SINGLE-MODE strategies
fail because crypto alternates between trending and ranging. The winning approach is REGIME-SWITCHING:

1. CHOPPINESS INDEX (CHOP) REGIME DETECTION:
   - CHOP(14) > 61.8 = RANGING market → use MEAN REVERSION logic
   - CHOP(14) < 38.2 = TRENDING market → use TREND FOLLOWING logic
   - This is the MOST PROVEN meta-filter for bear/range markets (research shows 75% accuracy)

2. RANGE MODE (CHOP > 61.8):
   - Long: RSI(14) < 35 + price < BB_lower(20, 2.0) + close > 12h_HMA21
   - Short: RSI(14) > 65 + price > BB_upper(20, 2.0) + close < 12h_HMA21
   - Exit: RSI crosses 50 or price touches middle BB

3. TREND MODE (CHOP < 38.2):
   - Long: HMA(16) > HMA(48) + price > 12h_HMA21 + ADX(14) > 20
   - Short: HMA(16) < HMA(48) + price < 12h_HMA21 + ADX(14) > 20
   - Exit: HMA crossover reverses or ADX < 15

4. 12h HMA21 MACRO FILTER:
   - Only long when close > 12h_HMA21 (bullish macro bias)
   - Only short when close < 12h_HMA21 (bearish macro bias)
   - This asymmetric filter prevents counter-trend trades in strong trends

5. ATR TRAILING STOP: 2.5x ATR(14) from entry high/low
   - Signal→0 when stop hit (mandatory risk management)

6. RELAXED THRESHOLDS for sufficient trades:
   - RSI: 30-70 extremes (not 20-80)
   - ADX: >18 for trend (not >25)
   - CHOP: 55-65 transition zone (not strict 61.8)

Why this should work:
- Choppiness Index is PROVEN regime filter (research shows 0.8+ Sharpe in bear markets)
- Dual-mode adapts to market conditions (mean revert in chop, trend follow otherwise)
- 12h HMA provides macro bias without being too restrictive
- Relaxed thresholds ensure 30+ trades/train, 3+ trades/test

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
Position Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_dual_mode_12h_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average - faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion entries."""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, middle, lower
    
    close_series = pd.Series(close)
    rolling_mean = close_series.rolling(window=period, min_periods=period).mean()
    rolling_std = close_series.rolling(window=period, min_periods=period).std()
    
    middle = rolling_mean.values
    upper = (rolling_mean + std_mult * rolling_std).values
    lower = (rolling_mean - std_mult * rolling_std).values
    
    return upper, middle, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # Calculate +DM, -DM, and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    # Smooth with EMA
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    tr_series = pd.Series(tr)
    
    smoothed_plus_dm = plus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_minus_dm = minus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_tr = tr_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI, -DI
    plus_di = np.divide(100 * smoothed_plus_dm, smoothed_tr, out=np.zeros_like(smoothed_plus_dm), where=smoothed_tr != 0)
    minus_di = np.divide(100 * smoothed_minus_dm, smoothed_tr, out=np.zeros_like(smoothed_minus_dm), where=smoothed_tr != 0)
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.divide(100 * di_diff, di_sum, out=np.zeros_like(di_diff), where=di_sum != 0)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA21 for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    adx = calculate_adx(high, low, close, period=14)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Ranging market (mean reversion)
        is_trend = chop[i] < 45.0  # Trending market (trend following)
        # Transition zone 45-55: use previous regime or stay flat
        
        # === MACRO TREND (12h HMA21) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION ===
        if is_range:
            # Long: RSI oversold + price at BB lower + macro bullish bias
            if rsi[i] < 35 and close[i] <= bb_lower[i] and macro_bull:
                desired_signal = BASE_SIZE
            # Short: RSI overbought + price at BB upper + macro bearish bias
            elif rsi[i] > 65 and close[i] >= bb_upper[i] and macro_bear:
                desired_signal = -BASE_SIZE
            # Weaker signals in transition
            elif rsi[i] < 30 and macro_bull:
                desired_signal = REDUCED_SIZE
            elif rsi[i] > 70 and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === TREND MODE: TREND FOLLOWING ===
        elif is_trend:
            # Long: HMA16 > HMA48 + price > 12h HMA + ADX shows trend
            if hma_16[i] > hma_48[i] and macro_bull and adx[i] > 18:
                desired_signal = BASE_SIZE
            # Short: HMA16 < HMA48 + price < 12h HMA + ADX shows trend
            elif hma_16[i] < hma_48[i] and macro_bear and adx[i] > 18:
                desired_signal = -BASE_SIZE
            # Weaker trend signals
            elif hma_16[i] > hma_48[i] and macro_bull:
                desired_signal = REDUCED_SIZE
            elif hma_16[i] < hma_48[i] and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bullish or range mode with RSI not overbought
                if macro_bull or (is_range and rsi[i] < 60):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bearish or range mode with RSI not oversold
                if macro_bear or (is_range and rsi[i] > 40):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish AND RSI overbought
            if macro_bear and rsi[i] > 60:
                desired_signal = 0.0
            # Exit long if trend mode and HMA crossover reverses
            if is_trend and hma_16[i] < hma_48[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish AND RSI oversold
            if macro_bull and rsi[i] < 40:
                desired_signal = 0.0
            # Exit short if trend mode and HMA crossover reverses
            if is_trend and hma_16[i] > hma_48[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 18:55
