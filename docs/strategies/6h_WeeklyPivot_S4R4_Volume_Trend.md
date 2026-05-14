# Strategy: 6h_WeeklyPivot_S4R4_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.006 | +18.1% | -17.1% | 138 | DISCARD |
| ETHUSDT | 0.192 | +31.0% | -16.6% | 125 | KEEP |
| SOLUSDT | 1.036 | +199.3% | -22.6% | 113 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.126 | +7.3% | -10.4% | 49 | KEEP |
| SOLUSDT | -0.368 | +0.7% | -7.8% | 12 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points (R4/S4 levels) with volume confirmation and daily EMA(50) trend filter.
# Weekly pivots provide stronger support/resistance than daily, reducing false breakouts.
# Enters long when price breaks above S4 with volume, short when breaks below R4 with volume.
# Trend filter ensures alignment with daily trend to avoid counter-trend entries.
# Designed for ~15-25 trades/year by requiring significant breakouts (R4/S4) rather than minor levels.
# Works in bull/bear: buys support breaks, sells resistance breaks.
# Uses volume filter (volume > 1.8x 20-period average) to avoid false breakouts.
# Exit when price returns to weekly pivot or trend changes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week OHLC)
    high_prev = np.roll(high_1w, 1)
    low_prev = np.roll(low_1w, 1)
    close_prev = np.roll(close_1w, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    r2 = pivot + (high_prev - low_prev)
    s2 = pivot - (high_prev - low_prev)
    r3 = high_prev + 2 * (pivot - low_prev)
    s3 = low_prev - 2 * (high_prev - pivot)
    r4 = high_prev + 3 * (pivot - low_prev)
    s4 = low_prev - 3 * (high_prev - pivot)
    
    # Align weekly pivots to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily trend: price above/below daily EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.8 x 20-period average (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivots (1), daily EMA (50), volume MA (20)
    start_idx = max(1, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.8 * vol_avg
        
        # Trend filters
        daily_bullish = price > ema_50_1d_aligned[i]
        daily_bearish = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above S4 with volume and daily bullish
            if price > s4_aligned[i] and vol_filter and daily_bullish:
                signals[i] = size
                position = 1
            # Short: price breaks below R4 with volume and daily bearish
            elif price < r4_aligned[i] and vol_filter and daily_bearish:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below pivot or daily trend turns bearish
            if price < pivot_aligned[i] or not daily_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above pivot or daily trend turns bullish
            if price > pivot_aligned[i] or not daily_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_S4R4_Volume_Trend"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 10:43
