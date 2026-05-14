# Strategy: 6h_Pivot_S3R3_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.455 | +40.4% | -11.4% | 222 | PASS |
| ETHUSDT | 0.227 | +31.4% | -13.6% | 183 | PASS |
| SOLUSDT | 0.897 | +115.0% | -18.1% | 158 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.639 | +0.2% | -6.9% | 85 | FAIL |
| ETHUSDT | 0.383 | +11.3% | -6.9% | 66 | PASS |
| SOLUSDT | 0.155 | +7.7% | -6.6% | 62 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily pivot points with R3/S3 levels and volume confirmation.
# Enters long when price breaks above S3 with volume, short when breaks below R3 with volume.
# Uses 1d EMA(34) as trend filter. Designed for 12-37 trades/year by requiring
# significant breakouts (R3/S3) rather than minor S1/R1 levels, reducing trade frequency.
# Works in bull/bear: buys support breaks, sells resistance breaks.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day OHLC)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
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
    
    # Align daily pivots to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily trend: price above/below daily EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 2.0 x 30-period average (6h) for significance
    vol_ma_30 = np.full(n, np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need pivots (1), daily EMA (34), volume MA (30)
    start_idx = max(1, 34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_30[i]
        
        # Volume filter (strict)
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filters
        daily_bullish = price > ema_34_1d_aligned[i]
        daily_bearish = price < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above S3 with volume and daily bullish
            if price > s3_aligned[i] and vol_filter and daily_bullish:
                signals[i] = size
                position = 1
            # Short: price breaks below R3 with volume and daily bearish
            elif price < r3_aligned[i] and vol_filter and daily_bearish:
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

name = "6h_Pivot_S3R3_Volume_Trend"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 10:26
