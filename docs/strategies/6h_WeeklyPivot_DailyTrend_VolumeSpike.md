# Strategy: 6h_WeeklyPivot_DailyTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.090 | +13.2% | -18.8% | 76 | FAIL |
| ETHUSDT | 0.280 | +38.2% | -14.5% | 64 | PASS |
| SOLUSDT | 1.385 | +327.3% | -21.7% | 67 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.055 | +6.3% | -6.6% | 14 | PASS |
| SOLUSDT | 0.211 | +8.5% | -8.3% | 15 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Using previous week's data (already complete when we get it)
    prev_week_high = df_1w['high'].shift(1).values  # Previous week high
    prev_week_low = df_1w['low'].shift(1).values    # Previous week low
    prev_week_close = df_1w['close'].shift(1).values # Previous week close
    
    # Pivot point calculation
    pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot_point - prev_week_low
    s1 = 2 * pivot_point - prev_week_high
    r2 = pivot_point + (prev_week_high - prev_week_low)
    s2 = pivot_point - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pivot_point - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pivot_point)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_point_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 34-period EMA on daily close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 34 for daily EMA and 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_point_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot = pivot_point_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        ema_1d = ema_34_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Price breaks above R1 with volume AND price > daily EMA34 (uptrend)
            if close[i] > r1_level and vol > 1.5 * vol_ma_val and close[i] > ema_1d:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S1 with volume AND price < daily EMA34 (downtrend)
            elif close[i] < s1_level and vol > 1.5 * vol_ma_val and close[i] < ema_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below pivot OR trend reverses (price < daily EMA34)
            if close[i] < pivot or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above pivot OR trend reverses (price > daily EMA34)
            if close[i] > pivot or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 03:17
