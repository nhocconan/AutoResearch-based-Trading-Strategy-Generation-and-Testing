# Strategy: 6h_ElderRay_WeeklyPivot_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.015 | +20.0% | -8.6% | 110 | FAIL |
| ETHUSDT | 0.188 | +28.8% | -12.2% | 97 | PASS |
| SOLUSDT | 0.818 | +97.9% | -15.2% | 89 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.295 | +9.3% | -6.5% | 44 | PASS |
| SOLUSDT | -1.012 | -4.7% | -13.1% | 33 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Elder Ray + Weekly Pivot + Volume Confirmation
Hypothesis: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13. 
Weekly pivot provides key support/resistance levels from higher timeframe. 
Volume confirmation filters weak breaks. Works in bull (long when Bull Power > 0, price > weekly pivot R1, volume spike) 
and bear (short when Bear Power < 0, price < weekly pivot S1, volume spike) via symmetric logic.
Target: 12-25 trades/year on 6h to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivot to 6h (no extra delay as pivot is based on completed weekly bar)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Get 1d data for EMA13 (Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d
    close_1d = pd.Series(df_1d['close'])
    ema_13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate ATR(14) for stop management
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA13, ATR, volume MA
    start_idx = max(13, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_13_val = ema_13_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        pivot_val = pivot_1w_aligned[i]
        r1_val = r1_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        
        # Elder Ray components
        bull_power = curr_high - ema_13_val  # Bull Power: High - EMA13
        bear_power = curr_low - ema_13_val   # Bear Power: Low - EMA13
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: Bull Power > 0, price > weekly R1, volume confirmation
            long_entry = (bull_power > 0) and (curr_close > r1_val) and volume_confirm
            # Short: Bear Power < 0, price < weekly S1, volume confirmation
            short_entry = (bear_power < 0) and (curr_close < s1_val) and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below weekly pivot OR 2.0*ATR trailing stop OR Bull Power <= 0
            if curr_close < pivot_val or curr_close < (highest_since_entry - 2.0 * atr_val) or bull_power <= 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above weekly pivot OR 2.0*ATR trailing stop OR Bear Power >= 0
            if curr_close > pivot_val or curr_close > (lowest_since_entry + 2.0 * atr_val) or bear_power >= 0:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 04:29
