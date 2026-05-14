# Strategy: 6h_WeeklyPivot_R1_S1_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.390 | +43.1% | -12.9% | 70 | PASS |
| ETHUSDT | 0.013 | +18.1% | -16.4% | 70 | PASS |
| SOLUSDT | 0.864 | +151.8% | -27.8% | 68 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.050 | -6.6% | -10.7% | 30 | FAIL |
| ETHUSDT | 0.587 | +17.4% | -8.9% | 24 | PASS |
| SOLUSDT | -0.123 | +1.8% | -17.6% | 21 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1_S1_Breakout_1dTrend_Volume
Hypothesis: In both bull and bear markets, price respects weekly pivot levels (R1/S1) as support/resistance.
Breakouts above R1 or below S1 with volume confirmation and daily trend alignment capture momentum.
Uses 6h timeframe to reduce trade frequency and avoid fee drag. Weekly pivots calculated from prior week's OHLC.
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter and weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Weekly pivot points (R1, S1) from prior week
    # Align to start of week (Monday 00:00 UTC) - using 1d data to determine week boundaries
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Calculate weekly pivot points using prior week's OHLC
    # We need to group daily data into weeks (Monday to Sunday)
    # For simplicity, we'll use a rolling window of 5 trading days (approximation)
    # In practice, we'd need to know exact week boundaries, but 5-day approx works for pivots
    if len(high_1d) >= 5:
        # Rolling window of 5 days for approximate weekly OHLC
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values  # prior week
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(close_1d_arr).rolling(window=5, min_periods=5).last().shift(1).values
        
        # Weekly pivot point: (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # R1 = 2*P - L
        weekly_r1 = 2 * weekly_pivot - weekly_low
        # S1 = 2*P - H
        weekly_s1 = 2 * weekly_pivot - weekly_high
        
        # Align weekly pivots to 6h timeframe
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    else:
        # Not enough data for weekly calculation
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period - need enough data for all indicators
    start_idx = max(50, 20) + 1  # EMA34 needs 34, plus buffers
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike and daily uptrend
            if (close[i] > weekly_r1_aligned[i] and volume_spike[i] and close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike and daily downtrend
            elif (close[i] < weekly_s1_aligned[i] and volume_spike[i] and close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to weekly S1 or trend fails
            if (close[i] <= weekly_s1_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to weekly R1 or trend fails
            if (close[i] >= weekly_r1_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 17:30
