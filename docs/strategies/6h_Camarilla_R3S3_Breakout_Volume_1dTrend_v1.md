# Strategy: 6h_Camarilla_R3S3_Breakout_Volume_1dTrend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.392 | +35.0% | -7.5% | 101 | PASS |
| ETHUSDT | 0.206 | +28.5% | -7.3% | 92 | PASS |
| SOLUSDT | 0.744 | +70.2% | -9.8% | 68 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.864 | +0.1% | -4.0% | 29 | FAIL |
| ETHUSDT | 0.659 | +13.2% | -8.3% | 36 | PASS |
| SOLUSDT | -0.363 | +2.6% | -6.3% | 22 | FAIL |

## Code
```python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Levels + Volume Spike + 1d Trend Filter
# - Long when price breaks above Camarilla R3 with volume spike (>1.5x 20-period 1d avg volume) in uptrend (price > 1d EMA50)
# - Short when price breaks below Camarilla S3 with volume spike in downtrend (price < 1d EMA50)
# - Exit when price returns to Camarilla pivot (central level) or trend reverses
# - Designed to capture breakouts with institutional volume in trending markets
# - Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag
# - Works in both bull and bear markets by following 1d trend filter

name = "6h_Camarilla_R3S3_Breakout_Volume_1dTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # H, L, C from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    range_ = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = camarilla_pivot + 1.1 * range_ / 6
    camarilla_s3 = camarilla_pivot - 1.1 * range_ / 6
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 1d volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 6h: 1d has 4x 6h bars, so divide by 4
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 4.0)
        
        if position == 0:
            # Look for long entry: price breaks above R3 + volume spike + uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i-1] <= camarilla_r3_aligned[i-1] and  # Just broke above
                volume_filter and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below S3 + volume spike + downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i-1] >= camarilla_s3_aligned[i-1] and  # Just broke below
                  volume_filter and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to pivot or trend reverses
            if (close[i] <= camarilla_pivot_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to pivot or trend reverses
            if (close[i] >= camarilla_pivot_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 15:38
