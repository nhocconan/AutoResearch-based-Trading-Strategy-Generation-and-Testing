# Strategy: 12h_1d_Camarilla_R3S3_VolumeSpike_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.178 | +14.5% | -5.6% | 68 | DISCARD |
| ETHUSDT | 0.193 | +28.9% | -7.1% | 57 | KEEP |
| SOLUSDT | 0.015 | +17.2% | -26.7% | 63 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.289 | +9.6% | -6.2% | 22 | KEEP |
| SOLUSDT | -1.460 | -11.8% | -17.3% | 23 | DISCARD |

## Code
```python
#!/usr/bin/env python3
# 12h_1d_Camarilla_R3S3_VolumeSpike_TrendFilter
# Hypothesis: On 12h timeframe, trade breakouts from daily Camarilla R3/S3 levels with volume spike confirmation and daily EMA trend filter.
# Uses daily EMA34 to filter trades in trending markets. Targets 15-25 trades per year. Works in bull/bear via trend-aligned breakouts.

name = "12h_1d_Camarilla_R3S3_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, S3
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily levels to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R3, volume spike, and price above daily EMA34 (uptrend)
            if (close[i] > r3_aligned[i] * 1.005 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S3, volume spike, and price below daily EMA34 (downtrend)
            elif (close[i] < s3_aligned[i] * 0.995 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S3 or trend reversal (below EMA34)
            if close[i] < s3_aligned[i] * 0.995 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R3 or trend reversal (above EMA34)
            if close[i] > r3_aligned[i] * 1.005 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 03:39
