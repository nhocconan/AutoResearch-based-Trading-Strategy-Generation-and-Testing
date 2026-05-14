# Strategy: 12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.186 | +29.3% | -12.2% | 89 | PASS |
| ETHUSDT | 0.246 | +34.2% | -12.1% | 84 | PASS |
| SOLUSDT | 0.483 | +68.6% | -27.5% | 77 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.747 | -2.1% | -7.2% | 34 | FAIL |
| ETHUSDT | 0.410 | +12.5% | -6.3% | 29 | PASS |
| SOLUSDT | -0.340 | -1.1% | -15.4% | 29 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS_v2
Hypothesis: On 12h timeframe, price breaking above Camarilla R3 or below S3 levels from the prior 1d period, combined with 1d EMA34 trend filter and volume confirmation, captures high-probability momentum moves in both bull and bear markets. The 1d trend filter ensures alignment with higher timeframe momentum, reducing false breakouts in choppy conditions. Lower trade frequency on 12h reduces fee drag, improving generalization. This version adds minimum holding period to reduce trade frequency and avoid overtrading.
"""
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS_v2"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Camarilla R3/S3 levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1666 * range_1d * 1.1 / 2
    s3_1d = close_1d - 1.1666 * range_1d * 1.1 / 2
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above R3 + 1d uptrend + volume
            if close[i] > r3_1d_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S3 + 1d downtrend + volume
            elif close[i] < s3_1d_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Minimum holding period of 3 bars to reduce trade frequency
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit: price crosses back through the opposite S3/R3 level
            if position == 1:
                if close[i] < s3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 06:25
