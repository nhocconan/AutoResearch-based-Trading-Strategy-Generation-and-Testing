# Strategy: 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeS

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.420 | +48.4% | -14.5% | 138 | PASS |
| ETHUSDT | 0.666 | +80.9% | -20.1% | 124 | PASS |
| SOLUSDT | 0.623 | +107.1% | -33.0% | 111 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.146 | -9.4% | -14.6% | 51 | FAIL |
| ETHUSDT | 1.219 | +35.5% | -7.6% | 41 | PASS |
| SOLUSDT | 0.914 | +27.2% | -11.3% | 36 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeS
Hypothesis: Camarilla R3/S3 breakouts on 6h with daily trend filter and volume confirmation capture strong momentum moves while avoiding chop. Works in both bull and bear markets by targeting high-probability breakouts with strong volume. Target: 12-37 trades/year.
"""
name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
timeframe = "6h"
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
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 1d Camarilla R3/S3 levels (stronger breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1666 * range_1d * 1.1 / 2
    s3_1d = close_1d - 1.1666 * range_1d * 1.1 / 2
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + daily uptrend + volume
            if close[i] > r3_1d_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 + daily downtrend + volume
            elif close[i] < s3_1d_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite S3/R3 level
            if position == 1:
                if close[i] < s3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if close[i] > r3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-07 06:21
