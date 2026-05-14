# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.165 | +27.6% | -7.7% | 209 | PASS |
| ETHUSDT | 0.236 | +32.5% | -13.6% | 197 | PASS |
| SOLUSDT | 0.751 | +97.1% | -21.0% | 168 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.377 | -6.2% | -9.7% | 81 | FAIL |
| ETHUSDT | 0.712 | +16.7% | -11.5% | 64 | PASS |
| SOLUSDT | -0.376 | -0.3% | -11.6% | 57 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 12h data once for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Load 1d data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Daily OHLC for Camarilla R3 and S3 (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for previous day
    p = (high_1d + low_1d + close_1d_vals) / 3
    r3 = close_1d_vals + (high_1d - low_1d) * 1.1 / 4
    s3 = close_1d_vals - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + above 12h EMA50 + volume spike
            if (close[i] > r3_aligned[i] and close[i] > ema_50_12h_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below 12h EMA50 + volume spike
            elif (close[i] < s3_aligned[i] and close[i] < ema_50_12h_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 02:16
