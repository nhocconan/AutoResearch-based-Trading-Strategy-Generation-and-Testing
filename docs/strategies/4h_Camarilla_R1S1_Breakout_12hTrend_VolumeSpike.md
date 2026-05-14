# Strategy: 4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.428 | +36.8% | -8.8% | 294 | PASS |
| ETHUSDT | 0.608 | +46.8% | -5.5% | 264 | PASS |
| SOLUSDT | 0.445 | +48.7% | -12.4% | 224 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.199 | -2.5% | -4.6% | 117 | FAIL |
| ETHUSDT | 1.464 | +25.8% | -5.7% | 103 | PASS |
| SOLUSDT | 0.345 | +9.7% | -6.2% | 76 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Get 12h data for trend and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + Range * 1.1 / 12
    # S1 = C - Range * 1.1 / 12
    # R3 = C + Range * 1.1 / 4
    # S3 = C - Range * 1.1 / 4
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r1_12h = close_12h + range_12h * 1.1 / 12
    s1_12h = close_12h - range_12h * 1.1 / 12
    r3_12h = close_12h + range_12h * 1.1 / 4
    s3_12h = close_12h - range_12h * 1.1 / 4
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Volume spike: current volume > 2x 12-period average (6 days on 4h)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > (vol_ma * 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above 12h EMA50 (uptrend) AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND below 12h EMA50 (downtrend) AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 OR below 12h EMA50 (trend change)
            if close[i] < s1_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R1 OR above 12h EMA50 (trend change)
            if close[i] > r1_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
```

## Last Updated
2026-05-11 15:03
