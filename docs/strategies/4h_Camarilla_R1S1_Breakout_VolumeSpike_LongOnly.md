# Strategy: 4h_Camarilla_R1S1_Breakout_VolumeSpike_LongOnly

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.103 | +24.7% | -16.3% | 98 | PASS |
| ETHUSDT | 0.435 | +43.9% | -10.6% | 89 | PASS |
| SOLUSDT | 1.121 | +167.8% | -16.1% | 78 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.672 | +0.4% | -9.0% | 30 | FAIL |
| ETHUSDT | 0.422 | +11.4% | -15.0% | 30 | PASS |
| SOLUSDT | 0.457 | +12.8% | -12.7% | 28 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_VolumeSpike_LongOnly"
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
    
    # Load 1d data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Daily OHLC for Camarilla R1 and S1 (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 for previous day
    p = (high_1d + low_1d + close_1d_vals) / 3
    r1 = close_1d_vals + (high_1d - low_1d) * 1.1 / 2
    s1 = close_1d_vals - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + volume spike (long only strategy)
            if (close[i] > r1_aligned[i] and vol_spike[i]):
                signals[i] = 0.30
                position = 1
        elif position == 1:
            # Exit long: price closes below S1
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
    
    return signals
```

## Last Updated
2026-05-12 02:11
