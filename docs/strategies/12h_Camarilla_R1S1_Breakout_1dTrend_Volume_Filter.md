# Strategy: 12h_Camarilla_R1S1_Breakout_1dTrend_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.131 | +25.6% | -8.1% | 76 | PASS |
| ETHUSDT | 0.017 | +20.7% | -7.8% | 69 | PASS |
| SOLUSDT | 0.086 | +21.9% | -28.7% | 68 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.581 | -7.9% | -12.3% | 35 | FAIL |
| ETHUSDT | 0.429 | +11.2% | -4.3% | 28 | PASS |
| SOLUSDT | -0.907 | -6.3% | -16.3% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume_Filter"
timeframe = "12h"
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
    
    # Load 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla R1 and S1 for previous day
    p = (high_1d + low_1d + close_1d) / 3
    r1 = p + (high_1d - low_1d) * 1.1 / 12
    s1 = p - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h (wait for daily close)
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
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above 1d EMA34 + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 + below 1d EMA34 + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below S1
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above R1
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-12 02:33
