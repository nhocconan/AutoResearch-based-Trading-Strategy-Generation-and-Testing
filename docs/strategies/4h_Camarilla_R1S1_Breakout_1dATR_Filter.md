# Strategy: 4h_Camarilla_R1S1_Breakout_1dATR_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.108 | +10.6% | -14.1% | 178 | FAIL |
| ETHUSDT | 0.304 | +41.6% | -15.7% | 160 | PASS |
| SOLUSDT | 0.693 | +118.2% | -27.5% | 135 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.003 | +28.6% | -10.5% | 53 | PASS |
| SOLUSDT | 0.005 | +4.5% | -14.8% | 48 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dATR_Filter"
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
    
    # Load 1d data once for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Daily OHLC for Camarilla R1 and S1 (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 for previous day
    p = (high_1d + low_1d + close_1d_vals) / 3
    r1 = close_1d_vals + (high_1d - low_1d) * 1.1 / 2
    s1 = close_1d_vals - (high_1d - low_1d) * 1.1 / 2
    
    # Daily ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d_vals[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d_vals[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Camarilla levels and ATR to 4h (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + volatility filter + volume spike
            if (close[i] > r1_aligned[i] and 
                atr_1d_aligned[i] > 0 and  # volatility present
                vol_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 + volatility filter + volume spike
            elif (close[i] < s1_aligned[i] and 
                  atr_1d_aligned[i] > 0 and 
                  vol_spike[i]):
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
2026-05-12 02:10
