# Strategy: 4h_4H_Camarilla_R2_S2_Breakout_1dEMA34_Trend_VolumeS

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.432 | +32.4% | -3.2% | 217 | PASS |
| ETHUSDT | 0.250 | +29.0% | -5.9% | 200 | PASS |
| SOLUSDT | 0.306 | +34.4% | -8.3% | 168 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.795 | -2.1% | -5.0% | 79 | FAIL |
| ETHUSDT | 0.423 | +9.4% | -4.6% | 66 | PASS |
| SOLUSDT | 0.640 | +10.7% | -3.5% | 55 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_4H_Camarilla_R2_S2_Breakout_1dEMA34_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Data for EMA34 trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA34 for trend ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Daily data for Camarilla levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R2, S2) from previous day
    # R2 = Close + (High - Low) * 1.1/2
    # S2 = Close - (High - Low) * 1.1/2
    camarilla_range = (high_1d - low_1d) * 1.1
    r2_level = close_1d + camarilla_range / 2.0
    s2_level = close_1d - camarilla_range / 2.0
    
    # Align Camarilla levels to 4h
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_level)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_level)
    
    # === Volume spike detection (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 34)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R2 with volume spike and 1d trend up
            if (close[i] > r2_aligned[i] and 
                volume_spike[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S2 with volume spike and 1d trend down
            elif (close[i] < s2_aligned[i] and 
                  volume_spike[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses back below R2 or trend breaks
            if close[i] < r2_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses back above S2 or trend breaks
            if close[i] > s2_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 05:23
