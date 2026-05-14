# Strategy: 12H_Camarilla_S3R3_Breakout_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.587 | -0.8% | -23.4% | 184 | FAIL |
| ETHUSDT | 0.003 | +19.9% | -20.6% | 161 | PASS |
| SOLUSDT | -0.259 | -2.2% | -37.7% | 143 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.180 | +8.0% | -6.8% | 60 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12-hour Camarilla pivot (S3/R3) breakout with 1-day volume spike filter.
Long when price breaks above R3 with volume > 1.5x 20-period average volume.
Short when price breaks below S3 with volume > 1.5x 20-period average volume.
Exit when price returns to the Camarilla H-L (close) level.
Camarilla levels derived from prior 1-day range provide institutional support/resistance.
Volume filter ensures breakout validity. Works in trending and ranging markets by
filtering false breakouts. Designed for low trade frequency (~15-25/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla pivot and volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1-day range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    camarilla_h_l = (high_1d + low_1d + close_1d) / 3.0  # H-L close level for exit
    
    # Align Camarilla levels to 12h timeframe (wait for prior day's close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h_l_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h_l)
    
    # Volume filter: 20-period average volume on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h_l_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume confirmation
            if close[i] > camarilla_r3_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to H-L level
                if close[i] <= camarilla_h_l_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to H-L level
                if close[i] >= camarilla_h_l_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_S3R3_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-22 23:39
