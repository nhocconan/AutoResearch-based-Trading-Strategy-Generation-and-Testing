# Strategy: 4h_VolumeSpike_CamarillaR3S3_Breakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.120 | +25.7% | -13.0% | 243 | PASS |
| ETHUSDT | 0.347 | +42.6% | -15.2% | 227 | PASS |
| SOLUSDT | 0.489 | +68.9% | -26.7% | 205 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.345 | -8.5% | -12.5% | 91 | FAIL |
| ETHUSDT | 0.126 | +7.3% | -15.3% | 80 | PASS |
| SOLUSDT | -0.014 | +4.7% | -15.2% | 67 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_VolumeSpike_CamarillaR3S3_Breakout
Hypothesis: Focus on volume spikes (>2x 20-period average) combined with breakouts of 1d-derived Camarilla R3/S3 levels. Uses price action (close > level) for entry and opposite level for exit. Designed for low trade frequency (<30/year) to minimize fee drift, working in trending and ranging markets by requiring volatility expansion.
"""

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
    
    # Calculate Camarilla levels from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for previous day's close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for volume average
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation
            if close[i] > camarilla_r3_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with volume confirmation
            elif close[i] < camarilla_s3_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (opposite level)
            if close[i] < camarilla_s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R3 (opposite level)
            if close[i] > camarilla_r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_VolumeSpike_CamarillaR3S3_Breakout"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 04:09
