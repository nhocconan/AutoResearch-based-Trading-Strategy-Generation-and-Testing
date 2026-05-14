# Strategy: 12h_1d_Donchian20_Breakout_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.737 | -9.4% | -18.4% | 178 | FAIL |
| ETHUSDT | 0.101 | +24.5% | -15.5% | 150 | PASS |
| SOLUSDT | -0.345 | -12.6% | -32.5% | 148 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.051 | +6.1% | -8.6% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Donchian breakout and volume confirmation.
# Uses 1d Donchian channels (20-period high/low) to define breakout levels,
# with 12h close breaking above/below previous period's high/low.
# Volume filter ensures breakout strength. Works in both bull and bear markets
# by capturing breakouts in either direction.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "12h_1d_Donchian20_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels on 1d timeframe (20-period high/low)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Use previous 12h period high/low for breakout levels
        prev_high = high[i-1]
        prev_low = low[i-1]
        
        if position == 0:
            # Long when price breaks above previous high with volume
            if close[i] > prev_high and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below previous low with volume
            elif close[i] < prev_low and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below previous low
            if close[i] < prev_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above previous high
            if close[i] > prev_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 16:51
