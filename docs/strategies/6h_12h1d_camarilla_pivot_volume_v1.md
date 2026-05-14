# Strategy: 6h_12h1d_camarilla_pivot_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.156 | +28.0% | -16.3% | 141 | PASS |
| ETHUSDT | 0.338 | +43.5% | -16.4% | 116 | PASS |
| SOLUSDT | 0.485 | +74.2% | -27.4% | 105 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.407 | +0.6% | -12.0% | 40 | FAIL |
| ETHUSDT | 0.756 | +20.7% | -7.5% | 40 | PASS |
| SOLUSDT | 0.208 | +9.0% | -15.0% | 35 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_12h1d_camarilla_pivot_volume_v1
# Hypothesis: 6h strategy using Camarilla pivot levels from 1d timeframe with volume confirmation.
# Long: Price breaks above Camarilla H3 level, volume > 1.5x 20-period average.
# Short: Price breaks below Camarilla L3 level, volume > 1.5x 20-period average.
# Exit: Opposite pivot break.
# Uses 1d Camarilla pivots for structure, volume confirmation to filter weak breakouts.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h1d_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from 1d OHLC
    # Camarilla: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla H3 and L3 levels
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align HTF Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below L3
            if low[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above H3
            if high[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H3, volume confirmed
            if (high[i] > camarilla_h3_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3, volume confirmed
            elif (low[i] < camarilla_l3_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 00:37
