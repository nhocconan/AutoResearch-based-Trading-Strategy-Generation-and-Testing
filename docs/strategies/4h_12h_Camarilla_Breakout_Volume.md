# Strategy: 4h_12h_Camarilla_Breakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.021 | +18.3% | -22.4% | 102 | PASS |
| ETHUSDT | 0.219 | +34.2% | -28.5% | 80 | PASS |
| SOLUSDT | 0.942 | +215.3% | -37.5% | 66 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.646 | +15.2% | -7.9% | 28 | PASS |
| ETHUSDT | 0.031 | +4.9% | -17.7% | 31 | PASS |
| SOLUSDT | 0.378 | +13.8% | -14.6% | 26 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_12h_Camarilla_Breakout_Volume
Hypothesis: Breakout above R4 or below S4 of 12h Camarilla levels with volume expansion.
12h Camarilla provides stronger support/resistance than 1h/4h, reducing false breakouts.
Volume confirmation ensures institutional participation. Works in both bull (breakouts up)
and bear (breakdowns down) markets. Target: 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    range_val = high - low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    C = close
    H = high
    L = low
    R1 = C + ((H - L) * 1.0833)
    R2 = C + ((H - L) * 1.1666)
    R3 = C + ((H - L) * 1.2500)
    R4 = C + ((H - L) * 1.5000)
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels
    R1_12h, R2_12h, R3_12h, R4_12h, S1_12h, S2_12h, S3_12h, S4_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Align all data to 4h timeframe
    R4_12h_aligned = align_htf_to_ltf(prices, df_12h, R4_12h)
    S4_12h_aligned = align_htf_to_ltf(prices, df_12h, S4_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_12h_aligned[i]) or np.isnan(S4_12h_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above R4 with volume expansion
        long_condition = (close[i] > R4_12h_aligned[i]) and volume_expansion[i]
        
        # Short: breakdown below S4 with volume expansion
        short_condition = (close[i] < S4_12h_aligned[i]) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_12h_Camarilla_Breakout_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 17:43
