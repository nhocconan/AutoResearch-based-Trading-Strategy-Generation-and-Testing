# Strategy: 12h_1dCamarilla_R1_S1_Breakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.388 | +5.0% | -17.7% | 237 | FAIL |
| ETHUSDT | 0.342 | +38.0% | -13.6% | 225 | PASS |
| SOLUSDT | 0.053 | +19.5% | -33.0% | 193 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.104 | +7.0% | -13.4% | 83 | PASS |
| SOLUSDT | -1.131 | -10.0% | -18.4% | 74 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation
# - Uses 1d Camarilla pivot levels (R1-R4, S1-S4) for support/resistance
# - Uses 12h volume spike for entry confirmation
# - Enters long when price breaks above R1 with volume spike
# - Enters short when price breaks below S1 with volume spike
# - Exits when price returns to pivot point (PP) or opposite side
# - Designed to capture intraday momentum with institutional level respect
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1dCamarilla_R1_S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and ranges
    pp = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r1 = pp + (range_hl * 1.1 / 12)
    r2 = pp + (range_hl * 1.1 / 6)
    r3 = pp + (range_hl * 1.1 / 4)
    r4 = pp + (range_hl * 1.1 / 2)
    s1 = pp - (range_hl * 1.1 / 12)
    s2 = pp - (range_hl * 1.1 / 6)
    s3 = pp - (range_hl * 1.1 / 4)
    s4 = pp - (range_hl * 1.1 / 2)
    
    # Align 1d Camarilla levels to 12h timeframe
    pp_12h = align_htf_to_ltf(prices, df_1d, pp)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter (12h timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma_10)  # Moderate volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(pp_12h[i]) or np.isnan(r1_12h[i]) or 
            np.isnan(s1_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike
            if close[i] > r1_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike
            elif close[i] < s1_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point OR breaks below S1
            if close[i] < pp_12h[i] or close[i] < s1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point OR breaks above R1
            if close[i] > pp_12h[i] or close[i] > r1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 23:30
