# Strategy: 6h_1dCamarilla_R4_S4_Breakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.599 | +4.0% | -16.5% | 151 | FAIL |
| ETHUSDT | 0.201 | +28.7% | -10.5% | 131 | PASS |
| SOLUSDT | 0.209 | +31.3% | -12.6% | 81 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.996 | +18.4% | -6.3% | 53 | PASS |
| SOLUSDT | -0.522 | +0.2% | -14.0% | 39 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume spike confirmation
# Long when price breaks above 1d Camarilla R4 level AND 6h volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Camarilla S4 level AND 6h volume > 2.0 * avg_volume(20)
# Exit when price returns to 1d Camarilla midpoint (PP)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla R4/S4 represent strong breakout levels from 1d structure
# Volume spike filters for institutional participation in breakouts
# Works in both bull (continuation breakouts) and bear (continuation breakdowns) markets

name = "6h_1dCamarilla_R4_S4_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 completed daily bars for Camarilla
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R4 = PP + (High - Low) * 1.1/2
    # S4 = PP - (High - Low) * 1.1/2
    pp = (high_1d + low_1d + close_1d) / 3.0
    r4 = pp + (high_1d - low_1d) * 1.1 / 2.0
    s4 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R4 level with volume spike
            if (close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S4 level with volume spike
            elif (close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Camarilla pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1d Camarilla pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 05:28
