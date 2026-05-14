# Strategy: 4h_1d_camarilla_breakout_v17

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.192 | +8.4% | -13.3% | 179 | FAIL |
| ETHUSDT | 0.135 | +26.6% | -14.1% | 162 | PASS |
| SOLUSDT | 0.672 | +98.4% | -23.4% | 135 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.945 | +23.6% | -8.8% | 53 | PASS |
| SOLUSDT | -0.054 | +4.0% | -12.4% | 48 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v17
# Hypothesis: 4-hour breakouts at Camarilla pivot levels (H3/L3) from daily timeframe with volume confirmation (>2x 20-bar average volume).
# Camarilla levels act as intraday support/resistance; breaks signal momentum continuation.
# Volume filter reduces false breakouts. Works in bull markets (upward breaks) and bear markets (downward breaks).
# Target: 20-50 trades per year per symbol (~80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v17"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily close for Camarilla levels
    daily_close = df_1d['close'].values
    
    # Camarilla levels: H3/L3 = C ± (H-L)*1.1/2
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    camarilla_h3 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_l3 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below L3 level
            if close[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above H3 level
            if close[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volume confirmation
            if close[i] > camarilla_h3_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 08:36
