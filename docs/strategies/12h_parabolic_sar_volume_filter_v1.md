# Strategy: 12h_parabolic_sar_volume_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.391 | -1.7% | -13.4% | 205 | FAIL |
| ETHUSDT | 0.081 | +22.5% | -15.0% | 206 | PASS |
| SOLUSDT | 0.940 | +169.3% | -17.6% | 185 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.305 | +11.0% | -9.1% | 63 | PASS |
| SOLUSDT | -0.562 | -6.2% | -17.9% | 63 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h Parabolic SAR with Volume Filter
Long when Parabolic SAR flips below price with above-average volume
Short when Parabolic SAR flips above price with above-average volume
Exit when SAR flips opposite direction
Parabolic SAR works in trending markets (both bull and bear) and volume filter reduces whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_parabolic_sar_volume_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Parabolic SAR ===
    # Initialize
    psar = np.zeros(n)
    bull = True  # True for long, False for short
    af = 0.02    # acceleration factor
    max_af = 0.2
    ep = high[0] if bull else low[0]  # extreme point
    psar[0] = low[0] if bull else high[0]
    
    # Calculate SAR
    for i in range(1, n):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR is within prior period's range
            psar[i] = min(psar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR is within prior period's range
            psar[i] = max(psar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
        
        # Reverse if price crosses SAR
        reverse = False
        if bull and low[i] < psar[i]:
            bull = False
            reverse = True
            ep = low[i]
            af = 0.02
        elif not bull and high[i] > psar[i]:
            bull = True
            reverse = True
            ep = high[i]
            af = 0.02
        
        if reverse:
            psar[i] = ep  # SAR at reversal point is the extreme point
        else:
            # Update extreme point and acceleration factor
            if bull:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(psar[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: SAR flips above price (trend reversal)
            if psar[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: SAR flips below price (trend reversal)
            if psar[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.3:
                signals[i] = 0.0
                continue
            
            # Entry: SAR flip with volume confirmation
            if close[i] > psar[i]:
                # Price above SAR -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < psar[i]:
                # Price below SAR -> short
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 23:06
