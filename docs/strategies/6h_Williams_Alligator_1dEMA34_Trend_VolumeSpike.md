# Strategy: 6h_Williams_Alligator_1dEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.603 | +44.1% | -6.0% | 140 | PASS |
| ETHUSDT | 0.383 | +38.1% | -13.9% | 126 | PASS |
| SOLUSDT | 0.259 | +36.7% | -23.0% | 116 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.497 | +1.8% | -4.4% | 57 | FAIL |
| ETHUSDT | 0.269 | +9.1% | -5.0% | 52 | PASS |
| SOLUSDT | -0.507 | -1.1% | -8.7% | 44 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d EMA34 Trend Filter and Volume Spike
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via aligned SMAs
- Only trade when Alligator is "eating" (jaws, teeth, lips aligned and separated) in direction of 1d EMA34
- Volume confirmation (> 2.0x 20-period MA) ensures breakout validity
- Designed for 6h timeframe to capture medium-term trends with controlled trade frequency
- Works in bull via long alignments and bear via short alignments
- Target: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components (SMAs of median price)
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)  # need Alligator jaw, EMA34_1d, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator aligned bullish (lips > teeth > jaw) AND price > 1d EMA34 AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned bearish (lips < teeth < jaw) AND price < 1d EMA34 AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator loses alignment OR price crosses 1d EMA34
            exit_signal = False
            if position == 1:
                # Exit long when Alligator turns bearish OR price < 1d EMA34
                if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when Alligator turns bullish OR price > 1d EMA34
                if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Williams_Alligator_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 15:57
