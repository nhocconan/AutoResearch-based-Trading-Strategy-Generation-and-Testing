# Strategy: 12h_PivotBreakout_VolumeSurge_ATRFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.403 | +4.1% | -25.6% | 247 | FAIL |
| ETHUSDT | 0.268 | +33.8% | -17.1% | 212 | PASS |
| SOLUSDT | 0.077 | +21.1% | -35.1% | 194 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.185 | +8.1% | -7.3% | 75 | PASS |
| SOLUSDT | -0.764 | -5.4% | -16.0% | 67 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Daily high/low for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculation
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # 12h volatility filter - ATR(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S1 with volume surge and ATR filter
            if (close[i] > s1_12h[i] and vol_surge[i] and atr[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R1 with volume surge and ATR filter
            elif (close[i] < r1_12h[i] and vol_surge[i] and atr[i] > 0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses pivot or volatility drops
            if position == 1:
                if close[i] < pivot_12h[i] or atr[i] < 0.5 * atr[i]:  # Volatility drop
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_12h[i] or atr[i] < 0.5 * atr[i]:  # Volatility drop
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_PivotBreakout_VolumeSurge_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-22 05:08
