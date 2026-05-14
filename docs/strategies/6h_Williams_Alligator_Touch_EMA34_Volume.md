# Strategy: 6h_Williams_Alligator_Touch_EMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.040 | +22.4% | -8.3% | 145 | PASS |
| ETHUSDT | 0.005 | +21.2% | -9.8% | 128 | PASS |
| SOLUSDT | 0.600 | +52.5% | -14.7% | 97 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.719 | +1.8% | -3.9% | 52 | FAIL |
| ETHUSDT | 0.241 | +8.3% | -6.1% | 47 | PASS |
| SOLUSDT | -1.154 | -2.7% | -5.2% | 28 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Williams_Alligator_Touch_EMA34_Volume
Hypothesis: Williams Alligator (13,8,5 SMAs) touching price with volume confirmation and 1d EMA34 trend filter on 6h timeframe.
Long when price touches Alligator's Jaw (13 SMA) from below with volume spike in uptrend.
Short when price touches Jaw from above with volume spike in downtrend.
Uses 1d EMA34 to filter trend direction and avoid counter-trend trades.
Target: 15-25 trades/year to minimize fee drag while capturing strong trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA34 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 13)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(sma_5[i]) or np.isnan(sma_8[i]) or np.isnan(sma_13[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw = sma_13[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price touches Jaw from below with volume spike in uptrend
            if (low[i] <= jaw <= high[i] and  # price touches Jaw
                close[i] > jaw and           # closes above Jaw (confirms upward touch)
                vol_spike and
                price > ema34):              # uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: price touches Jaw from above with volume spike in downtrend
            elif (low[i] <= jaw <= high[i] and  # price touches Jaw
                  close[i] < jaw and           # closes below Jaw (confirms downward touch)
                  vol_spike and
                  price < ema34):              # downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below Teeth (8 SMA) or trend reverses
            if close[i] < sma_8[i] or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above Teeth (8 SMA) or trend reverses
            if close[i] > sma_8[i] or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Williams_Alligator_Touch_EMA34_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-18 03:38
