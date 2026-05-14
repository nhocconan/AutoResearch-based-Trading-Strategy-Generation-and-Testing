# Strategy: 4h_Volume_Spike_Trend_Following_12hFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.057 | +22.9% | -11.5% | 369 | PASS |
| ETHUSDT | 0.308 | +31.1% | -7.1% | 331 | PASS |
| SOLUSDT | -0.353 | +2.5% | -15.8% | 272 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.105 | -0.7% | -4.3% | 144 | FAIL |
| ETHUSDT | 1.212 | +18.7% | -4.1% | 126 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Volume Spike + Trend Following with 12h Trend Filter
Hypothesis: Volume spikes confirm institutional interest. Combined with 12h EMA trend filter,
this captures strong momentum moves in both bull and bear markets. Volume acts as confirmation
rather than entry signal, keeping trade frequency low.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume spike detection: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Price momentum: close > previous close
    price_up = close > np.roll(close, 1)
    price_down = close < np.roll(close, 1)
    # Handle first element
    price_up[0] = False
    price_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_12h_aligned[i]
        vol_ok = vol_spike[i]
        up = price_up[i]
        down = price_down[i]
        
        if position == 0:
            # Enter long on volume spike + upward momentum + uptrend
            if vol_ok and up and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short on volume spike + downward momentum + downtrend
            elif vol_ok and down and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on momentum reversal or trend change
            if down or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on momentum reversal or trend change
            if up or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Volume_Spike_Trend_Following_12hFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 06:12
