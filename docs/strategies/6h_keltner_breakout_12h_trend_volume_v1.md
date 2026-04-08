# Strategy: 6h_keltner_breakout_12h_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.199 | +8.3% | -13.8% | 84 | FAIL |
| ETHUSDT | 0.091 | +23.4% | -14.7% | 72 | PASS |
| SOLUSDT | 0.725 | +112.6% | -23.6% | 62 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.170 | +8.1% | -10.2% | 29 | PASS |
| SOLUSDT | -0.406 | -3.8% | -22.9% | 26 | FAIL |

## Code
```python
#/usr/bin/env python3
"""
6h Keltner Channel Breakout + 12h Trend + Volume Confirmation v1
Hypothesis: Keltner Channel breakouts with volatility-adjusted bands, filtered by 12h EMA trend and volume confirmation, capture sustained moves while avoiding whipsaws. The 6h timeframe targets 15-40 trades/year, balancing responsiveness with low turnover. Volume validates breakout strength, and 12h trend ensures alignment with higher-timeframe momentum, working in both bull and bear regimes by adapting to volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_breakout_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(30) for trend filter
    ema_30_12h = df_12h['close'].ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    # 6h ATR(20) for Keltner Channel
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar TR
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 6h EMA(20) for Keltner Channel middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    upper_keltner = ema_20 + (2.0 * atr)
    lower_keltner = ema_20 - (2.0 * atr)
    
    # Volume filter (>1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_30_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower Keltner or trend reverses
            if close[i] <= lower_keltner[i] or close[i] < ema_30_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper Keltner or trend reverses
            if close[i] >= upper_keltner[i] or close[i] > ema_30_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with trend alignment and volume
            if (close[i] >= upper_keltner[i] and 
                close[i] > ema_30_12h_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with trend alignment and volume
            elif (close[i] <= lower_keltner[i] and 
                  close[i] < ema_30_12h_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 00:28
