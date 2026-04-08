# Strategy: 4h_donchian_breakout_1d_trend_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.224 | +34.3% | -16.2% | 68 | KEEP |
| ETHUSDT | -0.563 | -29.1% | -44.8% | 77 | DISCARD |
| SOLUSDT | 0.948 | +256.6% | -37.3% | 69 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.104 | +3.5% | -8.9% | 22 | DISCARD |
| SOLUSDT | 0.299 | +11.9% | -19.1% | 22 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend Filter - Version 1
Hypothesis: Donchian channel breakouts capture volatility expansion, while 1-day trend (price > 50 EMA) filters for directional moves, reducing whipsaws in ranging markets. Volume confirmation (>1.5x average) ensures institutional participation. Works in both bull and bear markets by trading breakouts in the direction of the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_filter_v1"
timeframe = "4h"
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
    
    # Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend reverses (price < EMA50)
            if close[i] < donch_low[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend reverses (price > EMA50)
            if close[i] > donch_high[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long: breakout above Donchian high + price > EMA50 + volume filter
            if (close[i] > donch_high[i-1] and 
                close[i] > ema_50_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.30
            # Short: breakout below Donchian low + price < EMA50 + volume filter
            elif (close[i] < donch_low[i-1] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-04-08 00:03
