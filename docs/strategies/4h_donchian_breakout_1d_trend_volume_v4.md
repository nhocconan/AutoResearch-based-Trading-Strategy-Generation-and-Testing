# Strategy: 4h_donchian_breakout_1d_trend_volume_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.247 | +34.6% | -14.5% | 75 | PASS |
| ETHUSDT | -0.488 | -15.9% | -33.2% | 81 | FAIL |
| SOLUSDT | 0.803 | +151.4% | -29.5% | 63 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.133 | +3.7% | -6.9% | 25 | FAIL |
| SOLUSDT | 0.397 | +14.2% | -13.7% | 24 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Donchian channel breakouts on 4h timeframe, filtered by 1d EMA trend and volume spikes,
provide high-probability entries with controlled trade frequency. Works in bull markets via breakouts
and in bear markets via short breakdowns. Uses 1d trend filter for multi-timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v4"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend reverses
            if (close[i] < low_20[i] or 
                close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend reverses
            if (close[i] > high_20[i] or 
                close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1d EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: price breaks above Donchian high with uptrend and volume spike
            if (high[i] > high_20[i-1] and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with downtrend and volume spike
            elif (low[i] < low_20[i-1] and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 01:56
