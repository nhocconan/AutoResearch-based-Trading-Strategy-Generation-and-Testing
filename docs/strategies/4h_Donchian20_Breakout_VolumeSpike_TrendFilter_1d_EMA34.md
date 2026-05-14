# Strategy: 4h_Donchian20_Breakout_VolumeSpike_TrendFilter_1d_EMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.576 | +61.0% | -14.7% | 91 | PASS |
| ETHUSDT | 0.108 | +23.9% | -21.4% | 96 | PASS |
| SOLUSDT | 0.823 | +160.9% | -37.2% | 94 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.227 | -9.1% | -16.3% | 43 | FAIL |
| ETHUSDT | 0.303 | +10.9% | -10.0% | 36 | PASS |
| SOLUSDT | 0.677 | +21.0% | -12.4% | 27 | PASS |

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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA for trend filter (1d)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34 = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34[i]) or np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above upper Donchian + trend up + volume spike
        long_breakout = (close[i] > highest_high[i-1] and close[i] > ema34[i] and volume_spike[i])
        # Short conditions: price breaks below lower Donchian + trend down + volume spike
        short_breakout = (close[i] < lowest_low[i-1] and close[i] < ema34[i] and volume_spike[i])
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout with volume
        elif position == 1 and close[i] < lowest_low[i-1] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i-1] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_TrendFilter_1d_EMA34"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 21:27
