# Strategy: 6h_12h_EMA50_Donchian_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.342 | -0.0% | -17.8% | 98 | FAIL |
| ETHUSDT | 0.191 | +31.1% | -13.0% | 96 | PASS |
| SOLUSDT | 1.201 | +253.2% | -20.9% | 83 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.526 | +15.8% | -9.0% | 29 | PASS |
| SOLUSDT | -0.146 | +1.0% | -16.3% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h trend filter and volume confirmation
# Trend from 12h EMA (50) provides directional bias to avoid counter-trend trades
# 6h Donchian(20) breakout captures momentum in direction of 12h trend
# Volume > 1.5x average confirms institutional participation
# Works in bull/bear as 12h EMA adapts to trend
# Target: 20-40 trades/year per symbol (80-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(50) for trend filter
    ema_len = 50
    if len(df_12h) < ema_len:
        return np.zeros(n)
    
    ema_12h = pd.Series(df_12h['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channel (20 periods) on 6h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA50
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + above 12h EMA + volume
            if (close[i] > dc_upper[i] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + below 12h EMA + volume
            elif (close[i] < dc_lower[i] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 12h EMA or breaks below Donchian lower
            if close[i] < ema_12h_aligned[i] or close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 12h EMA or breaks above Donchian upper
            if close[i] > ema_12h_aligned[i] or close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_EMA50_Donchian_Volume_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-14 06:58
