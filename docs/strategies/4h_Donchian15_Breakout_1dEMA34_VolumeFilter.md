# Strategy: 4h_Donchian15_Breakout_1dEMA34_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.624 | +61.7% | -10.8% | 143 | PASS |
| ETHUSDT | 0.186 | +30.7% | -15.2% | 141 | PASS |
| SOLUSDT | 0.809 | +145.2% | -27.0% | 145 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.024 | -6.4% | -11.0% | 56 | FAIL |
| ETHUSDT | 0.106 | +6.9% | -12.5% | 47 | PASS |
| SOLUSDT | -0.453 | -4.8% | -16.8% | 51 | FAIL |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA 34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian channels (15-period for more signals)
    highest_high = pd.Series(high).rolling(window=15, min_periods=15).max().values
    lowest_low = pd.Series(low).rolling(window=15, min_periods=15).min().values
    
    # Volume filter: volume > 1.3x 20-period average (moderate filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Long conditions: price breaks above upper Donchian + above 1d EMA + volume
        long_breakout = (close[i] > highest_high[i-1] and price_above_ema and volume_filter[i])
        # Short conditions: price breaks below lower Donchian + below 1d EMA + volume
        short_breakout = (close[i] < lowest_low[i-1] and price_below_ema and volume_filter[i])
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout
        elif position == 1 and close[i] < lowest_low[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i-1]:
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

name = "4h_Donchian15_Breakout_1dEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 21:48
