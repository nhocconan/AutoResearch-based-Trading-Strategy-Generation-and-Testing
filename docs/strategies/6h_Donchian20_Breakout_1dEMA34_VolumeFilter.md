# Strategy: 6h_Donchian20_Breakout_1dEMA34_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.042 | +21.0% | -12.6% | 75 | PASS |
| ETHUSDT | 0.311 | +41.3% | -12.5% | 72 | PASS |
| SOLUSDT | 0.899 | +159.4% | -26.7% | 65 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.175 | -8.6% | -13.0% | 30 | FAIL |
| ETHUSDT | 0.169 | +8.1% | -10.4% | 27 | PASS |
| SOLUSDT | -0.400 | -4.0% | -18.7% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 1d EMA 34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h Donchian channels (20-period for structure)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.8x 30-period average (strong filter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
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

name = "6h_Donchian20_Breakout_1dEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 21:59
