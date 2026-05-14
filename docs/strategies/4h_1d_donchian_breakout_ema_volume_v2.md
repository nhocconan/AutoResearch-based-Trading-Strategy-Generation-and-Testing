# Strategy: 4h_1d_donchian_breakout_ema_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.094 | +16.2% | -12.9% | 156 | FAIL |
| ETHUSDT | 0.516 | +50.3% | -10.7% | 142 | PASS |
| SOLUSDT | 0.465 | +60.1% | -26.2% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.550 | +13.9% | -6.1% | 53 | PASS |
| SOLUSDT | -0.098 | +3.8% | -14.7% | 45 | FAIL |

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
    
    # Hypothesis: 4h Donchian(20) breakout + 1d EMA(20) trend filter + volume confirmation (>2.0x 20-period MA).
    # Donchian channels capture institutional breakouts; 1d EMA ensures alignment with long-term trend.
    # High volume filter (2.0x MA) confirms strong institutional participation. Discrete sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 75-150 total trades over 4 years (19-38/year) to stay within fee drag limits.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(20) trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-period volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels on primary timeframe (4h)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d EMA to 4h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0 * 20-period MA (stricter to reduce trades)
        volume_filter = volume[i] > 2.0 * volume_ma[i]
        
        # Trend filter: price above/below 1d EMA(20)
        uptrend = close[i] > ema_20_1d_aligned[i]
        downtrend = close[i] < ema_20_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = (close[i] > highest_high[i-1]) and volume_filter and uptrend
        short_breakout = (close[i] < lowest_low[i-1]) and volume_filter and downtrend
        
        # Exit conditions: price returns to midpoint of Donchian channel
        midpoint = (highest_high[i] + lowest_low[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_ema_volume_v2"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 05:47
