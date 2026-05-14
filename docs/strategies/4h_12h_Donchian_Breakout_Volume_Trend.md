# Strategy: 4h_12h_Donchian_Breakout_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.225 | +33.1% | -18.9% | 117 | PASS |
| ETHUSDT | 0.307 | +42.3% | -14.1% | 109 | PASS |
| SOLUSDT | 1.158 | +277.4% | -26.3% | 109 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.003 | -6.6% | -12.8% | 46 | FAIL |
| ETHUSDT | 0.852 | +24.4% | -7.8% | 35 | PASS |
| SOLUSDT | 0.438 | +14.9% | -12.9% | 36 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_12h_Donchian_Breakout_Volume_Trend
Hypothesis: Combines 4-hour Donchian breakout with 12-hour EMA trend filter and volume confirmation.
Trades only in direction of higher timeframe trend to avoid counter-trend whipsaws.
Designed for 15-30 trades/year per symbol with high win rate during trends.
Works in bull/bear by following 12h trend direction - avoids counter-trend losses.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Breakout conditions using Donchian channels
        breakout_up = close[i] > high_20[i-1]  # Break above 20-period high
        breakdown_down = close[i] < low_20[i-1]  # Break below 20-period low
        
        # Entry conditions: only trade in direction of 12h trend
        long_entry = breakout_up and volume_filter and uptrend
        short_entry = breakdown_down and volume_filter and downtrend
        
        # Exit conditions: return to opposite Donchian level or trend reversal
        long_exit = (close[i] < low_20[i]) or (not uptrend)  # Break below 20-period low or trend change
        short_exit = (close[i] > high_20[i]) or (not downtrend)  # Break above 20-period high or trend change
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 23:42
