# Strategy: 4h_DonchianBreakout_12hEMA50_TrendFilter_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.150 | +27.5% | -11.7% | 192 | PASS |
| ETHUSDT | 0.189 | +30.8% | -16.8% | 186 | PASS |
| SOLUSDT | 1.296 | +289.1% | -25.3% | 176 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.342 | -9.4% | -10.8% | 71 | FAIL |
| ETHUSDT | 0.239 | +9.5% | -10.6% | 59 | PASS |
| SOLUSDT | -0.149 | +1.4% | -14.3% | 63 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
# Donchian breakouts capture breakouts from volatility contractions, which often precede strong moves.
# EMA filter ensures we trade in the direction of the higher timeframe trend, avoiding counter-trend trades.
# Volume confirmation ensures breakouts have institutional participation, reducing false breakouts.
# Works in bull markets (catching uptrends) and bear markets (catching downtrends) by using EMA direction.
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).

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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(50)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA(50)
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] > highest_high[i-1]  # Current high exceeds prior period's highest high
        breakout_down = low[i] < lowest_low[i-1]   # Current low is below prior period's lowest low
        
        # Entry conditions with volume confirmation
        long_entry = uptrend and breakout_up and volume_filter[i]
        short_entry = downtrend and breakout_down and volume_filter[i]
        
        # Exit conditions: when trend reverses or opposite breakout occurs
        long_exit = (not uptrend) or breakout_down
        short_exit = (not downtrend) or breakout_up
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_12hEMA50_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 09:14
