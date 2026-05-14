# Strategy: 4h_Donchian_20_Breakout_Volume_Trend_HTF

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.032 | +20.9% | -13.2% | 164 | PASS |
| ETHUSDT | 0.257 | +35.3% | -13.7% | 153 | PASS |
| SOLUSDT | 0.240 | +36.9% | -34.5% | 145 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.851 | -2.7% | -8.0% | 66 | FAIL |
| ETHUSDT | 0.318 | +10.7% | -9.6% | 52 | PASS |
| SOLUSDT | 0.119 | +7.1% | -16.8% | 47 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_Trend_HTF
Strategy: 4h Donchian(20) breakout with volume confirmation and 12h trend filter.
- Long when price breaks above 20-period high + volume > 1.8x 20-period avg + 12h close > 12h EMA34
- Short when price breaks below 20-period low + volume > 1.8x 20-period avg + 12h close < 12h EMA34
- Exit when price returns to 20-period midpoint or opposite breakout occurs
- Position size: ±0.25
- Uses 4h timeframe as primary with 12h for trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels and midpoint
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max20 + low_min20) / 2.0
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema34_12h = close_series_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20, 34)  # Donchian20, volume MA20, EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max20[i]) or 
            np.isnan(low_min20[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Breakout conditions
        breakout_up = close[i] > high_max20[i-1]  # break above 20-period high
        breakout_down = close[i] < low_min20[i-1]  # break below 20-period low
        
        # Return to midpoint for exit
        return_to_mid = abs(close[i] - donchian_mid[i]) < 0.002 * close[i]  # within 0.2% of midpoint
        
        if position == 0:
            # Long: breakout up + volume filter + 12h uptrend
            if breakout_up and volume_filter and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + volume filter + 12h downtrend
            elif breakout_down and volume_filter and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to midpoint or opposite breakout
            if return_to_mid or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to midpoint or opposite breakout
            if return_to_mid or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Breakout_Volume_Trend_HTF"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 07:54
