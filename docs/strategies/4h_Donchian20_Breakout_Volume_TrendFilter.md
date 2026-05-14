# Strategy: 4h_Donchian20_Breakout_Volume_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.297 | +39.5% | -18.7% | 444 | PASS |
| ETHUSDT | 0.121 | +24.7% | -19.5% | 486 | PASS |
| SOLUSDT | 0.368 | +58.4% | -36.4% | 580 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.087 | +6.7% | -8.1% | 167 | PASS |
| ETHUSDT | 1.100 | +32.5% | -10.6% | 157 | PASS |
| SOLUSDT | 0.713 | +23.7% | -15.7% | 169 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_TrendFilter
Hypothesis: Trade Donchian(20) breakouts in direction of 1d EMA(34) trend with volume > 1.5x 24-period average. Uses 1d trend filter to avoid counter-trend trades. Position size 0.25 targeting ~30 trades/year to minimize fee filter. Works in bull/bear by trading breakouts with trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) trend filter (previous day's close)
    close_1d = df_1d['close'].values
    ema_period = 34
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align daily EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback, len(high)):
        highest_high[i] = np.max(high[i - lookback:i])
        lowest_low[i] = np.min(low[i - lookback:i])
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and above daily EMA
            if close[i] > highest_high[i] and vol_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and below daily EMA
            elif close[i] < lowest_low[i] and vol_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below Donchian low or below daily EMA
            if close[i] < lowest_low[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Donchian high or above daily EMA
            if close[i] > highest_high[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 16:14
