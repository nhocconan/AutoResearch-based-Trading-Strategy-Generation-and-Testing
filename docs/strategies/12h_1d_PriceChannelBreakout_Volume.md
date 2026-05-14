# Strategy: 12h_1d_PriceChannelBreakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.224 | +3.0% | -19.4% | 156 | FAIL |
| ETHUSDT | 0.023 | +15.8% | -19.2% | 191 | PASS |
| SOLUSDT | 0.708 | +133.5% | -35.4% | 170 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.460 | +15.5% | -10.7% | 55 | PASS |
| SOLUSDT | 0.384 | +14.2% | -14.1% | 53 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
12h_1d_PriceChannelBreakout_Volume
Hypothesis: Breakouts above/below 12h Donchian channels (20-period) in the direction of 1d EMA(50) trend, confirmed by volume >1.5x 20-period average. Uses 1d EMA to filter counter-trend trades. Position size 0.25, targeting 15-25 trades/year to avoid fee drag. Works in bull/bear by trading breakouts with trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 12h calculations
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's OHLC (completed bar)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    prev_close_12h[0] = close_12h[0]
    
    # 12h Donchian channel (20-period)
    donchian_period = 20
    upper = np.full_like(high_12h, np.nan)
    lower = np.full_like(low_12h, np.nan)
    
    if len(high_12h) >= donchian_period:
        for i in range(donchian_period - 1, len(high_12h)):
            upper[i] = np.max(high_12h[i - donchian_period + 1:i + 1])
            lower[i] = np.min(low_12h[i - donchian_period + 1:i + 1])
    
    # 1d EMA trend filter (50-period)
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 12h Donchian levels to 12h timeframe (no additional delay needed)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    
    # Align 1d EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, donchian_period, ema_period, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and above 1d EMA
            if close[i] > upper_aligned[i] and vol_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume and below 1d EMA
            elif close[i] < lower_aligned[i] and vol_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below lower Donchian (reverse signal) or below 1d EMA
            if close[i] < lower_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper Donchian (reverse signal) or above 1d EMA
            if close[i] > upper_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_PriceChannelBreakout_Volume"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-18 15:47
