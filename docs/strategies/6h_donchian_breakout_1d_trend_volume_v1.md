# Strategy: 6h_donchian_breakout_1d_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.030 | +15.3% | -16.6% | 62 | FAIL |
| ETHUSDT | -0.485 | -17.4% | -33.7% | 78 | FAIL |
| SOLUSDT | 1.104 | +260.8% | -28.4% | 67 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.176 | +8.2% | -15.2% | 23 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_donchian_breakout_1d_trend_volume_v1
Hypothesis: Donchian channel breakouts (20-period) on 6h timeframe filtered by 1-day EMA50 trend and volume confirmation.
In long: price breaks above upper Donchian band with volume > 20-period average and price above 1d EMA50.
In short: price breaks below lower Donchian band with volume > 20-period average and price below 1d EMA50.
Uses Donchian for breakout signals, EMA for trend direction, and volume for confirmation.
Designed for 15-30 trades/year on 6h timeframe with clear breakout logic that works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channel (20-period)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper[i-1]  # Break above previous upper band
        breakout_down = close[i] < lower[i-1]  # Break below previous lower band
        
        # 1d trend filter
        above_1d_ema50 = close[i] > ema50_1d_aligned[i]
        below_1d_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price moves below lower Donchian band or trend turns bearish
            if close[i] < lower[i] or below_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above upper Donchian band or trend turns bullish
            if close[i] > upper[i] or above_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Donchian breakout up with volume confirmation and bullish trend
            if breakout_up and vol_confirmed and above_1d_ema50:
                position = 1
                signals[i] = 0.25
            # Short: Donchian breakout down with volume confirmation and bearish trend
            elif breakout_down and vol_confirmed and below_1d_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 20:44
