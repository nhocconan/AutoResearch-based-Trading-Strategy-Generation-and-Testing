# Strategy: 4H_Donchian_Breakout_1D_Volume_12hEMA50

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.356 | +35.4% | -11.6% | 25 | PASS |
| ETHUSDT | 0.237 | +31.5% | -9.6% | 25 | PASS |
| SOLUSDT | 1.372 | +224.9% | -19.7% | 27 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.137 | -0.6% | -5.5% | 8 | FAIL |
| ETHUSDT | 0.178 | +7.8% | -7.3% | 8 | PASS |
| SOLUSDT | -0.373 | +1.5% | -7.6% | 7 | FAIL |

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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['close'].values  # Use close for lower band to avoid whipsaw
    
    # Calculate 20-day Donchian upper band (breakout trigger)
    donch_upper = np.full(len(high_1d), np.nan)
    for i in range(19, len(high_1d)):
        donch_upper[i] = np.max(high_1d[i-19:i+1])
    
    # Calculate 20-day Donchian lower band (exit trigger)
    donch_lower = np.full(len(low_1d), np.nan)
    for i in range(19, len(low_1d)):
        donch_lower[i] = np.min(low_1d[i-19:i+1])
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 4h timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    size = 0.25   # 25% position size (long-only to avoid short whipsaw in 2022)
    
    # Warmup: need Donchian, EMA, and volume MA
    start_idx = max(20, ema_period, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Break above 1d Donchian upper + volume spike + above 12h EMA50
            if (price > donch_upper_aligned[i] and 
                vol_ratio > 1.5 and 
                price > ema_12h_aligned[i]):
                signals[i] = size
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below 1d Donchian lower OR loses trend
            if (price < donch_lower_aligned[i] or 
                price < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
    
    return signals

name = "4H_Donchian_Breakout_1D_Volume_12hEMA50"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 11:37
