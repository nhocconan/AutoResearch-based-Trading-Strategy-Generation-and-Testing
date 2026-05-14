# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.525 | +58.2% | -12.7% | 97 | PASS |
| ETHUSDT | 0.170 | +29.3% | -18.7% | 106 | PASS |
| SOLUSDT | 0.838 | +169.0% | -31.3% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.384 | +0.3% | -7.6% | 45 | FAIL |
| ETHUSDT | 0.805 | +23.8% | -9.9% | 33 | PASS |
| SOLUSDT | 0.760 | +24.6% | -10.5% | 32 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume_Trend
Hypothesis: Camarilla pivot levels (R1/S1) from daily chart + EMA34 trend filter + volume spike.
Long when price breaks above R1 in uptrend with volume confirmation; short when breaks below S1 in downtrend.
Exit when price crosses EMA34 (trend reversal) to avoid whipsows.
Designed for range/breakout markets in both bull and bear regimes.
Target: 20-40 trades/year to minimize fee drag.
"""

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
    
    # Get daily data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):  # Start from 1 to use previous day
        rng = high_1d[i-1] - low_1d[i-1]
        camarilla_r1[i] = close_1d[i-1] + (1.1 * rng / 12)
        camarilla_s1[i] = close_1d[i-1] - (1.1 * rng / 12)
    
    # Align Camarilla levels to 4h timeframe (previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate EMA(34) on daily close for trend filter
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align EMA to 4h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (20-period average)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(2, ema_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below daily EMA34
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume
            if uptrend and volume_confirmation and price > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume
            elif downtrend and volume_confirmation and price < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below EMA34 (trend reversal)
            if price < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above EMA34 (trend reversal)
            if price > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume_Trend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 14:55
