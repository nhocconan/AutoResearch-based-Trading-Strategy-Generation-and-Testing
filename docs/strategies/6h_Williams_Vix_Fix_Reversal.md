# Strategy: 6h_Williams_Vix_Fix_Reversal

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.148 | +26.9% | -15.6% | 83 | PASS |
| ETHUSDT | 0.139 | +26.9% | -16.5% | 98 | PASS |
| SOLUSDT | 1.198 | +205.1% | -24.1% | 87 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.783 | +0.3% | -5.6% | 28 | FAIL |
| ETHUSDT | 0.217 | +8.5% | -8.9% | 25 | PASS |
| SOLUSDT | -0.241 | +2.0% | -8.9% | 28 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Williams_Vix_Fix_Reversal
Mean reversion strategy using Williams Vix Fix (volatility-based oversold/overbought) on 6h timeframe.
Long when VIX Fix < 20 (oversold) and price > 6h EMA200 (uptrend filter).
Short when VIX Fix > 80 (overbought) and price < 6h EMA200 (downtrend filter).
Exit when VIX Fix crosses back to neutral (40-60 range) or trend filter fails.
Uses 1d trend filter (EMA50) to avoid counter-trend trades in strong trends.
Target: 15-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Vix Fix parameters
    vix_period = 22
    vix_ma_period = 10
    
    # Calculate highest close over vix_period
    highest_close = np.full(n, np.nan)
    for i in range(vix_period - 1, n):
        highest_close[i] = np.max(close[i - vix_period + 1:i + 1])
    
    # Williams Vix Fix: measures volatility/spike in sell pressure
    vix_fix = np.full(n, np.nan)
    for i in range(vix_period - 1, n):
        if highest_close[i] > 0 and not np.isnan(highest_close[i]):
            vix_fix[i] = ((highest_close[i] - low[i]) / highest_close[i]) * 100
        else:
            vix_fix[i] = 0
    
    # Smooth VIX Fix with moving average
    vix_fix_ma = np.full(n, np.nan)
    for i in range(vix_ma_period - 1, n):
        vix_fix_ma[i] = np.mean(vix_fix[i - vix_ma_period + 1:i + 1])
    
    # 6h EMA200 for trend filter
    ema_period = 200
    ema_6h = np.full(n, np.nan)
    if n >= ema_period:
        ema_6h[ema_period - 1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema_6h[i] = (close[i] * (2 / (ema_period + 1)) + 
                         ema_6h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_1d_period + 1))))
    
    # Align 1d EMA50 to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need VIX Fix MA, EMA200, and EMA1d
    start_idx = max(vix_ma_period - 1, ema_period - 1, ema_1d_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vix_fix_ma[i]) or np.isnan(ema_6h[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vix_value = vix_fix_ma[i]
        ema6h_val = ema_6h[i]
        ema1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Long: VIX Fix oversold (<20) and price above both EMAs (uptrend)
            if (vix_value < 20 and price > ema6h_val and price > ema1d_val):
                signals[i] = size
                position = 1
            # Short: VIX Fix overbought (>80) and price below both EMAs (downtrend)
            elif (vix_value > 80 and price < ema6h_val and price < ema1d_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: VIX Fix returns to neutral (40-60) or trend fails
            if (vix_value >= 40 and vix_value <= 60) or price < ema6h_val or price < ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: VIX Fix returns to neutral (40-60) or trend fails
            if (vix_value >= 40 and vix_value <= 60) or price > ema6h_val or price > ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Williams_Vix_Fix_Reversal"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 11:20
