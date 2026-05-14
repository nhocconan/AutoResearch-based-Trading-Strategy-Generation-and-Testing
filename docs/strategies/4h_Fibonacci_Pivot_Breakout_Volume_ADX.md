# Strategy: 4h_Fibonacci_Pivot_Breakout_Volume_ADX

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.588 | +51.8% | -8.2% | 162 | PASS |
| ETHUSDT | 0.586 | +57.2% | -8.4% | 150 | PASS |
| SOLUSDT | 0.632 | +76.9% | -18.5% | 115 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.172 | -4.5% | -9.2% | 58 | FAIL |
| ETHUSDT | 0.373 | +11.0% | -8.1% | 51 | PASS |
| SOLUSDT | -0.572 | -2.8% | -18.4% | 40 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h Fibonacci Pivot Breakout with Volume Spike and Trend Filter
Hypothesis: Price breaking above/below Fibonacci pivot levels (R1/S1) on 4h with volume confirmation
(volume > 2x average) and trend strength (ADX > 25) indicates strong momentum.
Fibonacci pivots derived from 1d OHLC provide institutional support/resistance.
Target: 20-30 trades/year to minimize fee drain and work in both bull and bear markets.
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
    
    # Get 1d data for Fibonacci pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Fibonacci pivot levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Fibonacci pivot formulas: R1 = C + (H-L)*0.382, S1 = C - (H-L)*0.382
    fib_r1 = close_1d + (high_1d - low_1d) * 0.382
    fib_s1 = close_1d - (high_1d - low_1d) * 0.382
    
    # Align to 4h timeframe with proper delay (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, fib_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, fib_s1)
    
    # EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ADX for trend strength (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 2x 20-period EMA (stricter for fewer trades)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators (max of 20,20,14)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_val = ema20[i]
        adx_val = adx[i]
        vol_conf = vol_ratio[i] > 2.0  # Stricter volume filter
        
        if position == 0:
            # Strong trend (ADX > 25) and volume confirmation
            # Price breaks above R1 = long
            if adx_val > 25 and price > r1 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Price breaks below S1 = short
            elif adx_val > 25 and price < s1 and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if trend weakens or price returns below EMA20
            if adx_val < 20 or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if trend weakens or price returns above EMA20
            if adx_val < 20 or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Fibonacci_Pivot_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 05:47
