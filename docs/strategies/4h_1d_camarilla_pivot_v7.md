# Strategy: 4h_1d_camarilla_pivot_v7

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.251 | +28.2% | -7.6% | 314 | PASS |
| ETHUSDT | -0.004 | +21.1% | -6.3% | 287 | FAIL |
| SOLUSDT | 0.452 | +45.6% | -7.9% | 231 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.639 | -3.2% | -4.5% | 123 | FAIL |
| SOLUSDT | 0.738 | +13.9% | -3.8% | 95 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_1d_camarilla_pivot_v7
Hypothesis: 4-hour strategy using daily context with Camarilla pivot levels and volume confirmation.
Long when price crosses above daily Pivot with volume > 2.0x average and price > daily EMA200 (bullish trend).
Short when price crosses below daily Pivot with volume > 2.0x average and price < daily EMA200 (bearish trend).
Exit when price crosses opposite daily support/resistance or volume drops below 1.5x average.
Uses discrete position sizing (0.30) to balance return and risk. Target: 25-40 trades/year.
Fixed: Tightened volume confirmation and exit conditions to reduce trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_v7"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    if len(high) < 1:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    # Standard Camarilla multipliers
    H3 = pivot + (range_val * 1.1 / 4)
    L3 = pivot - (range_val * 1.1 / 4)
    H4 = pivot + (range_val * 1.1 / 2)
    L4 = pivot - (range_val * 1.1 / 2)
    
    return H3, L3, H4, L4

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context and Pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily Pivot (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # Daily support/resistance levels (Corrected: Standard Camarilla S1/R1)
    S1_1d = pivot_1d - (range_1d * 1.1 / 12)  # Daily S1
    R1_1d = pivot_1d + (range_1d * 1.1 / 12)  # Daily R1
    
    # Calculate daily EMA for trend filter
    ema_200_1d = calculate_ema(close_1d, 200)
    
    # Align indicators to 4-hour timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        pivot = pivot_1d_aligned[i]
        S1 = S1_1d_aligned[i]
        R1 = R1_1d_aligned[i]
        trend_up_1d = price > ema_200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below daily S1 or volume drops below 1.5x average
            if price < S1 or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short
            # Exit: price crosses above daily R1 or volume drops below 1.5x average
            if price > R1 or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long: price crosses above daily Pivot with volume expansion and uptrend on daily
            if price > pivot and vol_ratio > 2.0 and trend_up_1d:
                position = 1
                signals[i] = 0.30
            # Enter short: price crosses below daily Pivot with volume expansion and downtrend on daily
            elif price < pivot and vol_ratio > 2.0 and not trend_up_1d:
                position = -1
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-04-08 21:19
