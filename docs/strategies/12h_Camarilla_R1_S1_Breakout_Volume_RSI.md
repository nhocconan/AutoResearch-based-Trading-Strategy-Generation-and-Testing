# Strategy: 12h_Camarilla_R1_S1_Breakout_Volume_RSI

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.702 | -3.5% | -18.7% | 177 | FAIL |
| ETHUSDT | 0.098 | +24.5% | -17.2% | 159 | PASS |
| SOLUSDT | 0.223 | +34.0% | -26.7% | 148 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.015 | +5.8% | -5.9% | 55 | PASS |
| SOLUSDT | -0.500 | -0.9% | -13.0% | 50 | FAIL |

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
    
    # Get daily data for pivot points (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Use previous day's pivots (avoid look-ahead)
    r1_1d_prev = np.roll(r1_1d, 1)
    s1_1d_prev = np.roll(s1_1d, 1)
    r1_1d_prev[0] = np.nan
    s1_1d_prev[0] = np.nan
    
    # Align daily pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d_prev)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d_prev)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need RSI (14+20), volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(r1_12h[i]) or 
            np.isnan(s1_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above R1 with volume and RSI > 50
            if (close[i] > r1_12h[i] and volume_filter and rsi[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and RSI < 50
            elif (close[i] < s1_12h[i] and volume_filter and rsi[i] < 50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot level
            if close[i] < pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot level
            if close[i] > pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_RSI"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-17 09:43
