# Strategy: 6h_Camarilla_R2_S2_Breakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.117 | +17.4% | -9.4% | 199 | FAIL |
| ETHUSDT | 0.213 | +29.2% | -7.9% | 174 | PASS |
| SOLUSDT | 0.503 | +57.9% | -12.0% | 172 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.853 | +16.0% | -4.9% | 66 | PASS |
| SOLUSDT | -1.017 | -5.1% | -11.7% | 68 | FAIL |

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
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Use previous day's pivots (avoid look-ahead)
    r1_1d_prev = np.roll(r1_1d, 1)
    s1_1d_prev = np.roll(s1_1d, 1)
    r2_1d_prev = np.roll(r2_1d, 1)
    s2_1d_prev = np.roll(s2_1d, 1)
    r1_1d_prev[0] = np.nan
    s1_1d_prev[0] = np.nan
    r2_1d_prev[0] = np.nan
    s2_1d_prev[0] = np.nan
    
    # Align daily pivot levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d_prev)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d_prev)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_1d_prev)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_1d_prev)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need volume MA20, ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or
            np.isnan(s2_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume and volatility (breakout continuation)
            if (close[i] > r2_6h[i] and volume_filter and volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume and volatility (breakout continuation)
            elif (close[i] < s2_6h[i] and volume_filter and volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or volatility drops
            if close[i] < r1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or volatility drops
            if close[i] > s1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R2_S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 09:37
