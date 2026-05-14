# Strategy: 4h_Pivot_R2_S2_Breakout_Vol

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.416 | +40.2% | -14.9% | 203 | PASS |
| ETHUSDT | 0.167 | +28.3% | -10.3% | 205 | PASS |
| SOLUSDT | 0.334 | +45.8% | -24.3% | 179 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.073 | -10.3% | -10.8% | 75 | FAIL |
| ETHUSDT | 0.704 | +16.2% | -10.8% | 78 | PASS |
| SOLUSDT | -0.613 | -2.9% | -11.6% | 59 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Shift to use previous day's pivots (avoid look-ahead)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r2_prev = np.roll(r2, 1)
    s2_prev = np.roll(s2, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    r2_prev[0] = np.nan
    s2_prev[0] = np.nan
    
    # Align daily pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_prev)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2_prev)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2_prev)
    
    # Volume confirmation: current volume > 1.5 * 8-period average (4h * 8 = 32h)
    volume_ma8 = pd.Series(volume).rolling(window=8, min_periods=8).mean().values
    
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
    
    start_idx = 20  # Need R2/S2 and ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma8[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r2_4h[i]) or 
            np.isnan(s2_4h[i]) or
            np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 8-period average
        volume_filter = volume[i] > (1.5 * volume_ma8[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume and volatility (strong breakout)
            if close[i] > r2_4h[i] and volume_filter and volatility_filter:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S2 with volume and volatility (strong breakdown)
            elif close[i] < s2_4h[i] and volume_filter and volatility_filter:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or volatility drops
            if close[i] < r1_4h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price returns above S1 or volatility drops
            if close[i] > s1_4h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Pivot_R2_S2_Breakout_Vol"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 09:03
