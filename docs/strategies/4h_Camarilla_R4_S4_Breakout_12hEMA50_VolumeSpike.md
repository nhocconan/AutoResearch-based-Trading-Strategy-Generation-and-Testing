# Strategy: 4h_Camarilla_R4_S4_Breakout_12hEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.084 | +23.8% | -6.8% | 177 | PASS |
| ETHUSDT | 0.450 | +41.6% | -10.0% | 163 | PASS |
| SOLUSDT | 0.725 | +80.2% | -20.9% | 133 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.667 | -4.6% | -5.5% | 68 | FAIL |
| ETHUSDT | 0.878 | +16.4% | -6.8% | 58 | PASS |
| SOLUSDT | -0.022 | +5.5% | -8.2% | 50 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R4_S4_Breakout_12hEMA50_VolumeSpike
Hypothesis: Uses stronger Camarilla R4/S4 levels (1.5x range) with 12h EMA50 trend filter and volume spike (2x 24-bar avg) to capture high-probability breakouts. R4/S4 are more extreme, reducing false signals. Works in both bull and bear by following trend direction. Targets 20-30 trades/year via strict conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels (R4/S4: 1.5x range)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    R4 = typical_price + (range_ * 1.5 / 2)
    S4 = typical_price - (range_ * 1.5 / 2)
    
    # Align Camarilla levels to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4.values)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4.values)
    
    # Volume confirmation: >2x 24-period MA (4 days of 4h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_24[i])
        
        # Breakout conditions at R4/S4
        long_breakout = close[i] > R4_aligned[i] and vol_confirm and uptrend
        short_breakout = close[i] < S4_aligned[i] and vol_confirm and downtrend
        
        # Exit conditions: return to midpoint of R4/S4
        midpoint = (R4_aligned[i] + S4_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R4_S4_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 03:35
