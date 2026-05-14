# Strategy: 4h_Pivot_Volume_EMA34_MidExit

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.172 | +27.1% | -7.7% | 381 | PASS |
| ETHUSDT | 0.329 | +35.0% | -7.3% | 342 | PASS |
| SOLUSDT | 0.517 | +57.3% | -20.2% | 312 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.555 | -5.4% | -5.8% | 137 | FAIL |
| ETHUSDT | 0.517 | +12.2% | -8.8% | 127 | PASS |
| SOLUSDT | 1.142 | +20.7% | -5.5% | 102 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: The 4-hour price often tests daily pivot levels before reversing or breaking out. 
By combining the daily pivot point with a 4-hour EMA34 trend filter and volume confirmation, 
we aim to capture both breakout and mean-reversion opportunities. The strategy enters long 
when price crosses above the daily pivot with volume > 1.8x average and price above EMA34, 
and short when price crosses below the daily pivot with volume > 1.8x average and price 
below EMA34. Exits occur when price returns to the midpoint between pivot and the prior 
day's high/low, reducing exposure in ranging markets. Designed for 4h timeframe to work in 
bull (breakouts) and bear (mean reversion to pivot) regimes with ~20-30 trades per year.
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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot and support/resistance levels
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    
    # Define exit levels: midpoint between pivot and prior day's high/low
    upper_exit = (pivot + phigh) / 2
    lower_exit = (pivot + plow) / 2
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all daily levels to 4h timeframe (waits for daily bar to close)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    upper_exit_4h = align_htf_to_ltf(prices, df_1d, upper_exit)
    lower_exit_4h = align_htf_to_ltf(prices, df_1d, lower_exit)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_4h[i]) or np.isnan(upper_exit_4h[i]) or np.isnan(lower_exit_4h[i]) or
            np.isnan(ema_34_4h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price crosses above pivot with volume spike and above daily EMA34
            if price > pivot_4h[i] and vol > 1.8 * vol_ma and price > ema_34_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below pivot with volume spike and below daily EMA34
            elif price < pivot_4h[i] and vol > 1.8 * vol_ma and price < ema_34_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to upper exit level (midpoint between pivot and prior high)
            if price < upper_exit_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to lower exit level (midpoint between pivot and prior low)
            if price > lower_exit_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_Volume_EMA34_MidExit"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 22:21
