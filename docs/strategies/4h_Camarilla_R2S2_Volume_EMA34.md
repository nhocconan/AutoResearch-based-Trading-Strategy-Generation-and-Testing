# Strategy: 4h_Camarilla_R2S2_Volume_EMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.546 | +42.3% | -5.7% | 236 | PASS |
| ETHUSDT | 0.405 | +40.2% | -9.2% | 225 | PASS |
| SOLUSDT | 0.768 | +87.2% | -15.6% | 186 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.643 | -6.1% | -7.6% | 96 | FAIL |
| ETHUSDT | 0.784 | +16.2% | -5.7% | 81 | PASS |
| SOLUSDT | 0.510 | +12.3% | -6.3% | 66 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: The 4-hour price often respects key intraday support/resistance derived from 
the prior day's range (Camarilla levels). By combining daily Camarilla S2/R2 levels with 
4-hour EMA34 trend filter and volume spikes, we create high-probability breakout trades. 
The strategy targets ~25 trades/year by requiring confluence: price breaks S2/R2, volume 
> 2x 20-bar average, and price on correct side of EMA34. Exits occur when price returns to 
the daily pivot, limiting adverse exposure in ranging markets. Designed for 4h timeframe 
to work in both bull (breakouts continuation) and bear (mean reversion to pivot) regimes.
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
    
    # Get daily data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    
    # Camarilla S2/R2 levels (used for entry)
    R2 = pivot + (range_ * 1.1 / 6)
    S2 = pivot - (range_ * 1.1 / 6)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all daily levels to 4h timeframe (waits for daily bar to close)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(R2_4h[i]) or np.isnan(S2_4h[i]) or np.isnan(pivot_4h[i]) or
            np.isnan(ema_34_4h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: break above R2 with volume spike and above daily EMA34
            if price > R2_4h[i] and vol > 2.0 * vol_ma and price > ema_34_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with volume spike and below daily EMA34
            elif price < S2_4h[i] and vol > 2.0 * vol_ma and price < ema_34_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot
            if price < pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot
            if price > pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R2S2_Volume_EMA34"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 22:18
