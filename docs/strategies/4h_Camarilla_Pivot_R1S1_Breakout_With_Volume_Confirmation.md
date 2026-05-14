# Strategy: 4h_Camarilla_Pivot_R1S1_Breakout_With_Volume_Confirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.184 | +28.3% | -8.8% | 505 | PASS |
| ETHUSDT | 0.092 | +24.2% | -9.8% | 487 | PASS |
| SOLUSDT | 0.646 | +78.9% | -15.1% | 432 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.467 | -6.3% | -9.1% | 178 | FAIL |
| ETHUSDT | 0.357 | +10.8% | -7.1% | 171 | PASS |
| SOLUSDT | 0.441 | +12.2% | -10.4% | 158 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_With_Volume_Confirmation
Hypothesis: Camarilla pivot levels (R1, S1) from the 12-hour timeframe act as key support/resistance. 
Breakout above R1 with volume > 1.5x 20-period average and price > 12h EMA34 = long; 
breakdown below S1 with volume confirmation and price < 12h EMA34 = short. 
Designed for 4-hour timeframe with ~20-40 trades/year to minimize fee drag and work in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour typical price for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12-hour OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla pivot levels for 12h timeframe
    # P = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    r1_12h = close_12h + (high_12h - low_12h) * 1.1 / 12
    s1_12h = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    # Align 12h levels to 4h timeframe (wait for 12h bar close)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    
    # 12-hour EMA trend filter (34-period)
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_12h_aligned[i]
        s1 = s1_12h_aligned[i]
        ema_trend = ema_12h_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above 12h R1 with volume in uptrend
            if price > r1 and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h S1 with volume in downtrend
            elif price < s1 and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns below 12h pivot or trend reverses
            if price < pivot_12h_aligned[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns above 12h pivot or trend reverses
            if price > pivot_12h_aligned[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 04:36
