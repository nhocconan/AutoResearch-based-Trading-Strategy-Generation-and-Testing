# Strategy: 4h_Camarilla_R1S1_Breakout_VolumeSpike_12hEMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.357 | +37.8% | -8.3% | 204 | PASS |
| ETHUSDT | 0.015 | +19.5% | -15.5% | 197 | PASS |
| SOLUSDT | 0.390 | +52.2% | -29.2% | 167 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.928 | -3.1% | -7.7% | 77 | FAIL |
| ETHUSDT | 0.500 | +13.8% | -8.9% | 62 | PASS |
| SOLUSDT | 0.320 | +10.5% | -12.1% | 53 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeSpike_12hEMA34
Hypothesis: Camarilla pivot levels (R1, S1) from 12h chart act as strong support/resistance.
Breakouts beyond these levels with volume confirmation and 12h EMA34 trend filter capture momentum.
Designed for 20-50 trades/year on 4h timeframe with low trade frequency to minimize fee drag.
Works in bull/bear markets by requiring volume spike and 12h EMA34 trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h pivot points using standard formula
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # Using previous 12h bar's data to avoid look-ahead
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    close_12h = df_12h['close']
    
    pivot = (high_12h + low_12h + close_12h) / 3
    r1 = 2 * pivot - low_12h
    s1 = 2 * pivot - high_12h
    
    # Shift by 1 to use previous 12h bar's levels only
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_prev)
    
    # Get 12h data for EMA trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and price above 12h EMA (uptrend)
            if price > r1_val and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and price below 12h EMA (downtrend)
            elif price < s1_val and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or breaks below 12h EMA
            if price <= s1_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or breaks above 12h EMA
            if price >= r1_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeSpike_12hEMA34"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 01:38
