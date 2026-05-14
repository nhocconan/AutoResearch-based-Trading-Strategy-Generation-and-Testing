# Strategy: 4h_DailyPivot_R1S1_Breakout_VolumeSpike_EMA34

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.093 | +24.2% | -9.5% | 163 | PASS |
| ETHUSDT | 0.070 | +22.5% | -12.8% | 153 | PASS |
| SOLUSDT | 0.847 | +129.0% | -24.3% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.029 | -4.8% | -7.9% | 59 | FAIL |
| ETHUSDT | 1.148 | +26.3% | -7.4% | 49 | PASS |
| SOLUSDT | 0.284 | +10.1% | -11.8% | 41 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Daily Pivot R1/S1 Breakout with Volume Spike and 1d EMA34 Trend Filter
Hypothesis: Daily pivot R1/S1 levels act as strong support/resistance. Breakouts with
volume confirmation and 1d EMA34 trend filter capture momentum moves in both bull
and bear markets. Designed for 20-50 trades/year on 4h timeframe.
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
    
    # Get daily data for pivot levels (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot = (df_d['high'] + df_d['low'] + df_d['close']) / 3.0
    r1 = 2 * pivot - df_d['low']
    s1 = 2 * pivot - df_d['high']
    
    # Shift to get previous day's levels (non-lookahead)
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1_prev)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_d, ema_34)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (4h ATR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema = ema_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and price above EMA34 (uptrend)
            if price > r1_level and volume_spike[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and price below EMA34 (downtrend)
            elif price < s1_level and volume_spike[i] and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or ATR trailing stop
            if price <= s1_level or price < (high[i] - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or ATR trailing stop
            if price >= r1_level or price > (low[i] + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_DailyPivot_R1S1_Breakout_VolumeSpike_EMA34"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 01:17
