# Strategy: 4h_Camarilla_R1S1_R2S2_Breakout_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.401 | +39.4% | -8.8% | 210 | PASS |
| ETHUSDT | 0.000 | +18.8% | -11.7% | 205 | PASS |
| SOLUSDT | 0.978 | +139.2% | -21.6% | 171 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.606 | -9.1% | -9.3% | 85 | FAIL |
| ETHUSDT | 0.978 | +21.9% | -8.2% | 71 | PASS |
| SOLUSDT | -0.072 | +4.2% | -8.5% | 57 | FAIL |

## Code
```python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_R2S2_Breakout_Volume_Trend
Hypothesis: Price breaks above/below Camarilla pivot levels (R1/S1 or R2/S2) with volume spike and daily EMA34 trend filter on 4h timeframe.
Uses 1d EMA34 for trend direction to filter breakouts in both bull/bear markets.
Target: 20-40 trades/year to minimize fee drift while capturing strong directional moves with proper risk control.
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
    
    # Daily EMA34 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily pivot points from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    s2 = pivot - (range_hl * 1.1 / 6)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 or R2 with volume spike and uptrend (price > daily EMA34)
            if ((price > r1 or price > r2) and vol_spike and price > ema34):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 or S2 with volume spike and downtrend (price < daily EMA34)
            elif ((price < s1 or price < s2) and vol_spike and price < ema34):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below daily EMA34 OR breaks below S1 (reversal)
            if price < ema34 or price < s1:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above daily EMA34 OR breaks above R1 (reversal)
            if price > ema34 or price > r1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_R2S2_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 03:35
