# Strategy: 12h_Camarilla_R1S1_VolumeSpike_ChopRegime

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.346 | +7.9% | -18.2% | 196 | FAIL |
| ETHUSDT | 0.284 | +34.1% | -18.8% | 179 | PASS |
| SOLUSDT | -0.263 | -0.5% | -35.3% | 171 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.351 | +10.4% | -9.2% | 68 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla Pivot R1/S1 Breakout + Volume Spike + Chop Regime Filter.
Long when price breaks above R1 with volume > 1.5x average and choppy market (CHOP > 61.8).
Short when price breaks below S1 with volume > 1.5x average and choppy market (CHOP > 61.8).
Exit when price reverts to pivot point (PP) or chop regime ends (CHOP < 38.2).
Uses 1d for Camarilla pivot calculation, 12h for price/volume, 1d for chop filter.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Camarilla pivots and chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1, PP)
    def calculate_camarilla(high, low, close):
        pp = (high + low + close) / 3.0
        r1 = close + (high - low) * 1.1 / 12.0
        s1 = close - (high - low) * 1.1 / 12.0
        return pp, r1, s1
    
    pp_1d = np.zeros_like(close_1d)
    r1_1d = np.zeros_like(close_1d)
    s1_1d = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        pp, r1, s1 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        pp_1d[i] = pp
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Calculate 1d Choppiness Index (CHOP)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's ATR
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Max true range over period
        max_tr = np.zeros_like(close)
        for i in range(period, len(close)):
            max_tr[i] = np.max(tr[i-period+1:i+1])
        
        # Chop formula: 100 * log10(atr_sum / max_tr) / log10(period)
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if max_tr[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / max_tr[i]) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop_1d_aligned[i]
        pp = pp_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        
        # Chop regime: CHOP > 61.8 = ranging (good for mean reversion at pivots)
        is_choppy = chop_val > 61.8
        # Exit chop regime: CHOP < 38.2 = trending (avoid false signals)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long: price breaks above R1 with volume spike in choppy market
            if price > r1 and vol_spike and is_choppy:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike in choppy market
            elif price < s1 and vol_spike and is_choppy:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point OR chop regime ends (trending)
            if price <= pp or is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point OR chop regime ends (trending)
            if price >= pp or is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-17 19:14
