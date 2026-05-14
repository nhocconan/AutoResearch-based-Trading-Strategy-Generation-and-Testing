# Strategy: 4h_Camarilla_R1S1_Breakout_Volume_Spike_ADXFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.133 | +25.9% | -11.9% | 279 | PASS |
| ETHUSDT | 0.045 | +21.9% | -8.3% | 256 | PASS |
| SOLUSDT | 0.119 | +25.3% | -19.7% | 205 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.388 | +2.9% | -6.9% | 108 | FAIL |
| ETHUSDT | 0.986 | +18.5% | -5.6% | 94 | PASS |
| SOLUSDT | 0.344 | +10.1% | -11.5% | 81 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Spike_ADXFilter
Hypothesis: Breakout above/below Camarilla R1/S1 with volume spike and ADX>25 confirms strong momentum. 
Exit when price crosses back below/above S1/R1 or ADX weakens (<20). Designed for low trade frequency 
to avoid fee drag while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily high, low, close for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ADX(14) trend strength filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - low[:-1]), np.absolute(low[1:] - high[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20, 14*2)  # Need warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        adx_val = adx[i]
        
        if position == 0:
            # Long: price > Camarilla R1 with volume spike and strong trend (ADX>25)
            if price > r1 and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 with volume spike and strong trend (ADX>25)
            elif price < s1 and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < Camarilla S1 OR ADX weakens (<20)
            if price < s1 or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > Camarilla R1 OR ADX weakens (<20)
            if price > r1 or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Spike_ADXFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 02:46
