# Strategy: 4h_camarilla_volume_breakout_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.226 | +31.8% | -10.3% | 236 | PASS |
| ETHUSDT | 0.289 | +38.0% | -16.7% | 220 | PASS |
| SOLUSDT | 0.487 | +67.7% | -27.4% | 199 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.302 | -7.9% | -12.2% | 86 | FAIL |
| ETHUSDT | 0.131 | +7.4% | -15.3% | 79 | PASS |
| SOLUSDT | -0.057 | +4.0% | -14.2% | 63 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_camarilla_volume_breakout_v3
# Hypothesis: Camarilla pivot levels on 1d with volume spike and ADX trend filter capture breakouts in both bull and bear markets.
# Uses L3/L3 and H3/H3 levels for entry, volume > 2x 20-period average, and ADX > 20 to avoid chop.
# Target: 20-40 trades/year (80-160 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_volume_breakout_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ADX(14) for trend strength
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    for i in range(14, n):
        plus_dm_sum = np.sum(plus_dm[i-13:i+1])
        minus_dm_sum = np.sum(minus_dm[i-13:i+1])
        tr_sum = np.sum(tr[i-13:i+1])
        if tr_sum > 0:
            plus_di[i] = 100 * plus_dm_sum / tr_sum
            minus_di[i] = 100 * minus_dm_sum / tr_sum
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros(n)
    adx[13] = dx[13]
    for i in range(14, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Get daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = np.zeros(len(df_1d))
    camarilla_l3 = np.zeros(len(df_1d))
    camarilla_h4 = np.zeros(len(df_1d))
    camarilla_l4 = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        range_ = high_1d[i-1] - low_1d[i-1]
        close_prev = close_1d[i-1]
        camarilla_h4[i] = close_prev + range_ * 1.1 / 2
        camarilla_l4[i] = close_prev - range_ * 1.1 / 2
        camarilla_h3[i] = close_prev + range_ * 1.1 / 4
        camarilla_l3[i] = close_prev - range_ * 1.1 / 4
    
    # Align to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(adx[i]) or np.isnan(vol_ma_20[i]) or np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma_20[i] * 2.0
        
        # ADX trend filter
        trending = adx[i] > 20
        
        if position == 1:  # Long position
            # Exit: price below L3 or ADX drops
            if close[i] < l3_4h[i] or adx[i] < 15:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above H3 or ADX drops
            if close[i] > h3_4h[i] or adx[i] < 15:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volume and trend
            if close[i] > h3_4h[i] and vol_spike and trending:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume and trend
            elif close[i] < l3_4h[i] and vol_spike and trending:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 06:35
