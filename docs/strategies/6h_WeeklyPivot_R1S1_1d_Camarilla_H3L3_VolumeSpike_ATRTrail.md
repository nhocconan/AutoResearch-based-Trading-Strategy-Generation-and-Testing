# Strategy: 6h_WeeklyPivot_R1S1_1d_Camarilla_H3L3_VolumeSpike_ATRTrail

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.155 | +26.9% | -15.4% | 94 | PASS |
| ETHUSDT | 0.088 | +24.0% | -18.7% | 90 | PASS |
| SOLUSDT | 0.999 | +138.0% | -15.1% | 91 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.280 | -2.4% | -6.9% | 34 | FAIL |
| ETHUSDT | 0.135 | +7.3% | -7.7% | 27 | PASS |
| SOLUSDT | 0.173 | +7.9% | -7.8% | 29 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using weekly pivot direction (R1/S1) for trend filter, 
1d Camarilla H3/L3 breakout with volume confirmation.
- Weekly pivot bias: long when price above weekly R1, short when below weekly S1
- Breakout triggers when price closes beyond 1d H3 (long) or L3 (short) with volume > 1.8x 20-period 6h MA
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.0x ATR) to lock in profits
- Designed for very low trade frequency (target: 50-150 trades over 4 years) to avoid fee drag
- Works in bull markets (buying H3 breakouts above weekly R1) and bear markets (selling L3 breakdowns below weekly S1)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (based on previous week)
    # P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]
    
    weekly_p = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    weekly_r1 = 2 * weekly_p - prev_low_1w
    weekly_s1 = 2 * weekly_p - prev_high_1w
    
    # Get 1d data for Camarilla pivots (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    camarilla_h3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Get 6h data for volume confirmation and ATR (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Volume average (20-period) on 6h
    volume_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) on 6h for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_10)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and weekly pivot filter
            # Long: price closes above H3 + volume spike + price above weekly R1
            if price > h3_val and vol > 1.8 * vol_ma and price > weekly_r1_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price closes below L3 + volume spike + price below weekly S1
            elif price < l3_val and vol > 1.8 * vol_ma and price < weekly_s1_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.5 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
    
    return signals

name = "6h_WeeklyPivot_R1S1_1d_Camarilla_H3L3_VolumeSpike_ATRTrail"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 21:30
