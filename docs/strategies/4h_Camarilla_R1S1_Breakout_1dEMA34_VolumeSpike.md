# Strategy: 4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.285 | +33.7% | -8.5% | 240 | KEEP |
| ETHUSDT | 0.013 | +19.4% | -14.4% | 229 | KEEP |
| SOLUSDT | 0.627 | +82.6% | -18.6% | 210 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.261 | -5.7% | -8.4% | 94 | DISCARD |
| ETHUSDT | 0.215 | +8.7% | -10.9% | 80 | KEEP |
| SOLUSDT | 0.395 | +11.8% | -8.6% | 70 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
- Long: Close breaks above Camarilla R1 + price > 1d EMA34 (bullish trend) + volume > 1.8x 20-period avg
- Short: Close breaks below Camarilla S1 + price < 1d EMA34 (bearish trend) + volume > 1.8x 20-period avg
- Exit: Close crosses Camarilla H4/L4 levels (mean reversion at core pivot)
- Uses Camarilla pivot levels from daily HTF for structure, 1d EMA34 for trend filter, and volume confirmation
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with trend) and bear markets (mean reversion at core pivots)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from 1d HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_range = high_1d - low_1d
    camarilla_h4 = close_1d + camarilla_range * 1.1 / 4
    camarilla_l4 = close_1d - camarilla_range * 1.1 / 4
    camarilla_h3 = close_1d + camarilla_range * 1.1 / 6
    camarilla_l3 = close_1d - camarilla_range * 1.1 / 6
    camarilla_h6 = close_1d + camarilla_range * 1.1
    camarilla_l6 = close_1d - camarilla_range * 1.1
    camarilla_r1 = camarilla_h3  # R1 = H3
    camarilla_s1 = camarilla_l3  # S1 = L3
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h6)
    camarilla_l6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h6_aligned[i]) or
            np.isnan(camarilla_l6_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Camarilla R1 + bullish trend + volume confirmation
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S1 + bearish trend + volume confirmation
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below Camarilla L4 (mean reversion)
            if close[i] < camarilla_l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above Camarilla H4 (mean reversion)
            if close[i] > camarilla_h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-05-07 12:28
