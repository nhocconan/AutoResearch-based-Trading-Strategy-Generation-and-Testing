# Strategy: 4h_Camarilla_H3L3_Breakout_1dTrendFilter_VolumeConfirm_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.411 | +38.8% | -7.7% | 198 | PASS |
| ETHUSDT | 0.097 | +24.4% | -13.1% | 189 | PASS |
| SOLUSDT | 0.704 | +88.6% | -18.7% | 169 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.855 | -1.5% | -7.4% | 81 | FAIL |
| ETHUSDT | 0.810 | +18.2% | -10.8% | 62 | PASS |
| SOLUSDT | -0.230 | +2.1% | -9.2% | 55 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dTrendFilter_VolumeConfirm_v2
Hypothesis: Trade Camarilla H3/L3 breakouts on 4h with 1d EMA34 trend filter and volume confirmation.
Uses daily trend to capture major market direction, reducing false breakouts in choppy markets.
Discrete sizing (0.25) limits fee drag. Designed to work in both bull and bear markets by aligning with 1d trend.
Target: 20-50 trades/year per symbol.
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
    
    # Get 1d data for HTF trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    camarilla_h3 = close_1d_vals + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d_vals - 1.1 * (high_1d - low_1d) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above H3 + 1d uptrend + volume confirmation
            long_setup = (close[i] > camarilla_h3_aligned[i]) and htf_1d_bullish and volume_confirm[i]
            
            # Short setup: price breaks below L3 + 1d downtrend + volume confirmation
            short_setup = (close[i] < camarilla_l3_aligned[i]) and htf_1d_bearish and volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches L3 (stop) OR 1d trend turns bearish
            if (close[i] <= camarilla_l3_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches H3 (stop) OR 1d trend turns bullish
            if (close[i] >= camarilla_h3_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dTrendFilter_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 15:31
