# Strategy: 12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.136 | +16.7% | -6.6% | 105 | DISCARD |
| ETHUSDT | 0.343 | +36.0% | -7.4% | 85 | KEEP |
| SOLUSDT | 0.118 | +24.9% | -23.3% | 88 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.011 | +5.7% | -6.9% | 38 | KEEP |
| SOLUSDT | -0.578 | -1.1% | -7.8% | 38 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeConfirm
Hypothesis: Trade 12h Camarilla H3/L3 breakouts with 1d EMA34 trend filter and volume confirmation (>1.8x 20-bar MA). 
12h timeframe reduces trade frequency to avoid fee drag. Camarilla H3/L3 provides institutional support/resistance levels. 
1d EMA34 filter ensures trading with higher timeframe trend. Volume confirmation adds conviction. 
Discrete sizing 0.25 balances profit and fee drag. Target: 12-25 trades/year (~50-100 over 4 years) to stay within fee drag limits.
Works in bull/bear: trend filter adapts to market direction, volume confirms breakout validity.
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
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar's OHLC
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high_1d - prev_low_1d
    h3 = prev_close_1d + 1.1 * camarilla_range / 6   # H3 level
    l3 = prev_close_1d - 1.1 * camarilla_range / 6   # L3 level
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND 1d trend bullish (close > EMA34) AND volume confirm
            long_setup = (close[i] > h3_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_confirm[i]
            # Short: price breaks below L3 AND 1d trend bearish (close < EMA34) AND volume confirm
            short_setup = (close[i] < l3_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_confirm[i]
            
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
            # Exit: price re-enters Camarilla H3/L3 range OR 1d trend turns bearish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR 1d trend turns bullish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-25 14:08
