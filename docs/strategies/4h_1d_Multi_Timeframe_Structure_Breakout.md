# Strategy: 4h_1d_Multi_Timeframe_Structure_Breakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.943 | +84.9% | -9.8% | 195 | PASS |
| ETHUSDT | 0.756 | +80.5% | -10.3% | 180 | PASS |
| SOLUSDT | 1.709 | +422.0% | -14.1% | 168 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.736 | +14.1% | -5.3% | 60 | PASS |
| ETHUSDT | 1.377 | +33.4% | -9.2% | 64 | PASS |
| SOLUSDT | 0.631 | +18.0% | -12.7% | 62 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_1d_Multi_Timeframe_Structure_Breakout
# Hypothesis: Combines 1d market structure (HH/HL/LH/LL) with 4h breakouts for trend-following entries.
# Uses 1d swing points to determine trend direction, and breaks of 4h swing highs/lows for entry timing.
# Volume confirmation (>1.5x 20-period average) filters for institutional participation.
# Designed for low trade frequency (<200 total 4h trades) to minimize fee drag.
# Works in bull/bear markets by following 1d structure while using 4h breaks for precise entries.

name = "4h_1d_Multi_Timeframe_Structure_Breakout"
timeframe = "4h"
leverage = 1.0

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
    
    # Volume spike: >1.5x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for market structure (swing points)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d swing points: Higher Highs (HH), Lower Lows (LL)
    # Swing high: current high > previous high AND current high > next high
    # Swing low: current low < previous low AND current low < next low
    swing_high = np.zeros(len(high_1d), dtype=bool)
    swing_low = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(1, len(high_1d)-1):
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1]:
            swing_high[i] = True
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1]:
            swing_low[i] = True
    
    # Determine trend structure: HH/HL = uptrend, LH/LL = downtrend
    # We'll track the last swing high and low to determine structure
    last_swing_high = np.full(len(high_1d), np.nan)
    last_swing_low = np.full(len(low_1d), np.nan)
    
    last_high_val = np.nan
    last_low_val = np.nan
    
    for i in range(len(high_1d)):
        if swing_high[i]:
            last_high_val = high_1d[i]
        if swing_low[i]:
            last_low_val = low_1d[i]
        last_swing_high[i] = last_high_val
        last_swing_low[i] = last_low_val
    
    # Determine market structure: 
    # Uptrend: making higher highs and higher lows
    # Downtrend: making lower highs and lower lows
    # We'll use the relationship between current price and last swing points
    structure_long = np.zeros(len(high_1d), dtype=bool)   # Bullish structure
    structure_short = np.zeros(len(high_1d), dtype=bool)  # Bearish structure
    
    for i in range(len(high_1d)):
        if not np.isnan(last_swing_high[i]) and not np.isnan(last_swing_low[i]):
            # Bullish structure: price above last swing low and making higher highs
            if close_1d[i] > last_swing_low[i]:
                structure_long[i] = True
            # Bearish structure: price below last swing high and making lower lows
            if close_1d[i] < last_swing_high[i]:
                structure_short[i] = True
    
    # Align 1d structure to 4h timeframe
    structure_long_aligned = align_htf_to_ltf(prices, df_1d, structure_long)
    structure_short_aligned = align_htf_to_ltf(prices, df_1d, structure_short)
    
    # 4h swing points for entry timing
    swing_high_4h = np.zeros(len(high), dtype=bool)
    swing_low_4h = np.zeros(len(low), dtype=bool)
    
    for i in range(1, len(high)-1):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            swing_high_4h[i] = True
        if low[i] < low[i-1] and low[i] < low[i+1]:
            swing_low_4h[i] = True
    
    # Calculate 4h swing high and low levels for breakout entries
    last_swing_high_4h = np.full(len(high), np.nan)
    last_swing_low_4h = np.full(len(low), np.nan)
    
    last_high_4h = np.nan
    last_low_4h = np.nan
    
    for i in range(len(high)):
        if swing_high_4h[i]:
            last_high_4h = high[i]
        if swing_low_4h[i]:
            last_low_4h = low[i]
        last_swing_high_4h[i] = last_high_4h
        last_swing_low_4h[i] = last_low_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(structure_long_aligned[i]) or
            np.isnan(structure_short_aligned[i]) or
            np.isnan(last_swing_high_4h[i]) or
            np.isnan(last_swing_low_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish 1d structure + price breaks above 4h swing high + volume spike
            if (structure_long_aligned[i] and 
                close[i] > last_swing_high_4h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish 1d structure + price breaks below 4h swing low + volume spike
            elif (structure_short_aligned[i] and 
                  close[i] < last_swing_low_4h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 4h swing low OR 1d structure turns bearish
            if (close[i] < last_swing_low_4h[i]) or \
               not structure_long_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 4h swing high OR 1d structure turns bullish
            if (close[i] > last_swing_high_4h[i]) or \
               not structure_short_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 12:08
