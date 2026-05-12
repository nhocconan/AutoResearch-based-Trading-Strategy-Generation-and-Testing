#!/usr/bin/env python3
# 6h_12h_1d_MarketStructure_Breakout
# Hypothesis: Combines 1d market structure (HH/HL/LH/LL) with 6h breakouts for trend-following entries.
# Uses 1d swing points to determine trend direction, and breaks of 6h swing highs/lows for entry timing.
# Volume confirmation (>1.5x 20-period average) filters for institutional participation.
# Designed for low trade frequency (<150 total 6h trades) to minimize fee drag.
# Works in bull/bear markets by following 1d structure while using 6h breaks for precise entries.

name = "6h_12h_1d_MarketStructure_Breakout"
timeframe = "6h"
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
    
    # Volume spike: >1.5x 20-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 12h data for market structure (swing points)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h swing points: Higher Highs (HH), Lower Lows (LL)
    # Swing high: current high > previous high AND current high > next high
    # Swing low: current low < previous low AND current low < next low
    swing_high = np.zeros(len(high_12h), dtype=bool)
    swing_low = np.zeros(len(low_12h), dtype=bool)
    
    for i in range(1, len(high_12h)-1):
        if high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i+1]:
            swing_high[i] = True
        if low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i+1]:
            swing_low[i] = True
    
    # Determine trend structure: HH/HL = uptrend, LH/LL = downtrend
    # We'll track the last swing high and low to determine structure
    last_swing_high = np.full(len(high_12h), np.nan)
    last_swing_low = np.full(len(low_12h), np.nan)
    
    last_high_val = np.nan
    last_low_val = np.nan
    
    for i in range(len(high_12h)):
        if swing_high[i]:
            last_high_val = high_12h[i]
        if swing_low[i]:
            last_low_val = low_12h[i]
        last_swing_high[i] = last_high_val
        last_swing_low[i] = last_low_val
    
    # Determine market structure: 
    # Uptrend: making higher highs and higher lows
    # Downtrend: making lower highs and lower lows
    # We'll use the relationship between current price and last swing points
    structure_long = np.zeros(len(high_12h), dtype=bool)   # Bullish structure
    structure_short = np.zeros(len(high_12h), dtype=bool)  # Bearish structure
    
    for i in range(len(high_12h)):
        if not np.isnan(last_swing_high[i]) and not np.isnan(last_swing_low[i]):
            # Bullish structure: price above last swing low and making higher highs
            if close_12h[i] > last_swing_low[i]:
                structure_long[i] = True
            # Bearish structure: price below last swing high and making lower lows
            if close_12h[i] < last_swing_high[i]:
                structure_short[i] = True
    
    # Align 12h structure to 6h timeframe
    structure_long_aligned = align_htf_to_ltf(prices, df_12h, structure_long)
    structure_short_aligned = align_htf_to_ltf(prices, df_12h, structure_short)
    
    # 6h swing points for entry timing
    swing_high_6h = np.zeros(len(high), dtype=bool)
    swing_low_6h = np.zeros(len(low), dtype=bool)
    
    for i in range(1, len(high)-1):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            swing_high_6h[i] = True
        if low[i] < low[i-1] and low[i] < low[i+1]:
            swing_low_6h[i] = True
    
    # Calculate 6h swing high and low levels for breakout entries
    last_swing_high_6h = np.full(len(high), np.nan)
    last_swing_low_6h = np.full(len(low), np.nan)
    
    last_high_6h = np.nan
    last_low_6h = np.nan
    
    for i in range(len(high)):
        if swing_high_6h[i]:
            last_high_6h = high[i]
        if swing_low_6h[i]:
            last_low_6h = low[i]
        last_swing_high_6h[i] = last_high_6h
        last_swing_low_6h[i] = last_low_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(structure_long_aligned[i]) or
            np.isnan(structure_short_aligned[i]) or
            np.isnan(last_swing_high_6h[i]) or
            np.isnan(last_swing_low_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish 12h structure + price breaks above 6h swing high + volume spike
            if (structure_long_aligned[i] and 
                close[i] > last_swing_high_6h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish 12h structure + price breaks below 6h swing low + volume spike
            elif (structure_short_aligned[i] and 
                  close[i] < last_swing_low_6h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 6h swing low OR 12h structure turns bearish
            if (close[i] < last_swing_low_6h[i]) or \
               not structure_long_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 6h swing high OR 12h structure turns bullish
            if (close[i] > last_swing_high_6h[i]) or \
               not structure_short_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals