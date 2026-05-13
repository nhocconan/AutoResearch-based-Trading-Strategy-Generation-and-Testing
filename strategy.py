#!/usr/bin/env python3
# 6h_MultiTF_Structure_Trend
# Hypothesis: Combine 12h market structure (HH/HL or LH/LL) with 60-period EMA trend on 6h and volume confirmation.
# In bull markets: long when 12h structure is bullish (HH/HL), price above 60 EMA, and volume spike.
# In bear markets: short when 12h structure is bearish (LH/LL), price below 60 EMA, and volume spike.
# Uses market structure for trend direction (more reliable than single MA), EMA for dynamic support/resistance,
# and volume to confirm institutional interest. Designed for low frequency (est. 20-40 trades/year) to minimize fee drag.

name = "6h_MultiTF_Structure_Trend"
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

    # Get 12h data for market structure analysis (swing points)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate swing highs and lows on 12h
    # Swing high: high > previous high and high > next high
    # Swing low: low < previous low and low < next low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Initialize swing arrays
    swing_high = np.full_like(high_12h, np.nan)
    swing_low = np.full_like(low_12h, np.nan)
    
    # Find swing points (need 3 points: prev, curr, next)
    for i in range(1, len(high_12h) - 1):
        if high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i+1]:
            swing_high[i] = high_12h[i]
        if low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i+1]:
            swing_low[i] = low_12h[i]
    
    # Determine market structure: bullish (HH/HL) or bearish (LH/LL)
    # Track last two swing points
    structure = np.full_like(high_12h, 0)  # 1: bullish, -1: bearish, 0: unclear
    last_swing_high = np.nan
    last_swing_low = np.nan
    last_swing_type = 0  # 1: high, -1: low
    
    for i in range(len(high_12h)):
        if not np.isnan(swing_high[i]):
            if not np.isnan(last_swing_high) and high_12h[i] > last_swing_high:
                # Higher high
                if last_swing_type == -1 and not np.isnan(last_swing_low) and low_12h[i] > last_swing_low:
                    # Also higher low -> bullish structure
                    structure[i] = 1
                else:
                    structure[i] = 0  # Need both HH and HL
            else:
                structure[i] = 0
            last_swing_high = high_12h[i]
            last_swing_type = 1
        elif not np.isnan(swing_low[i]):
            if not np.isnan(last_swing_low) and low_12h[i] < last_swing_low:
                # Lower low
                if last_swing_type == 1 and not np.isnan(last_swing_high) and high_12h[i] < last_swing_high:
                    # Also lower high -> bearish structure
                    structure[i] = -1
                else:
                    structure[i] = 0  # Need both LL and LH
            else:
                structure[i] = 0
            last_swing_low = low_12h[i]
            last_swing_type = -1
    
    # Align 12h structure to 6h
    structure_12h_aligned = align_htf_to_ltf(prices, df_12h, structure)
    
    # Get 60-period EMA on 6h for dynamic support/resistance
    ema60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume spike: volume > 2.0 * 20-period average (~6.7 hours worth at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after EMA warmup
        # Skip if any required value is NaN
        if (np.isnan(structure_12h_aligned[i]) or 
            np.isnan(ema60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish 12h structure + price above EMA60 + volume spike
            if structure_12h_aligned[i] == 1 and close[i] > ema60[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish 12h structure + price below EMA60 + volume spike
            elif structure_12h_aligned[i] == -1 and close[i] < ema60[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Structure turns bearish OR price closes below EMA60
            if structure_12h_aligned[i] == -1 or close[i] < ema60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Structure turns bullish OR price closes above EMA60
            if structure_12h_aligned[i] == 1 or close[i] > ema60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals