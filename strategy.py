#!/usr/bin/env python3
# 1h_4d_Structure_Reversion
# Hypothesis: In 1h timeframe, use 4h and 1d structure to identify trend exhaustion points for mean-reversion entries.
# In strong trends (defined by 1d structure), wait for 4h pullbacks to enter in trend direction.
# Uses volume confirmation to avoid false signals. Designed for low trade frequency (15-37/year) to minimize fee drag.
# Works in bull/bear markets by following higher timeframe structure while using 1h for precise reversion entries.

name = "1h_4d_Structure_Reversion"
timeframe = "1h"
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
    
    # Session filter: 8-20 UTC (already datetime64, no conversion needed)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    # 4h trend structure (direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h swing points
    swing_high_4h = np.zeros(len(high_4h), dtype=bool)
    swing_low_4h = np.zeros(len(low_4h), dtype=bool)
    
    for i in range(1, len(high_4h)-1):
        if high_4h[i] > high_4h[i-1] and high_4h[i] > high_4h[i+1]:
            swing_high_4h[i] = True
        if low_4h[i] < low_4h[i-1] and low_4h[i] < low_4h[i+1]:
            swing_low_4h[i] = True
    
    # Track last swing points for structure
    last_swing_high_4h = np.full(len(high_4h), np.nan)
    last_swing_low_4h = np.full(len(low_4h), np.nan)
    
    last_high_4h = np.nan
    last_low_4h = np.nan
    
    for i in range(len(high_4h)):
        if swing_high_4h[i]:
            last_high_4h = high_4h[i]
        if swing_low_4h[i]:
            last_low_4h = low_4h[i]
        last_swing_high_4h[i] = last_high_4h
        last_swing_low_4h[i] = last_low_4h
    
    # Determine 4h trend structure
    structure_long_4h = np.zeros(len(high_4h), dtype=bool)   # Bullish: HH/HL
    structure_short_4h = np.zeros(len(high_4h), dtype=bool)  # Bearish: LH/LL
    
    for i in range(len(high_4h)):
        if not np.isnan(last_swing_high_4h[i]) and not np.isnan(last_swing_low_4h[i]):
            # Bullish structure: price above last swing low and making higher highs
            if close_4h[i] > last_swing_low_4h[i]:
                structure_long_4h[i] = True
            # Bearish structure: price below last swing high and making lower lows
            if close_4h[i] < last_swing_high_4h[i]:
                structure_short_4h[i] = True
    
    # Align 4h structure to 1h timeframe
    structure_long_4h_aligned = align_htf_to_ltf(prices, df_4h, structure_long_4h)
    structure_short_4h_aligned = align_htf_to_ltf(prices, df_4h, structure_short_4h)
    
    # 1h pullback identification (for entry timing)
    # Calculate 1h RSI for pullback detection
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Oversold/overbought levels for pullback entries
    rsi_oversold = 30
    rsi_overbought = 70
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if (np.isnan(structure_long_4h_aligned[i]) or
            np.isnan(structure_short_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish 4h structure + 1h RSI oversold pullback + volume confirmation
            if (structure_long_4h_aligned[i] and 
                rsi[i] < rsi_oversold and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Bearish 4h structure + 1h RSI overbought pullback + volume confirmation
            elif (structure_short_4h_aligned[i] and 
                  rsi[i] > rsi_overbought and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral OR 4h structure turns bearish
            if (rsi[i] > 50) or \
               not structure_long_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral OR 4h structure turns bullish
            if (rsi[i] < 50) or \
               not structure_short_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals