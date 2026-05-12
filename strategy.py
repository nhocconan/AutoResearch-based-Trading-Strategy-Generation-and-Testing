#!/usr/bin/env python3
# 6h_1w_1d_Structure_Breakout
# Hypothesis: Combines 1-week market structure (higher highs/lows) with 1-day mean reversion and 6h breakouts for entries.
# Uses 1w swing points to determine long-term trend direction, 1d RSI for mean-reversion timing,
# and breaks of 6h swing highs/lows for entry. Volume confirmation (>1.5x 20-period average) filters for institutional participation.
# Designed for low trade frequency (<150 total 6h trades) to minimize fee drift.
# Works in bull markets by following 1w structure; in bear markets by fading 1d extremes against the trend.

name = "6h_1w_1d_Structure_Breakout"
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
    
    # Weekly data for long-term structure (swing points)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w swing points: Higher Highs (HH), Lower Lows (LL)
    swing_high_1w = np.zeros(len(high_1w), dtype=bool)
    swing_low_1w = np.zeros(len(low_1w), dtype=bool)
    
    for i in range(1, len(high_1w)-1):
        if high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i+1]:
            swing_high_1w[i] = True
        if low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i+1]:
            swing_low_1w[i] = True
    
    # Determine 1w trend structure: HH/HL = uptrend, LH/LL = downtrend
    last_swing_high_1w = np.full(len(high_1w), np.nan)
    last_swing_low_1w = np.full(len(low_1w), np.nan)
    
    last_high_1w = np.nan
    last_low_1w = np.nan
    
    for i in range(len(high_1w)):
        if swing_high_1w[i]:
            last_high_1w = high_1w[i]
        if swing_low_1w[i]:
            last_low_1w = low_1w[i]
        last_swing_high_1w[i] = last_high_1w
        last_swing_low_1w[i] = last_low_1w
    
    # Determine market structure: 
    # Uptrend: price above last swing low and making higher highs
    # Downtrend: price below last swing high and making lower lows
    structure_long_1w = np.zeros(len(high_1w), dtype=bool)   # Bullish structure
    structure_short_1w = np.zeros(len(high_1w), dtype=bool)  # Bearish structure
    
    for i in range(len(high_1w)):
        if not np.isnan(last_swing_high_1w[i]) and not np.isnan(last_swing_low_1w[i]):
            # Bullish structure: price above last swing low
            if close_1w[i] > last_swing_low_1w[i]:
                structure_long_1w[i] = True
            # Bearish structure: price below last swing high
            if close_1w[i] < last_swing_high_1w[i]:
                structure_short_1w[i] = True
    
    # Align 1w structure to 6h timeframe
    structure_long_1w_aligned = align_htf_to_ltf(prices, df_1w, structure_long_1w)
    structure_short_1w_aligned = align_htf_to_ltf(prices, df_1w, structure_short_1w)
    
    # Daily data for mean reversion (RSI)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Align 1d RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
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
        if (np.isnan(structure_long_1w_aligned[i]) or
            np.isnan(structure_short_1w_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(last_swing_high_6h[i]) or
            np.isnan(last_swing_low_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish 1w structure + 1d RSI < 30 (oversold) + price breaks above 6h swing high + volume spike
            if (structure_long_1w_aligned[i] and 
                rsi_1d_aligned[i] < 30 and 
                close[i] > last_swing_high_6h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish 1w structure + 1d RSI > 70 (overbought) + price breaks below 6h swing low + volume spike
            elif (structure_short_1w_aligned[i] and 
                  rsi_1d_aligned[i] > 70 and 
                  close[i] < last_swing_low_6h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 6h swing low OR 1w structure turns bearish OR RSI > 70 (overbought)
            if (close[i] < last_swing_low_6h[i]) or \
               not structure_long_1w_aligned[i] or \
               rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 6h swing high OR 1w structure turns bullish OR RSI < 30 (oversold)
            if (close[i] > last_swing_high_6h[i]) or \
               not structure_short_1w_aligned[i] or \
               rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals