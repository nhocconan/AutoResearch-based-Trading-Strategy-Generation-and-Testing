#!/usr/bin/env python3
# 6h_Liquidity_Sweep_Reversal_12hTrend
# Hypothesis: Price sweeps liquidity (equal highs/lows) then reverses in direction of 12h trend.
# Liquidity sweeps occur when price briefly breaks recent swing points but lacks follow-through.
# In trending markets (12h EMA50), these often precede strong continuation moves.
# Uses volume spike to confirm institutional participation. Works in both bull/bear markets
# by aligning with higher timeframe trend.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "6h_Liquidity_Sweep_Reversal_12hTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for trend filter and swing points
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema50_12h[i] = (close_12h[i] * 2 + ema50_12h[i-1] * 48) / 50
    
    # Align 12h EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate swing points (pivot highs/lows) on 12h
    # Swing high: high > previous 2 highs and next 2 highs
    # Swing low: low < previous 2 lows and next 2 lows
    swing_high_12h = np.full_like(high_12h, np.nan)
    swing_low_12h = np.full_like(low_12h, np.nan)
    
    for i in range(2, len(df_12h) - 2):
        if (high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i-2] and
            high_12h[i] > high_12h[i+1] and high_12h[i] > high_12h[i+2]):
            swing_high_12h[i] = high_12h[i]
        if (low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i-2] and
            low_12h[i] < low_12h[i+1] and low_12h[i] < low_12h[i+2]):
            swing_low_12h[i] = low_12h[i]
    
    # Align swing points to 6h timeframe
    swing_high_12h_aligned = align_htf_to_ltf(prices, df_12h, swing_high_12h)
    swing_low_12h_aligned = align_htf_to_ltf(prices, df_12h, swing_low_12h)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 12h EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(swing_high_12h_aligned[i]) or 
            np.isnan(swing_low_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        trend_up = close[i] > ema50_12h_aligned[i]
        
        if position == 0:
            # Liquidity sweep conditions:
            # Bullish sweep: price makes new 12h swing high but closes below it (trapped longs)
            # Bearish sweep: price makes new 12h swing low but closes above it (trapped shorts)
            is_swing_high = not np.isnan(swing_high_12h_aligned[i])
            is_swing_low = not np.isnan(swing_low_12h_aligned[i])
            
            bullish_sweep = is_swing_high and high[i] >= swing_high_12h_aligned[i] and close[i] < swing_high_12h_aligned[i]
            bearish_sweep = is_swing_low and low[i] <= swing_low_12h_aligned[i] and close[i] > swing_low_12h_aligned[i]
            
            # Enter in direction of 12h trend after liquidity sweep
            if bullish_sweep and trend_up and volume_ratio[i] > 1.8:
                signals[i] = 0.25
                position = 1
            elif bearish_sweep and not trend_up and volume_ratio[i] > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend turns down or opposite sweep occurs
            if not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend turns up or opposite sweep occurs
            if trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals