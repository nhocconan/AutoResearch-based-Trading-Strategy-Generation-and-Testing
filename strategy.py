#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_Volume_Regime_Filtered_v1
Hypothesis: Breakout of daily Camarilla R1/S1 levels with volume confirmation and chop regime filter.
Long when price > daily R1 + volume spike + chop > 61.8 (range).
Short when price < daily S1 + volume spike + chop > 61.8 (range).
Exit when price crosses daily pivot point (PP).
Uses 12h as primary timeframe for signal generation, with 1d for levels.
Target: 15-35 trades/year per symbol. Works in range markets via chop filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, and pivot point (PP)
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate Chop Index on daily data
    hl_range = high_1d - low_1d
    atr14 = pd.Series(hl_range).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(atr14 / np.nansum(hl_range[-14:]) * np.sqrt(14)) / np.log10(np.sqrt(14))
    chop = np.where(np.isnan(chop), 50, chop)  # Fill NaN with neutral value
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Chop regime filter: chop > 61.8 indicates ranging market
        chop_ok = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long conditions: break above R1 + volume + chop range
            if price > r1_aligned[i] and volume_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume + chop range
            elif price < s1_aligned[i] and volume_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_Regime_Filtered_v1"
timeframe = "12h"
leverage = 1.0