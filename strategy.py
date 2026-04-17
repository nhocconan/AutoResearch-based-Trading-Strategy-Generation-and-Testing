#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_TrendFilter
Hypothesis: Camarilla pivot R1/S1 breakouts with volume confirmation and 12h trend filter
capture institutional breakouts in both bull and bear markets. Uses 1d pivots for structure,
volume for confirmation, and 12h EMA34 for trend alignment. Low-frequency, high-conviction
trades to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    rng = high_1d - low_1d
    r1 = close_1d + 1.1 * rng / 12
    s1 = close_1d - 1.1 * rng / 12
    
    # Align 1d pivot levels to 4h timeframe (already uses previous day's close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 34)  # volume MA20, EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        if position == 0:
            # Long: break above R1 + volume filter + 12h uptrend
            if close[i] > r1_aligned[i] and volume_filter and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + volume filter + 12h downtrend
            elif close[i] < s1_aligned[i] and volume_filter and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: break below S1 or 12h trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: break above R1 or 12h trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0