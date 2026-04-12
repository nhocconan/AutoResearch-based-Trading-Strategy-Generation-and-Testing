#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter
    # Uses 1w Camarilla levels (H4/L4) as institutional breakout zones
    # Volume > 2.0x 20-period MA confirms institutional participation
    # Target: 12-30 trades/year (50-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla H4 and L4 levels (strong breakout levels)
    camarilla_h4 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_l4 = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align to 6h timeframe (use previous week's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Donchian channel (20-period) on 6h timeframe
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 2.0x 20-period MA
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine breakout direction based on weekly pivot
        bullish_bias = close[i] > (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
        bearish_bias = close[i] < (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Entry conditions: Donchian breakout with volume confirmation and weekly bias
        long_entry = breakout_up and (vol_ratio[i] > 2.0) and bullish_bias
        short_entry = breakout_down and (vol_ratio[i] > 2.0) and bearish_bias
        
        # Exit conditions: opposite Donchian breakout or midpoint reversion
        midpoint = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        opposite_breakout = (position == 1 and breakout_down) or (position == -1 and breakout_up)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (long_exit or opposite_breakout):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (short_exit or opposite_breakout):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_donchian_breakout_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0