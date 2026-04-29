#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot structure filter and volume confirmation
# Long when price breaks above Donchian(20) high AND weekly pivot shows bullish bias (price above weekly CPR pivot) AND volume spike
# Short when price breaks below Donchian(20) low AND weekly pivot shows bearish bias (price below weekly CPR pivot) AND volume spike
# Exit when price re-enters Donchian channel or weekly bias flips
# Donchian provides objective breakout levels, weekly CPR (Central Pivot Range) gives institutional structure,
# volume confirmation ensures breakout validity
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_Donchian20_WeeklyCPR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate weekly CPR (Central Pivot Range) from prior week
    # CPR = [BC, TC] where BC = (weekly_low + weekly_close)/2, TC = (pivot + BC)/2, pivot = (weekly_high + weekly_low + weekly_close)/3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_bc = (weekly_low + weekly_close) / 2.0  # Bottom of CPR
    weekly_tc = (weekly_pivot + weekly_bc) / 2.0   # Top of CPR
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_bc_aligned = align_htf_to_ltf(prices, df_1w, weekly_bc)
    weekly_tc_aligned = align_htf_to_ltf(prices, df_1w, weekly_tc)
    
    # Calculate Donchian(20) channels
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_window, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_bc_aligned[i]) or 
            np.isnan(weekly_tc_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_weekly_pivot = weekly_pivot_aligned[i]
        curr_weekly_bc = weekly_bc_aligned[i]
        curr_weekly_tc = weekly_tc_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Determine weekly bias: bullish if close > weekly TC, bearish if close < weekly BC
        weekly_bullish = curr_close > curr_weekly_tc
        weekly_bearish = curr_close < curr_weekly_bc
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel OR weekly bias turns bearish
            if curr_close <= curr_donchian_high or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel OR weekly bias turns bullish
            if curr_close >= curr_donchian_low or not weekly_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND weekly bullish bias AND volume spike
            if (curr_high > curr_donchian_high and 
                weekly_bullish and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND weekly bearish bias AND volume spike
            elif (curr_low < curr_donchian_low and 
                  weekly_bearish and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals