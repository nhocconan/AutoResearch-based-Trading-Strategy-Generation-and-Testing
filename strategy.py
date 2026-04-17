#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_VolumeFilter
Strategy: 12-hour Camarilla pivot breakout with volume confirmation and weekly trend filter.
Long: Price breaks above weekly S1 + volume > 1.5x 24-period average + price above weekly EMA34
Short: Price breaks below weekly R1 + volume > 1.5x 24-period average + price below weekly EMA34
Exit: Price returns to weekly pivot point
Position size: 0.25
Designed to capture breakouts aligned with weekly trend in both bull and bear markets.
Timeframe: 12h
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
    
    # Calculate weekly Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1 = pivot + (range_1w * 1.1 / 12)
    s1 = pivot - (range_1w * 1.1 / 12)
    pp = pivot  # pivot point
    
    # Calculate weekly EMA34 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation (24-period MA on 12h)
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # wait for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_ma24[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 24-period average
        volume_filter = volume[i] > (1.5 * volume_ma24[i])
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema34_1w_aligned[i]
        price_below_ema = close[i] < ema34_1w_aligned[i]
        
        # Breakout conditions
        breakout_above_s1 = close[i] > s1_aligned[i-1]  # break above previous period S1
        breakout_below_r1 = close[i] < r1_aligned[i-1]  # break below previous period R1
        
        # Return to pivot point
        return_to_pivot = abs(close[i] - pp_aligned[i]) < 0.1 * (r1_aligned[i] - s1_aligned[i])
        
        if position == 0:
            # Long: breakout above S1 + volume filter + price above EMA
            if breakout_above_s1 and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: breakout below R1 + volume filter + price below EMA
            elif breakout_below_r1 and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to pivot or break below S1
            if return_to_pivot or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to pivot or break above R1
            if return_to_pivot or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0