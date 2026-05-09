#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Choppiness_Filter_Donchian20_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Choppiness Index on weekly data (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR14
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.nan
        elif i == 14:
            atr[i] = np.nanmean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of True Range over 14 periods
    sum_tr14 = np.zeros_like(atr)
    for i in range(len(sum_tr14)):
        if i < 14:
            sum_tr14[i] = np.nan
        else:
            sum_tr14[i] = np.nansum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = np.full_like(high_1w, np.nan)
    lowest_low_14 = np.full_like(low_1w, np.nan)
    for i in range(len(highest_high_14)):
        if i < 13:
            highest_high_14[i] = np.nan
            lowest_low_14[i] = np.nan
        else:
            highest_high_14[i] = np.nanmax(high_1w[i-13:i+1])
            lowest_low_14[i] = np.nanmin(low_1w[i-13:i+1])
    
    # Choppiness Index
    chop = np.full_like(close_1w, 50.0)
    for i in range(len(chop)):
        if np.isnan(sum_tr14[i]) or np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or highest_high_14[i] == lowest_low_14[i]:
            chop[i] = 50.0
        else:
            chop[i] = 100 * np.log10(sum_tr14[i] / (highest_high_14[i] - lowest_low_14[i])) / np.log10(14)
    
    # Donchian channels on daily data (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = np.full_like(high_1d, np.nan)
    donchian_low = np.full_like(low_1d, np.nan)
    for i in range(len(donchian_high)):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.nanmax(high_1d[i-19:i+1])
            donchian_low[i] = np.nanmin(low_1d[i-19:i+1])
    
    # Align all to 12h
    chop_12h = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chop_12h[i]) or np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_value = chop_12h[i]
        upper_band = donchian_high_12h[i]
        lower_band = donchian_low_12h[i]
        
        if position == 0:
            # Enter long when choppy market (range-bound) and price breaks above upper Donchian
            if chop_value > 61.8 and close[i] > upper_band:
                signals[i] = 0.25
                position = 1
            # Enter short when choppy market and price breaks below lower Donchian
            elif chop_value > 61.8 and close[i] < lower_band:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when market starts trending or price breaks below lower band
            if chop_value < 38.2 or close[i] < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when market starts trending or price breaks above upper band
            if chop_value < 38.2 or close[i] > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals