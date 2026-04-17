#!/usr/bin/env python3
"""
6h_MarketProfile_ValueArea_Breakout
Strategy: Breakout from previous day's value area with volume confirmation.
Long: Close breaks above previous day's Value Area High (VAH) + volume > 2x average
Short: Close breaks below previous day's Value Area Low (VAL) + volume > 2x average
Exit: Close returns to Point of Control (POC)
Based on Market Profile concepts - institutional traders often defend value area.
Works in both trending and ranging markets by capturing breakouts from balanced areas.
Timeframe: 6h
"""

import numpy as np
import pandas as pd
from scipy import stats
from mtf_data import get_httf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Market Profile calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Value Area (POC, VAH, VAL) for each day
    # Using volume-weighted histogram approximation
    poc = np.full(len(df_1d), np.nan)
    vah = np.full(len(df_1d), np.nan)
    val = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        day_high = df_1d['high'].iloc[i]
        day_low = df_1d['low'].iloc[i]
        day_close = df_1d['close'].iloc[i]
        day_volume = df_1d['volume'].iloc[i]
        
        if day_high <= day_low:
            continue
            
        # Create price bins (100 bins between high and low)
        bins = np.linspace(day_low, day_high, 101)
        # Approximate volume distribution - use typical distribution
        # Higher volume near close, lower at extremes
        weights = np.exp(-0.5 * ((bins - day_close) / (day_high - day_low))**2)
        weights = weights / weights.sum() * day_volume
        
        # Find POC (price with maximum volume)
        poc_idx = np.argmax(weights[:-1])  # exclude last bin edge
        poc[i] = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Calculate Value Area (70% of volume around POC)
        sorted_idx = np.argsort(weights[:-1])[::-1]  # sort by volume descending
        cum_vol = 0
        target_vol = day_volume * 0.7
        va_bins = []
        
        for idx in sorted_idx:
            cum_vol += weights[idx]
            va_bins.append(idx)
            if cum_vol >= target_vol:
                break
        
        if va_bins:
            va_low = bins[min(va_bins)]
            va_high = bins[max(va_bins) + 1]
            val[i] = va_low
            vah[i] = va_high
        else:
            val[i] = day_low
            vah[i] = day_high
    
    # Align daily POC, VAH, VAL to 6h timeframe
    poc_aligned = align_htf_to_ltf(prices, df_1d, poc)
    vah_aligned = align_htf_to_ltf(prices, df_1d, vah)
    val_aligned = align_htf_to_ltf(prices, df_1d, val)
    
    # Volume confirmation (20-period MA on 6h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(poc_aligned[i]) or np.isnan(vah_aligned[i]) or 
            np.isnan(val_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Entry conditions
        if position == 0:
            # Long: Close breaks above VAH + volume
            if close[i] > vah_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below VAL + volume
            elif close[i] < val_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Exit long: Close returns to POC
            if close[i] < poc_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close returns to POC
            if close[i] > poc_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_MarketProfile_ValueArea_Breakout"
timeframe = "6h"
leverage = 1.0