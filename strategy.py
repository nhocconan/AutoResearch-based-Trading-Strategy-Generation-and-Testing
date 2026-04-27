#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    upper_20w = np.full(len(close_1w), np.nan)
    lower_20w = np.full(len(close_1w), np.nan)
    for i in range(19, len(close_1w)):
        upper_20w[i] = np.max(high_1w[i-19:i+1])
        lower_20w[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate weekly ATR (14-period) for stop loss
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], 
                       np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                                  np.abs(low_1w[1:] - close_1w[:-1])))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = np.full(len(tr_1w), np.nan)
    for i in range(14, len(tr_1w)):
        if i == 14:
            atr_1w[i] = np.mean(tr_1w[1:15])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Align weekly indicators to daily
    upper_20w_aligned = align_htf_to_ltf(prices, df_1w, upper_20w)
    lower_20w_aligned = align_htf_to_ltf(prices, df_1w, lower_20w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Daily volume filter: current volume > 2x 20-day average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly indicators and volume MA
    start_idx = max(19, 14, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_20w_aligned[i]) or np.isnan(lower_20w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper with volume confirmation
            if (price > upper_20w_aligned[i] and 
                vol_ratio > 2.0):
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly Donchian lower with volume confirmation
            elif (price < lower_20w_aligned[i] and 
                  vol_ratio > 2.0):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly Donchian lower or ATR-based stop
            if (price < lower_20w_aligned[i] or 
                price < close[i-1] - 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly Donchian upper or ATR-based stop
            if (price > upper_20w_aligned[i] or 
                price > close[i-1] + 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0