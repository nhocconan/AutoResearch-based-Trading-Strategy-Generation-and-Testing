#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Chop_Donchian_Volume_Signal"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily high/low for chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range for chop calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - low_1d[:-1])
    tr3 = np.abs(low_1d[1:] - high_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # ATR(14) for chop denominator
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily range for chop numerator
    daily_range = high_1d - low_1d
    
    # Chop calculation: sum of TR(14) / sum of ranges(14) * 100
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    sum_range_14 = pd.Series(daily_range).rolling(window=14, min_periods=14).sum().values
    chop = 100 * sum_tr_14 / sum_range_14
    
    # Chop aligned to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Chop > 61.8 indicates ranging market (mean reversion)
            if chop_aligned[i] > 61.8:
                # Long at Donchian lower band
                if close[i] <= low_20[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short at Donchian upper band
                elif close[i] >= high_20[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            # Chop < 38.2 indicates trending market (breakout)
            elif chop_aligned[i] < 38.2:
                # Long on upward breakout
                if close[i] > high_20[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short on downward breakout
                elif close[i] < low_20[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian middle or chop signals range
            mid_20 = (high_20[i] + low_20[i]) / 2
            if close[i] < mid_20 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian middle or chop signals range
            mid_20 = (high_20[i] + low_20[i]) / 2
            if close[i] > mid_20 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals