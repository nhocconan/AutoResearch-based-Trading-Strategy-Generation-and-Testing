#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Range_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-day) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], 
                 abs(high_1d[i] - close_1d[i-1]), 
                 abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = tr if i == 1 else (atr_1d[i-1] * 13 + tr) / 14
    
    sum_atr_14 = np.zeros(len(close_1d))
    for i in range(14, len(close_1d)):
        sum_atr_14[i] = np.sum(atr_1d[i-13:i+1])
    
    highest_high_14 = np.zeros(len(close_1d))
    lowest_low_14 = np.zeros(len(close_1d))
    for i in range(14, len(close_1d)):
        highest_high_14[i] = np.max(high_1d[i-13:i+1])
        lowest_low_14[i] = np.min(low_1d[i-13:i+1])
    
    chop = np.full(len(close_1d), 50.0)
    for i in range(14, len(close_1d)):
        if highest_high_14[i] != lowest_low_14[i]:
            chop[i] = 100 * np.log10(sum_atr_14[i] / (highest_high_14[i] - lowest_low_14[i])) / np.log10(14)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Bollinger Bands (20, 2) on 4h close
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 for Bollinger Bands and volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(chop_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: price at lower BB in choppy market (chop > 61.8)
            if close[i] <= bb_low and chop_val > 61.8 and vol > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price at upper BB in choppy market (chop > 61.8)
            elif close[i] >= bb_up and chop_val > 61.8 and vol > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches middle BB or chop drops below 38.2 (trending)
            if close[i] >= bb_middle[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches middle BB or chop drops below 38.2 (trending)
            if close[i] <= bb_middle[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals