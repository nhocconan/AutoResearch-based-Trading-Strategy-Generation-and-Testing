#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Chandelier_Exit_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    # Calculate daily ATR(22) for Chandelier Exit
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr22_1d = pd.Series(tr).rolling(window=22, min_periods=22).mean().values
    atr22_1d_aligned = align_htf_to_ltf(prices, df_1d, atr22_1d)
    
    # Calculate 22-period highest high and lowest low for Chandelier Exit
    highest_high_22 = pd.Series(high_1d).rolling(window=22, min_periods=22).max().values
    lowest_low_22 = pd.Series(low_1d).rolling(window=22, min_periods=22).min().values
    highest_high_22_aligned = align_htf_to_ltf(prices, df_1d, highest_high_22)
    lowest_low_22_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_22)
    
    # Chandelier Exit: long exit = highest_high - 3*ATR, short exit = lowest_low + 3*ATR
    chandelier_long_exit = highest_high_22_aligned - 3.0 * atr22_1d_aligned
    chandelier_short_exit = lowest_low_22_aligned + 3.0 * atr22_1d_aligned
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chandelier_long_exit[i]) or np.isnan(chandelier_short_exit[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        long_exit = chandelier_long_exit[i]
        short_exit = chandelier_short_exit[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Enter long: price above long exit + volume filter
            if close[i] > long_exit and vol_filt:
                signals[i] = 0.25
                position = 1
            # Enter short: price below short exit + volume filter
            elif close[i] < short_exit and vol_filt:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below long exit
            if close[i] < long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above short exit
            if close[i] > short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals