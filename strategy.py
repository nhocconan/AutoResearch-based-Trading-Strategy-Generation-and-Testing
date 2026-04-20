#!/usr/bin/env python3
# 6h_1d_Pivot_R4S4_Breakout_VolumeTrend
# Hypothesis: Trade momentum breakouts from 1d R4/S4 levels (strong breakout levels) on 6h timeframe with volume confirmation and trend filter.
# Uses ADX(14) > 20 to ensure trending market and avoid whipsaws in ranging conditions.
# Focuses on strong breaks with volume surge to capture institutional participation.
# Designed for 12-37 trades per year by requiring multiple confirmations.

name = "6h_1d_Pivot_R4S4_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d R4 and S4 levels using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R4 and S4 (strong breakout levels)
    s4_1d = close_1d - (range_1d * 1.5)
    r4_1d = close_1d + (range_1d * 1.5)
    
    # Align 1d levels to 6h timeframe
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX for trend filter (14-period)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s4_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(adx[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 20 (trending market)
        if adx[i] < 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R4 with volume surge
            if (close[i] > r4_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S4 with volume surge
            elif (close[i] < s4_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S4 or ADX drops below 15 (trend weakening)
            if close[i] < s4_aligned[i] * 0.997 or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R4 or ADX drops below 15 (trend weakening)
            if close[i] > r4_aligned[i] * 1.003 or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals