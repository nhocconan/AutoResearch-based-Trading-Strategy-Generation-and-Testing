#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Refined"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter (EMA50) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d candle (R1 and S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous 1d period's values
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d, 1)
    
    # Calculate Camarilla width for R1/S1: (H-L)*1.1/12
    camarilla_width = (high_1d_shifted - low_1d_shifted) * 1.1 / 12
    r1 = close_1d_shifted + camarilla_width  # R1 level
    s1 = close_1d_shifted - camarilla_width  # S1 level
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate volume confirmation (current volume vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    # Add chop filter using 1d data (more stable)
    # Calculate True Range for chop
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Chop = (sum of true ranges over period) / (max(high) - min(low)) * 100
    # Higher = more choppy, lower = more trending
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.sum(tr) / (highest_high - lowest_low)  # Simplified for alignment
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ratio[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 level, uptrend (price > EMA50), volume confirmation, not too choppy
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 2.0 and
                chop_aligned[i] < 61.8):  # Trending regime (chop < 61.8)
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 level, downtrend (price < EMA50), volume confirmation, not too choppy
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 2.0 and
                  chop_aligned[i] < 61.8):  # Trending regime
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 level (reversal signal) or chop increases significantly
            if close[i] < s1_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 level (reversal signal) or chop increases significantly
            if close[i] > r1_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals