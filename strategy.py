#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Get 1w data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
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
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate volume confirmation (current volume vs 50-period average)
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_ratio = volume / vol_ma50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 level, uptrend (price > 1w EMA20), volume confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_ratio[i] > 2.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 level, downtrend (price < 1w EMA20), volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_ratio[i] > 2.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 level (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 level (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals