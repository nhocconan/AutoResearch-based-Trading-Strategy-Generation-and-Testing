#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume_v2"
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
    
    # Get 12h data for trend filter (EMA20) and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate Camarilla levels from previous 12h candle (R1 and S1)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Shift to get previous 12h period's values
    high_12h_shifted = np.roll(high_12h, 1)
    low_12h_shifted = np.roll(low_12h, 1)
    close_12h_shifted = np.roll(close_12h, 1)
    
    # Calculate Camarilla width for R1/S1: (H-L)*1.1/12
    camarilla_width = (high_12h_shifted - low_12h_shifted) * 1.1 / 12
    r1 = close_12h_shifted + camarilla_width  # R1 level
    s1 = close_12h_shifted - camarilla_width  # S1 level
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Calculate volume confirmation (current volume vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 level, uptrend (price > EMA20), volume confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_20_12h_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 level, downtrend (price < EMA20), volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_20_12h_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 level (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above R1 level (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals