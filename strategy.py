#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Camarilla R1/S1 breakout + volume confirmation + trend filter (price > 4h EMA50).
Long when price breaks above 1d Camarilla R1 with volume confirmation and price > 4h EMA50 (uptrend).
Short when price breaks below 1d Camarilla S1 with volume confirmation and price < 4h EMA50 (downtrend).
Exit when price returns to the 1d Camarilla midpoint (R1+S1)/2.
Uses 1d timeframe for structure (reduces noise) and 4h for entry timing and trend filter.
Camarilla pivots work well in ranging markets and capture institutional levels.
Volume confirmation avoids false breakouts.
Designed for 75-150 trades/year with discrete sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R1, S1, midpoint)
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    # Mid = (R1 + S1)/2 = Close
    rng = high_1d - low_1d
    r1_1d = close_1d + 1.1 * rng / 12.0
    s1_1d = close_1d - 1.1 * rng / 12.0
    mid_1d = close_1d  # (R1 + S1)/2 simplifies to close
    
    # Calculate 4h EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    mid_1d_aligned = align_htf_to_ltf(prices, df_1d, mid_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(mid_1d_aligned[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R1 with volume and uptrend (price > EMA50)
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S1 with volume and downtrend (price < EMA50)
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below 1d Camarilla midpoint
            if close[i] <= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above 1d Camarilla midpoint
            if close[i] >= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dCamarilla_R1S1_Breakout_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0