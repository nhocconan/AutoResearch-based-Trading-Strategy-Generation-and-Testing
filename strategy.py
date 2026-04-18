#!/usr/bin/env python3
"""
1d Pivot Reversal with Volume Spike and Weekly Trend Filter
Uses daily Camarilla pivot levels with volume spike confirmation and weekly EMA trend filter.
Designed for low trade frequency (target: 20-30 trades/year) with strong reversal edges at key levels.
Works in both bull and bear markets by fading extremes in direction of weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R3 = C + (H-L)*1.1/2, R4 = C + (H-L)*1.1
    # S3 = C - (H-L)*1.1/2, S4 = C - (H-L)*1.1
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + rng * 1.1 / 2
    camarilla_r4 = close_1d + rng * 1.1
    camarilla_s3 = close_1d - rng * 1.1 / 2
    camarilla_s4 = close_1d - rng * 1.1
    
    # Align Camarilla levels to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: price rejects S3/S4 with volume spike and above weekly EMA
            if ((price <= s3 or price <= s4) and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: price rejects R3/R4 with volume spike and below weekly EMA
            elif ((price >= r3 or price >= r4) and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price reaches opposite pivot or trend reversal
            if price >= r3:  # Take profit at R3
                signals[i] = 0.0
                position = 0
            elif price < ema_trend:  # Trend reversal
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price reaches opposite pivot or trend reversal
            if price <= s3:  # Take profit at S3
                signals[i] = 0.0
                position = 0
            elif price > ema_trend:  # Trend reversal
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Pivot_Reversal_Volume_Spike_WeeklyTrend"
timeframe = "1d"
leverage = 1.0