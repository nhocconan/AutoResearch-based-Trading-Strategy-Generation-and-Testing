#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d Williams %R extreme filter and volume confirmation
# Williams %R < -80 (oversold) for longs, > -20 (overbought) for shorts on 1d timeframe.
# Only take breakouts in the direction of the 1d extreme to catch reversals from exhaustion.
# Volume confirmation (1.8x 20-period average) ensures participation.
# Discrete sizing 0.25 targets ~60-120 trades over 4 years (15-30/year).
# Works in bull/bear markets by fading extremes during strong momentum.

name = "6h_Camarilla_R3S3_Breakout_1dWilliamsR_Extreme_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d: (highest_high - close) / (highest_high - lowest_low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate Camarilla levels from previous day (using 1d OHLC)
    # Camarilla: R3 = close + 1.125*(high-low), S3 = close - 1.125*(high-low)
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_1d_close + 1.125 * (prev_1d_high - prev_1d_low)
    camarilla_s3 = prev_1d_close - 1.125 * (prev_1d_high - prev_1d_low)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > R3 with 1d oversold (Williams %R < -80)
            long_breakout = close[i] > camarilla_r3_aligned[i]
            williams_oversold = williams_r_aligned[i] < -80
            
            # Short breakdown: price < S3 with 1d overbought (Williams %R > -20)
            short_breakout = close[i] < camarilla_s3_aligned[i]
            williams_overbought = williams_r_aligned[i] > -20
            
            if long_breakout and williams_oversold and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and williams_overbought and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < S3 or Williams %R returns to neutral (> -50)
            if close[i] < camarilla_s3_aligned[i] or williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > R3 or Williams %R returns to neutral (< -50)
            if close[i] > camarilla_r3_aligned[i] or williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals