#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot points (HIGH/LOW/CLOSE)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points using previous week's data
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values (shifted by 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume filter: volume > 1.8x 20-period average (6h periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Break above R3 with volume spike (strong bullish breakout)
        # 2. Bounce from S1/S2 with volume spike (bullish reversal)
        long_breakout = (close[i] > r3_6h[i]) and volume_spike[i]
        long_bounce = ((close[i] > s1_6h[i] and close[i] < s2_6h[i]) or 
                       (close[i] > s2_6h[i] and close[i] < pivot_6h[i])) and volume_spike[i]
        
        # Short conditions:
        # 1. Break below S3 with volume spike (strong bearish breakdown)
        # 2. Rejection from R1/R2 with volume spike (bearish reversal)
        short_breakdown = (close[i] < s3_6h[i]) and volume_spike[i]
        short_rejection = ((close[i] < r1_6h[i] and close[i] > pivot_6h[i]) or 
                           (close[i] < r2_6h[i] and close[i] > r1_6h[i])) and volume_spike[i]
        
        if long_breakout or long_bounce:
            signals[i] = 0.25
            position = 1
        elif short_breakdown or short_rejection:
            signals[i] = -0.25
            position = -1
        # Exit conditions: return to opposite pivot level
        elif position == 1 and close[i] < s1_6h[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_6h[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0