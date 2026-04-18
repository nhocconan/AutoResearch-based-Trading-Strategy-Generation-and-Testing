#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_Trend_v3
Hypothesis: 6-hour breakouts above/below weekly Donchian channels with weekly pivot direction filter and volume confirmation.
Designed for low trade frequency (target: 12-37/year) with strong performance in both bull and bear markets.
Weekly pivots provide institutional reference points, Donchian channels capture breakouts, and volume confirms strength.
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
    
    # Calculate weekly Donchian channel (20-period)
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Donchian channels: 20-period high/low
    donchian_high = np.full(len(high_weekly), np.nan)
    donchian_low = np.full(len(low_weekly), np.nan)
    for i in range(20, len(high_weekly)):
        donchian_high[i] = np.max(high_weekly[i-20:i+1])
        donchian_low[i] = np.min(low_weekly[i-20:i+1])
    
    # Align weekly Donchian to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Calculate weekly pivot points (using weekly OHLC)
    close_weekly = df_weekly['close'].values
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3
    weekly_range = high_weekly - low_weekly
    # Weekly R1 and S1 (standard pivot)
    weekly_r1 = 2 * weekly_pivot - low_weekly
    weekly_s1 = 2 * weekly_pivot - high_weekly
    
    # Align weekly pivot levels to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly Donchian high with volume spike and above weekly pivot
            if (close[i] > donchian_high_aligned[i] and vol_spike[i] and 
                close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low with volume spike and below weekly pivot
            elif (close[i] < donchian_low_aligned[i] and vol_spike[i] and 
                  close[i] < weekly_pivot_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly Donchian low or below weekly S1
            if (close[i] < donchian_low_aligned[i] or close[i] < weekly_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly Donchian high or above weekly R1
            if (close[i] > donchian_high_aligned[i] or close[i] > weekly_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_Trend_v3"
timeframe = "6h"
leverage = 1.0