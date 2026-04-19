# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly pivot levels (R1/S1) breakout with volume confirmation
# - Uses weekly Pivot Points (R1, S1) calculated from prior week's OHLC
# - Long when price breaks above weekly R1 with volume confirmation
# - Short when price breaks below weekly S1 with volume confirmation
# - Volume filter: daily volume > 1.5x 20-day average
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by capturing breakouts
# - Target: 10-25 trades/year to avoid excessive fee drag
# - Weekly pivot levels provide structural support/resistance levels

name = "1d_WeeklyPivot_R1S1_Breakout_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to daily timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 20-day average volume
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume average
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current daily volume > 1.5x 20-day average
        volume_filter = vol_ma_20_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for long entry: price breaks above weekly R1 with volume
            if close[i] > weekly_r1_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below weekly S1 with volume
            elif close[i] < weekly_s1_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below weekly pivot or opposite S1
            if close[i] < weekly_pivot[i] if not np.isnan(weekly_pivot[i]) else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above weekly pivot or opposite R1
            if close[i] > weekly_pivot[i] if not np.isnan(weekly_pivot[i]) else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals