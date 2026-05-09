# 6H_WeeklyPivot_DailyTrend_VolumeBreakout
# Weekly pivot levels from 1-week high/low, with daily trend filter and volume confirmation
# Long: break above weekly high + daily uptrend + volume spike
# Short: break below weekly low + daily downtrend + volume spike
# Uses 1-week data for pivot points and 1-day for trend filter
# Designed to work in both bull and bear markets by following weekly structure
# Target: 20-50 trades per year to minimize fee drag

#!/usr/bin/env python3
name = "6H_WeeklyPivot_DailyTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot points (weekly high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly high and low (pivot levels)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and daily data to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2x 24-period average volume (4 days)
        avg_volume = np.mean(volume[max(0, i-24):i])
        volume_confirm = volume[i] > avg_volume * 2.0
        
        if position == 0:
            # Enter long: break above weekly high + daily uptrend + volume confirmation
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: break below weekly low + daily downtrend + volume confirmation
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: break below weekly low or lose daily uptrend
            if (close[i] < weekly_low_aligned[i] or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: break above weekly high or lose daily downtrend
            if (close[i] > weekly_high_aligned[i] or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals