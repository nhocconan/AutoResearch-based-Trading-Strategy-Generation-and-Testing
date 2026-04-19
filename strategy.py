#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with volume confirmation
# - Weekly pivot sets trend: price > weekly pivot = bullish bias, < = bearish bias
# - Daily Camarilla R1/S1 levels: break above R1 with volume = long, break below S1 with volume = short
# - Volume filter: current 6h volume > 1.5x 20-period average for conviction
# - Exit on opposite Camarilla level (R2/S2) or trend reversal
# - Position size: 0.25 to manage drawdown
# - Designed to work in both bull and bear markets by combining weekly trend with daily breakouts
# - Target: 15-35 trades/year to avoid excessive fee drag

name = "6h_Camarilla_R1_S1_Breakout_Volume_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot point (standard calculation)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Daily Camarilla levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla formulas
    daily_range = daily_high - daily_low
    camarilla_r1 = daily_close + daily_range * 1.1 / 12
    camarilla_s1 = daily_close - daily_range * 1.1 / 12
    camarilla_r2 = daily_close + daily_range * 1.1 / 6
    camarilla_s2 = daily_close - daily_range * 1.1 / 6
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Volume filter: current 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_r2_aligned[i]) or 
            np.isnan(camarilla_s2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for long entry: bullish weekly trend + price breaks above R1 + volume
            if (close[i] > weekly_pivot_aligned[i] and 
                close[i] > camarilla_r1_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: bearish weekly trend + price breaks below S1 + volume
            elif (close[i] < weekly_pivot_aligned[i] and 
                  close[i] < camarilla_s1_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on R2 break or trend reversal
            if (close[i] > camarilla_r2_aligned[i] or 
                close[i] < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on S2 break or trend reversal
            if (close[i] < camarilla_s2_aligned[i] or 
                close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals