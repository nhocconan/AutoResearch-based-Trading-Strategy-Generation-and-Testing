#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-week trend filter and daily pivot reversal
# - 1-week high/low defines trend (long when price above weekly midpoint, short when below)
# - Daily pivot points (R1/S1) for mean-reversion entries in ranging markets
# - Volume confirmation: 6h volume > 1.5x 20-period average
# - Exit on opposite pivot level or trend reversal
# - Designed for mean reversion in ranges (2022-2024, 2025) and trend following in breaks
# - Target: 15-25 trades/year to avoid excessive fee drag

name = "6h_WeeklyTrend_DailyPivot_Reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1-week high/low for trend definition
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_mid = (weekly_high + weekly_low) / 2
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    
    # Daily pivot calculation (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 6h volume average (20-period)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (00:00-23:00 UTC - trade all hours for 6h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    # For 6h timeframe, trade during active markets: 00-06, 06-12, 12-18, 18-24 UTC
    # Simple approach: trade all hours but avoid extreme lows if needed
    active_hours = ((hours >= 0) & (hours <= 23))  # Trade all hours for simplicity
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_mid_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x average
        volume_filter = vol_ma_6h[i] > 0 and volume[i] > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Look for long entry: price near S1 support in uptrend or above weekly mid
            if ((close[i] <= s1_aligned[i] * 1.005 and close[i] >= s1_aligned[i] * 0.995) or 
                close[i] > weekly_mid_aligned[i]) and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price near R1 resistance in downtrend or below weekly mid
            elif ((close[i] >= r1_aligned[i] * 0.995 and close[i] <= r1_aligned[i] * 1.005) or 
                  close[i] < weekly_mid_aligned[i]) and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit at R1 resistance or trend reversal below weekly mid
            if close[i] >= r1_aligned[i] * 0.995 or close[i] < weekly_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit at S1 support or trend reversal above weekly mid
            if close[i] <= s1_aligned[i] * 1.005 or close[i] > weekly_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals