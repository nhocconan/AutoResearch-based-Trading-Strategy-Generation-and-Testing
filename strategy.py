#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Weekly Pivot Range Breakout with Volume Confirmation
# Hypothesis: Price breaking out of weekly pivot ranges with volume confirmation
# captures institutional moves. Weekly pivot points act as key support/resistance
# levels where price often accelerates after breaking through. Works in both
# bull and bear markets as breakouts occur in any regime. Volume filter ensures
# only significant breaks are traded, reducing false signals. Target: 15-25 trades/year.

name = "1d_weekly_pivot_range_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points and ranges
    # Pivot = (H + L + C) / 3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Calculate weekly range
    weekly_range = weekly_high - weekly_low
    
    # Calculate pivot support/resistance levels
    # R1 = 2*P - L
    # S1 = 2*P - H
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly levels to daily
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    weekly_range_aligned = align_htf_to_ltf(prices, df_weekly, weekly_range)
    
    # Volume confirmation - 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to pivot or below S1
            if close[i] <= pivot_aligned[i] or close[i] < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price returns to pivot or above R1
            if close[i] >= pivot_aligned[i] or close[i] > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long breakout: price breaks above R1 with volume
            if close[i] > r1_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below S1 with volume
            elif close[i] < s1_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals