#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Williams %R with Volume Confirmation
# Hypothesis: Williams %R identifies overbought/oversold conditions on weekly timeframe,
# providing early reversal signals in both bull and bear markets. Volume confirmation
# ensures institutional participation, filtering out false signals. Weekly timeframe
# reduces noise from 6h fluctuations while capturing major trend reversals.
# Target: 15-35 trades/year (60-140 over 4 years).

name = "6h_weekly_williamsr_volume_v1"
timeframe = "6h"
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
    
    # Get weekly data for Williams %R calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate rolling max/min for Williams %R
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    highest_high = weekly_high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = weekly_low_series.rolling(window=14, min_periods=14).min().values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, 
                          ((highest_high - weekly_close) / denominator) * -100, 
                          -50)  # Neutral value when no range
    
    # Shift by 1 to use only completed weekly bars (avoid look-ahead)
    williams_r = np.roll(williams_r, 1)
    if len(williams_r) > 1:
        williams_r[0] = williams_r[1]
    else:
        williams_r[0] = -50
    
    # Align weekly Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_weekly, williams_r)
    
    # Volume filter: volume > 1.5x 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R becomes overbought (> -20) or volume filter fails
            if williams_r_aligned[i] > -20 or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: Williams %R becomes oversold (< -80) or volume filter fails
            if williams_r_aligned[i] < -80 or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: Williams %R oversold (< -80) with volume confirmation
            if williams_r_aligned[i] < -80 and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) with volume confirmation
            elif williams_r_aligned[i] > -20 and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals