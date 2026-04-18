#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-day pivot direction and volume confirmation
# Williams %R(14) identifies overbought/oversold conditions
# Long when %R crosses above -80 from below with price above 1-day pivot point and volume > 1.5x average
# Short when %R crosses below -20 from above with price below 1-day pivot point and volume > 1.5x average
# Exit when %R crosses back through -50 (center) in opposite direction
# Uses 1-day pivot for trend filter to avoid counter-trend trades
# Designed for ~20-40 trades/year on 6h timeframe
name = "6h_WilliamsR_Pivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for pivot points
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate pivot points from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Align pivot to 6h timeframe (use previous day's pivot)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Williams %R(14) calculation
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        wr_val = williams_r[i]
        pivot_val = pivot_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below with price above pivot and volume
            if i > 0 and williams_r[i-1] <= -80 and wr_val > -80 and close_val > pivot_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above with price below pivot and volume
            elif i > 0 and williams_r[i-1] >= -20 and wr_val < -20 and close_val < pivot_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses back below -50
            if wr_val < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses back above -50
            if wr_val > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals