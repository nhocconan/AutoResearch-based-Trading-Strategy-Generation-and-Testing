#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal + weekly pivot filter + volume confirmation.
# Williams %R(14) identifies oversold/overbought conditions (< -80 oversold, > -20 overbought).
# Weekly pivot acts as trend filter: only take long signals when price above weekly pivot,
# short signals when price below weekly pivot.
# Volume spike (>1.5x 20-period average) confirms reversal momentum.
# Works in ranging markets (mean reversion at extremes) and trending markets (pullbacks to pivot).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "6h_WilliamsR_WeeklyPivot_Volume"
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
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: (H+L+C)/3
    weekly_pivot = (df_weekly['high'] + df_weekly['low'] + df_weekly['close']) / 3
    weekly_pivot_values = weekly_pivot.values
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot_values)
    
    # Calculate Williams %R on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot_val = weekly_pivot_aligned[i]
        wr_val = williams_r[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price above weekly pivot AND volume spike
            if wr_val < -80 and close_val > pivot_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below weekly pivot AND volume spike
            elif wr_val > -20 and close_val < pivot_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns above -50 (momentum fading) OR price below weekly pivot
            if wr_val > -50 or close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns below -50 (momentum fading) OR price above weekly pivot
            if wr_val < -50 or close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals